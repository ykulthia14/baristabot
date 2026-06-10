# baristabot.py — Production-clean BaristaBot backend
# All top-level invoke() calls removed; module is safe to import.

# ── Imports ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterable
from random import randint
from typing import Annotated, Literal

import resend
from square import Square
from square.environment import SquareEnvironment
from square.core.api_error import ApiError

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

resend.api_key = os.environ.get("RESEND_API_KEY")

if not resend.api_key:

    raise ValueError("RESEND_API_KEY not found. Please check your .env file.")


SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN")
SQUARE_APP_ID = os.environ.get("SQUARE_APP_ID")
SQUARE_LOCATION_ID = os.environ.get("SQUARE_LOCATION_ID")

if not SQUARE_ACCESS_TOKEN or not SQUARE_APP_ID or not SQUARE_LOCATION_ID:
    raise ValueError(
        "SQUARE_ACCESS_TOKEN, SQUARE_APP_ID, and SQUARE_LOCATION_ID must be set in the set in your .env file."
    )

_square_client = Square(
    token=SQUARE_ACCESS_TOKEN,
    environment=SquareEnvironment.SANDBOX,
)


def charge_square(nonce: str, amount_cents: int) -> str:
    """Charge a card using a Square Web Payments SDK nonce."""
    try:
        result = _square_client.payments.create(
            source_id=nonce,
            idempotency_key=str(uuid.uuid4()),
            amount_money={
                "amount": amount_cents,
                "currency": "USD",
            },
            location_id=SQUARE_LOCATION_ID,
        )
        
        # Access the ID via dot-notation
        return f"payment_success:{result.payment.id}"
        
    except ApiError as e:
        # Extract the error details from the exception
        error_details = "; ".join([err.detail for err in e.errors]) if hasattr(e, 'errors') and e.errors else str(e)
        return f"payment_failed:{error_details}"
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "orders@yourdomain.com")


# ── State ────────────────────────────────────────────────────────────────────
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

class OrderItem(TypedDict):
    description: str  
    price: float


# ── Constants ────────────────────────────────────────────────────────────────
BARISTABOT_SYSINT = (
    "system",
    "You are a BaristaBot, an interactive cafe ordering system. "
    "First, you must politely ask the customer for their name and email address. "
    "Once the customer provides their name and email, you must call the get_full_menu tool and display the entire menu to them, asking what they would like to order. "
    "You will answer any questions about menu items (and only about menu items - no off-topic discussion). "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user). "
    "Always confirm_order with the user (double-check) before finalizing. Calling "
    "confirm_order will display the order items to the user and returns their response. "
    "Once the customer has finished ordering items and you have called confirm_order to ensure it is correct, "
    "call pay_order to collect payment. Only after pay_order succeeds, call place_order_and_email with the customer's name and email address. "
    "Once place_order_and_email has returned, thank the user, confirm their bill was sent to their email, and say goodbye!"
)

WELCOME_MSG = "Welcome to the BaristaBot cafe! Before we get started, could I please get your name and email address?"

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

# ── Menu vector store ────────────────────────────────────────────────────────
MENU_DOCS = [
    "Coffee Drinks (no milk): Espresso, Americano, Cold Brew.",
    "Coffee Drinks with Milk: Latte, Cappuccino, Cortado, Macchiato, Mocha, Flat White.",
    "Tea Drinks (no milk): English Breakfast Tea, Green Tea, Earl Grey.",
    "Tea Drinks with Milk: Chai Latte, Matcha Latte, London Fog.",
    "Other Drinks: Steamer, Hot Chocolate.",
    "Milk options: Whole (default), 2%, Oat, Almond, 2% Lactose Free. NOTE: Soy milk is out of stock today.",
    "Espresso shots: Single, Double (default), Triple, Quadruple.",
    "Caffeine options: Regular (default), Decaf.",
    "Temperature: Hot (default), Iced.",
    "Sweeteners (add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, "
    "chocolate sauce, sugar free vanilla sweetener. 'Sweetened' means plain sugar.",
    "Special requests: any reasonable modifier not involving off-menu items, e.g. extra hot, "
    "one pump, half caff, extra foam.",
    "'Dirty' means add a shot of espresso to a drink that does not normally contain it, "
    "e.g. Dirty Chai Latte.",
    "'Regular milk' is the same as whole milk.",
]

MENU_PRICES: dict[str, float] = {
    # Coffee — no milk
    "espresso": 2.50,
    "americano": 3.00,
    "cold brew": 4.00,
    # Coffee — with milk
    "latte": 4.50,
    "cappuccino": 4.50,
    "cortado": 4.00,
    "macchiato": 4.00,
    "mocha": 5.00,
    "flat white": 4.50,
    # Tea — no milk
    "english breakfast tea": 2.50,
    "green tea": 2.50,
    "earl grey": 2.50,
    # Tea — with milk
    "chai latte": 4.50,
    "matcha latte": 5.00,
    "london fog": 4.50,
    # Other
    "steamer": 3.50,
    "hot chocolate": 4.00,
}

def lookup_price(drink: str) -> float:
    """Return the price for a drink, or 0.00 if not found."""
    return MENU_PRICES.get(drink.strip().lower(), 0.00)
_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

menu_vectorstore = Chroma.from_texts(
    texts=MENU_DOCS,
    embedding=_embeddings,
    collection_name="cafe_menu",
)
menu_retriever = menu_vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Auto-executed tools (handled by ToolNode) ────────────────────────────────
@tool
def get_menu(query: str) -> str:
    """Retrieve relevant menu sections for a given customer query.

    Use this whenever the customer asks about available drinks, modifiers,
    milk options, or anything else menu-related.

    Args:
        query: The customer's question or the item they mentioned.
    Returns:
        Relevant menu sections as a single string.
    """
    docs = menu_retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)
@tool
def get_full_menu() -> str:
    """Retrieve the entire cafe menu. Use this immediately after getting the user's name and email."""
    return "\n".join(MENU_DOCS)


# ── Order tools (handled by order_node) ─────────────────────────────────────
@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
    """Adds the specified drink to the customer's order, including any modifiers.

    Args:
        drink: Name of the drink to add.
        modifiers: List of modifier strings (e.g. ["oat milk", "iced"]).
    Returns:
        The current list of items in the order as a string.
    """
    # Logic lives in order_node; this stub satisfies bind_tools schema generation.
    raise NotImplementedError("Handled by order_node")


@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct.

    Returns:
        The user's free-text response.
    """
    raise NotImplementedError("Handled by order_node")


@tool
def get_order() -> str:
    """Returns the user's order so far. One item per line."""
    raise NotImplementedError("Handled by order_node")


@tool
def clear_order() -> str:
    """Removes all items from the user's order."""
    raise NotImplementedError("Handled by order_node")


@tool
def pay_order() -> str:
    """Collects payment from the customer for the current order.

    Call this after confirm_order and before place_order_and_email.



    Returns:
        'payment_success' if payment was completed, or an error message.
    """
    raise NotImplementedError("Handled by order_node")

@tool
def place_order_and_email(name: str, email: str) -> int:
    """Sends the order to the barista and emails the bill to the customer.

    Args:
        name: Customer's name.
        email: Customer's email address.
    Returns:
        The estimated number of minutes until the order is ready.
    """
    raise NotImplementedError("Handled by order_node")


# ── Tool sets ────────────────────────────────────────────────────────────────
auto_tools = [get_menu, get_full_menu]
order_tools = [add_to_order, confirm_order, get_order, clear_order, pay_order, place_order_and_email]
tool_node = ToolNode(auto_tools)
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


# ── Graph nodes ──────────────────────────────────────────────────────────────
def chatbot_node(state: OrderState) -> OrderState:
    """Main LLM node. Sends welcome message on first turn, otherwise calls the model."""
    defaults: OrderState = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    return defaults | state | {"messages": [new_output]}


def human_node(state: OrderState) -> OrderState:
    """Pauses execution and waits for user input via LangGraph interrupt().

    In the Streamlit UI, app.py resumes this node with Command(resume=user_text).
    """
    user_input: str = interrupt("Waiting for user input")

    finished = user_input.strip().lower() in {"q", "quit", "exit", "goodbye"}
    return state | {"messages": [("user", user_input)], "finished": finished}


def order_node(state: OrderState) -> OrderState:
    """Handles all order-management tool calls (add, confirm, get, clear, place)."""
    tool_msg = state.get("messages", [])[-1]
    raw_order = state.get("order", [])
    order: list[OrderItem] = [
        item if isinstance(item, dict) else {"description": item, "price": 0.00}
        for item in raw_order
    ]
    outbound_msgs: list[ToolMessage] = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        name = tool_call["name"]

        if name == "add_to_order":
            drink = tool_call["args"]["drink"]
            modifiers = tool_call["args"].get("modifiers", [])
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"
            price = lookup_price(drink)
            order.append({"description": f"{drink} ({modifier_str})", "price": price})
            response = "\n".join(f"{item['description']} (${item['price']:.2f})" for item in order)

        elif name == "confirm_order":
            order_summary = "\n".join(f"{item['description']} (${item['price']:.2f})" for item in order) if order else "(no items)"
            response = interrupt(f"Your order:\n{order_summary}\n\nIs this correct?")

        elif name == "get_order":
            response = "\n".join(f"{item['description']} (${item['price']:.2f})" for item in order) if order else "(no order)"

        elif name == "clear_order":
            order.clear()
            response = "Order cleared."

        elif name == "pay_order":
            total = sum(item["price"] for item in order)
            # Interrupt: the frontend will show Square's card fields, tokenize,

            # call charge_square(), then resume with the result string.
            user_response = interrupt(
                f"💳 Your total is **${total:.2f}**. "
                "Please complete payment with card details."
            )
            # user_response will be "payment_success:<square_payment_id>" or "payment_failed:<error_message>"
            if str(user_response).startswith("payment_success"):
                response = "payment_success"
            else:
                response = f"Payment failed: {user_response}"
            

        elif name == "place_order_and_email":
            order_placed = True
            customer_name = tool_call["args"].get("name", "Customer")
            customer_email = tool_call["args"].get("email", "")

            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", customer_email):
                response = (
                    f"Error: '{customer_email}' is not a valid email address. "
                    "Please ask the customer to confirm their email and try again."
                )
            else:
                eta = randint(1, 5)
                total = sum(item["price"] for item in order)
                html_rows = "".join(
                    "<tr>"
                    f"<td style='padding:4px 16px 4px 0'>{item['description']}</td>"
                    f"<td style='text-align:right'>${item['price']:.2f}</td>"
                    "</tr>"
                    for item in order
                ) if order else "<tr><td>(no items)</td><td></td></tr>"
                html_rows += (
                    "<tr style='border-top:2px solid #333'>"
                    "<td style='padding-top:6px'><strong>Total</strong></td>"
                    f"<td style='text-align:right;padding-top:6px'><strong>${total:.2f}</strong></td>"
                    "</tr>"
                )
                html_body = (
                    "<h2>&#9749; Your BaristaBot Order</h2>"
                    f"<p>Hi {customer_name}, thanks for your order!</p>"
                    "<h3>Order Summary</h3>"
                    f"<table style='font-family:sans-serif;font-size:14px'>{html_rows}</table>"
                    f"<p><strong>Estimated wait:</strong> {eta} minute(s)</p>"
                    "<p>See you soon!</p>"
                )
                text_rows = "\n".join(
                    f"  {item['description']:<40} ${item['price']:.2f}"
                    for item in order
                ) if order else "  (no items)"
                text_rows += f"\n  {'─' * 47}\n  {'Total':<40} ${total:.2f}"
                text_body = (
                    f"Hi {customer_name}, thanks for your order!\n\n"
                    f"Order Summary:\n{text_rows}\n\n"
                    f"Estimated wait: {eta} minute(s)\n\nSee you soon!"
                )
                try:
                    resend.Emails.send({
                        "from": SENDER_EMAIL,
                        "to": customer_email,
                        "subject": "☕ Your BaristaBot Order Confirmation",
                        "html": html_body,
                        "text": text_body,
                    })
                    order_placed = True
                    response = str(eta)
                    print(f"\n[SERVER LOG] -> Email sent to {customer_name} <{customer_email}> for {len(order)} item(s), total ${total:.2f}")
                except Exception as exc:
                    print(f"\n[EMAIL ERROR] -> {exc}") 
                    response = (
                        f"Error: Failed to send confirmation email to {customer_email}. "
                        f"Reason: {exc}. Please inform the customer and try again."
                    )
        else:
            response = (
                f"Error: tool '{name}' does not exist. Please apologize and try again."
            )

        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=name,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_chatbot(state: OrderState) -> str:
    """Route to tools, ordering, human turn, or END after the chatbot speaks."""
    if state.get("finished", False):
        return END

    msgs = state.get("messages", [])
    if not msgs:
        raise ValueError("No messages found in state.")

    last = msgs[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Decide which tool handler owns these calls
        if any(tc["name"] in tool_node.tools_by_name for tc in last.tool_calls):
            return "tools"
        return "ordering"

    return "human"


def route_after_human(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Continue the conversation or end if the user said goodbye."""
    return END if state.get("finished", False) else "chatbot"


# ── Graph assembly ────────────────────────────────────────────────────────────
def _build_graph(checkpointer=None):
    builder = StateGraph(OrderState)

    builder.add_node("chatbot", chatbot_node)
    builder.add_node("human", human_node)
    builder.add_node("tools", tool_node)
    builder.add_node("ordering", order_node)

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", route_after_chatbot)
    builder.add_conditional_edges("human", route_after_human)
    builder.add_edge("tools", "chatbot")
    builder.add_edge("ordering", "chatbot")

    return builder.compile(checkpointer=checkpointer)


# The graph used by app.py (with in-memory persistence)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and _POSTGRES_AVAILABLE:
    connection_pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size = 10,
        open = True,
        kwargs = {"autocommit": True},
    )
    memory = PostgresSaver(connection_pool)
    memory.setup()
    print("Persistence: PostgreSQL")
elif DATABASE_URL and not _POSTGRES_AVAILABLE:
    print(
        "WARNING: DATABASE_URL is set but langgraph-checkpoint-postgres is not installed. "
        "Run: pip install langgraph-checkpoint-postgres psycopg[binary] psycopg-pool"
    )
    memory = MemorySaver()
    print("Persistence: in-memory (fallback)")
else:
    memory = MemorySaver()
    print("Persistence: in-memory (set DATABASE_URL in .env to use PostgreSQL)")

graph_with_persistence = _build_graph(checkpointer=memory)


# ── CLI runner (only runs when executed directly) ─────────────────────────────
def run_session(thread_id: str | None = None) -> str:
    """Start a new CLI session or resume an existing one by thread_id."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"Starting new session. Thread ID: {thread_id}")

    config = {
        "recursion_limit": 100,
        "configurable": {"thread_id": thread_id},
    }

    initial_input: dict | Command = {"messages": []}

    while True:
        for _ in graph_with_persistence.stream(initial_input, config, stream_mode="values"):
            pass

        snapshot = graph_with_persistence.get_state(config)
        if not snapshot.next:
            print("\nSession ended.")
            break

        user_input = input("You: ").strip()
        print()

        if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
            graph_with_persistence.invoke(Command(resume=user_input), config)
            print("\nSession ended.")
            break

        if user_input.lower() == "pause":
            print("\nSession paused. Resume with run_session(thread_id).")
            break

        initial_input = Command(resume=user_input)

    return thread_id


if __name__ == "__main__":
    run_session()


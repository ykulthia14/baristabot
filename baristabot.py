# baristabot.py — BaristaBot backend with Pydantic validation
from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from random import randint
from typing import Annotated, Literal

import resend
from square import Square
from square.environment import SquareEnvironment
from square.core.api_error import ApiError

from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, field_validator
from langchain_core.vectorstores import InMemoryVectorStore
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

# ── Environment ───────────────────────────────────────────────────────────────
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
        "SQUARE_ACCESS_TOKEN, SQUARE_APP_ID, and SQUARE_LOCATION_ID must be set in your .env file."
    )

_square_client = Square(
    token=SQUARE_ACCESS_TOKEN,
    environment=SquareEnvironment.SANDBOX,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class OrderItem(BaseModel):
    """A single validated item in the customer's order."""
    description: str = Field(min_length=1, description="Human-readable drink description")
    price: float = Field(ge=0.0, description="Price in USD")

    @field_validator("price")
    @classmethod
    def round_price(cls, v: float) -> float:
        return round(v, 2)

    def display(self) -> str:
        return f"{self.description} (${self.price:.2f})"


class AddToOrderArgs(BaseModel):
    """Validated arguments for the add_to_order tool call."""
    drink: str = Field(min_length=1)
    modifiers: list[str] = Field(default_factory=list)

    @field_validator("drink")
    @classmethod
    def clean_drink(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("modifiers")
    @classmethod
    def clean_modifiers(cls, v: list[str]) -> list[str]:
        return [m.strip().lower() for m in v if m.strip()]


class PlaceOrderArgs(BaseModel):
    """Validated arguments for the place_order_and_email tool call."""
    name: str = Field(min_length=1)
    email: EmailStr

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip().title()


class PaymentResult(BaseModel):
    """Parsed result from a Square charge attempt."""
    success: bool
    payment_id: str | None = None
    error: str | None = None

    @classmethod
    def from_string(cls, result: str) -> "PaymentResult":
        if result.startswith("payment_success:"):
            return cls(success=True, payment_id=result.split(":", 1)[1])
        elif result.startswith("payment_failed:"):
            return cls(success=False, error=result.split(":", 1)[1])
        return cls(success=False, error=f"Unexpected payment response: {result}")


class OrderSummary(BaseModel):
    """Aggregated order used for display and email."""
    items: list[OrderItem] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(item.price for item in self.items), 2)

    def display_lines(self) -> list[str]:
        return [item.display() for item in self.items]

    def is_empty(self) -> bool:
        return len(self.items) == 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def order_items_from_state(raw: list) -> list[OrderItem]:
    """Deserialize raw state dicts into validated OrderItem models."""
    result = []
    for item in raw:
        if isinstance(item, OrderItem):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(OrderItem(**item))
            except Exception:
                result.append(OrderItem(description=str(item), price=0.0))
    return result


# ── State ─────────────────────────────────────────────────────────────────────
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[dict]   # serialized OrderItem dicts (LangGraph requires plain dicts)
    finished: bool


# ── Constants ─────────────────────────────────────────────────────────────────
BARISTABOT_SYSINT = (
    "system",
    "You are BaristaBot, a friendly cafe ordering assistant. Follow these steps strictly:\n\n"
    "STEP 1 - COLLECT CONTACT INFO:\n"
    "Ask for the customer's name and email address before anything else. "
    "Do not proceed until you have BOTH. If they try to order first, politely remind them you need "
    "their name and email first. If the email looks invalid (no @ or no domain), ask them to confirm it.\n\n"
    "STEP 2 - SHOW MENU:\n"
    "Once you have name and email, call get_full_menu and display the full menu clearly. "
    "Then ask what they'd like to order.\n\n"
    "STEP 3 - TAKE THE ORDER:\n"
    "Add items with add_to_order. Rules:\n"
    "- Soy milk is OUT OF STOCK today. If requested, apologize and suggest an alternative "
    "(oat, almond, whole, 2%, lactose-free).\n"
    "- If a customer asks for something not on the menu, politely say it is not available.\n"
    "- If a customer wants to modify an item already added, call clear_order and re-add all items.\n"
    "- If a customer wants to cancel entirely, confirm then call clear_order and say goodbye.\n\n"
    "STEP 4 - CONFIRM ORDER:\n"
    "Before payment, call get_order to review, then call confirm_order to show the customer. "
    "If the order is empty, do not proceed - ask what they would like. "
    "If the customer wants changes after confirming, call clear_order and re-take the order.\n\n"
    "STEP 5 - PAYMENT:\n"
    "Only call pay_order after the customer confirms the order is correct. "
    "If payment fails, inform the customer and ask them to try again.\n\n"
    "STEP 6 - COMPLETE:\n"
    "Only after pay_order succeeds, call place_order_and_email with the customer's name and email. "
    "Then thank them, confirm the receipt was emailed, and say goodbye.\n\n"
    "GENERAL RULES:\n"
    "- Only answer questions about the menu and current order. For anything else say: "
    "I am here to help with your order - is there anything from our menu I can get for you?\n"
    "- Never make up menu items or prices. Always use lookup_price for pricing.\n"
    "- Be warm, concise, and helpful."
)

WELCOME_MSG = "Welcome to the BaristaBot cafe! Before we get started, could I please get your name and email address?"

SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "orders@yourdomain.com")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

# ── Menu vector store ─────────────────────────────────────────────────────────
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
    "chocolate sauce, sugar free vanilla sweetener. Sweetened means plain sugar.",
    "Special requests: any reasonable modifier not involving off-menu items, e.g. extra hot, "
    "one pump, half caff, extra foam.",
    "Dirty means add a shot of espresso to a drink that does not normally contain it, "
    "e.g. Dirty Chai Latte.",
    "Regular milk is the same as whole milk.",
]

MENU_PRICES: dict[str, float] = {
    "espresso": 2.50, "americano": 3.00, "cold brew": 4.00,
    "latte": 4.50, "cappuccino": 4.50, "cortado": 4.00,
    "macchiato": 4.00, "mocha": 5.00, "flat white": 4.50,
    "english breakfast tea": 2.50, "green tea": 2.50, "earl grey": 2.50,
    "chai latte": 4.50, "matcha latte": 5.00, "london fog": 4.50,
    "steamer": 3.50, "hot chocolate": 4.00,
}

def lookup_price(drink: str) -> float:
    return MENU_PRICES.get(drink.strip().lower(), 0.00)

_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
menu_vectorstore = InMemoryVectorStore.from_texts(
    texts=MENU_DOCS, embedding=_embeddings, collection_name="cafe_menu",
)
menu_retriever = menu_vectorstore.as_retriever(search_kwargs={"k": 3})


# ── Square ────────────────────────────────────────────────────────────────────
def charge_square(nonce: str, amount_cents: int) -> str:
    try:
        result = _square_client.payments.create(
            source_id=nonce,
            idempotency_key=str(uuid.uuid4()),
            amount_money={"amount": amount_cents, "currency": "USD"},
            location_id=SQUARE_LOCATION_ID,
        )
        return f"payment_success:{result.payment.id}"
    except ApiError as e:
        error_details = "; ".join([err.detail for err in e.errors]) if hasattr(e, "errors") and e.errors else str(e)
        return f"payment_failed:{error_details}"


# ── Auto tools ────────────────────────────────────────────────────────────────
@tool
def get_menu(query: str) -> str:
    """Retrieve relevant menu sections for a given customer query.

    Args:
        query: The customer's question or the item they mentioned.
    Returns:
        Relevant menu sections as a single string.
    """
    docs = menu_retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

@tool
def get_full_menu() -> str:
    """Retrieve the entire cafe menu. Use immediately after getting the user's name and email."""
    return "\n".join(MENU_DOCS)


# ── Order tools (stubs — logic lives in order_node) ───────────────────────────
@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
    """Adds the specified drink to the customer's order, including any modifiers.

    Args:
        drink: Name of the drink to add.
        modifiers: List of modifier strings (e.g. ["oat milk", "iced"]).
    Returns:
        The current list of items in the order as a string.
    """
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
        payment_success if completed, or an error message.
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


# ── Tool sets ─────────────────────────────────────────────────────────────────
auto_tools = [get_menu, get_full_menu]
order_tools = [add_to_order, confirm_order, get_order, clear_order, pay_order, place_order_and_email]
tool_node = ToolNode(auto_tools)
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


# ── Graph nodes ───────────────────────────────────────────────────────────────
def chatbot_node(state: OrderState) -> OrderState:
    defaults: OrderState = {"order": [], "finished": False}
    if state["messages"]:
        new_output = llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)
    return defaults | state | {"messages": [new_output]}


def human_node(state: OrderState) -> OrderState:
    user_input: str = interrupt("Waiting for user input")
    finished = user_input.strip().lower() in {"q", "quit", "exit", "goodbye"}
    return state | {"messages": [("user", user_input)], "finished": finished}


def order_node(state: OrderState) -> OrderState:
    tool_msg = state.get("messages", [])[-1]
    order: list[OrderItem] = order_items_from_state(state.get("order", []))
    outbound_msgs: list[ToolMessage] = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        name = tool_call["name"]
        response: str

        # ── add_to_order ──────────────────────────────────────────────────────
        if name == "add_to_order":
            try:
                args = AddToOrderArgs(**tool_call["args"])
            except Exception as exc:
                response = f"Error: invalid add_to_order arguments - {exc}"
                outbound_msgs.append(ToolMessage(content=response, name=name, tool_call_id=tool_call["id"]))
                continue

            # Reject out-of-stock soy milk before it enters the order
            all_text = args.drink + " " + " ".join(args.modifiers)
            if "soy" in all_text:
                response = "Error: Soy milk is out of stock today. Please choose another milk option (oat, almond, whole, 2%, lactose-free)."
                outbound_msgs.append(ToolMessage(content=response, name=name, tool_call_id=tool_call["id"]))
                continue

            price = lookup_price(args.drink)
            modifier_str = ", ".join(args.modifiers) if args.modifiers else "no modifiers"
            item = OrderItem(description=f"{args.drink} ({modifier_str})", price=price)
            order.append(item)
            response = "\n".join(i.display() for i in order)

        # ── confirm_order ─────────────────────────────────────────────────────
        elif name == "confirm_order":
            summary = OrderSummary(items=order)
            if summary.is_empty():
                response = "Error: The order is empty. Please add items before confirming."
                outbound_msgs.append(ToolMessage(content=response, name=name, tool_call_id=tool_call["id"]))
                continue
            lines = "\n".join(summary.display_lines())
            response = interrupt(f"Your order:\n{lines}\nTotal: ${summary.total:.2f}\n\nIs this correct?")

        # ── get_order ─────────────────────────────────────────────────────────
        elif name == "get_order":
            summary = OrderSummary(items=order)
            if summary.is_empty():
                response = "(no items in order)"
            else:
                lines = "\n".join(summary.display_lines())
                response = f"{lines}\nTotal: ${summary.total:.2f}"

        # ── clear_order ───────────────────────────────────────────────────────
        elif name == "clear_order":
            order.clear()
            response = "Order cleared."

        # ── pay_order ─────────────────────────────────────────────────────────
        elif name == "pay_order":
            summary = OrderSummary(items=order)
            if summary.is_empty():
                response = "Error: Cannot process payment for an empty order."
                outbound_msgs.append(ToolMessage(content=response, name=name, tool_call_id=tool_call["id"]))
                continue
            raw_result = interrupt(
                f"Your total is **${summary.total:.2f}**. "
                "Please complete payment with card details."
            )
            payment = PaymentResult.from_string(str(raw_result))
            response = "payment_success" if payment.success else f"Payment failed: {payment.error}"

        # ── place_order_and_email ─────────────────────────────────────────────
        elif name == "place_order_and_email":
            try:
                args = PlaceOrderArgs(**tool_call["args"])
            except Exception as exc:
                response = (
                    f"Error: invalid customer details - {exc}. "
                    "Please ask the customer to confirm their name and email."
                )
                outbound_msgs.append(ToolMessage(content=response, name=name, tool_call_id=tool_call["id"]))
                continue

            eta = randint(1, 5)
            summary = OrderSummary(items=order)

            html_rows = "".join(
                f"<tr><td style='padding:4px 16px 4px 0'>{i.description}</td>"
                f"<td style='text-align:right'>${i.price:.2f}</td></tr>"
                for i in summary.items
            ) or "<tr><td>(no items)</td><td></td></tr>"
            html_rows += (
                "<tr style='border-top:2px solid #333'>"
                f"<td style='padding-top:6px'><strong>Total</strong></td>"
                f"<td style='text-align:right;padding-top:6px'><strong>${summary.total:.2f}</strong></td></tr>"
            )
            html_body = (
                "<h2>&#9749; Your BaristaBot Order</h2>"
                f"<p>Hi {args.name}, thanks for your order!</p>"
                f"<h3>Order Summary</h3>"
                f"<table style='font-family:sans-serif;font-size:14px'>{html_rows}</table>"
                f"<p><strong>Estimated wait:</strong> {eta} minute(s)</p>"
                "<p>See you soon!</p>"
            )
            text_rows = "\n".join(
                f"  {i.description:<40} ${i.price:.2f}" for i in summary.items
            ) or "  (no items)"
            text_rows += f"\n  {'Total':<40} ${summary.total:.2f}"
            text_body = (
                f"Hi {args.name}, thanks for your order!\n\n"
                f"Order Summary:\n{text_rows}\n\n"
                f"Estimated wait: {eta} minute(s)\n\nSee you soon!"
            )

            try:
                resend.Emails.send({
                    "from": SENDER_EMAIL,
                    "to": str(args.email),
                    "subject": "Your BaristaBot Order Confirmation",
                    "html": html_body,
                    "text": text_body,
                })
                order_placed = True
                response = str(eta)
                print(f"[SERVER LOG] Email sent to {args.name} <{args.email}> — {len(summary.items)} item(s), total ${summary.total:.2f}")
            except Exception as exc:
                print(f"[EMAIL ERROR] {exc}")
                response = (
                    f"Error: Failed to send confirmation email to {args.email}. "
                    f"Reason: {exc}. Please inform the customer and try again."
                )

        else:
            response = f"Error: tool '{name}' does not exist. Please apologize and try again."

        outbound_msgs.append(
            ToolMessage(content=response, name=name, tool_call_id=tool_call["id"])
        )

    # Serialize Pydantic models back to plain dicts for LangGraph state storage
    return {
        "messages": outbound_msgs,
        "order": [i.model_dump() for i in order],
        "finished": order_placed,
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_chatbot(state: OrderState) -> str:
    if state.get("finished", False):
        return END
    msgs = state.get("messages", [])
    if not msgs:
        raise ValueError("No messages found in state.")
    last = msgs[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if any(tc["name"] in tool_node.tools_by_name for tc in last.tool_calls):
            return "tools"
        return "ordering"
    return "human"


def route_after_human(state: OrderState) -> Literal["chatbot", "__end__"]:
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


# ── Persistence ───────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and _POSTGRES_AVAILABLE:
    connection_pool = ConnectionPool(
        conninfo=DATABASE_URL, max_size=10, open=True,
        kwargs={"autocommit": True},
    )
    memory = PostgresSaver(connection_pool)
    memory.setup()
    print("Persistence: PostgreSQL")
elif DATABASE_URL and not _POSTGRES_AVAILABLE:
    print("WARNING: DATABASE_URL set but langgraph-checkpoint-postgres not installed.")
    memory = MemorySaver()
    print("Persistence: in-memory (fallback)")
else:
    memory = MemorySaver()
    print("Persistence: in-memory")

graph_with_persistence = _build_graph(checkpointer=memory)


# ── CLI runner ────────────────────────────────────────────────────────────────
def run_session(thread_id: str | None = None) -> str:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"Starting new session. Thread ID: {thread_id}")

    config = {"recursion_limit": 100, "configurable": {"thread_id": thread_id}}
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

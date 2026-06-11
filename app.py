import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv(override=True)

from baristabot import graph_with_persistence

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BaristaBot Cafe", page_icon="☕", layout="centered")

st.title("☕ BaristaBot Cafe")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

    # Boot up the graph immediately on first load to trigger the Welcome Message
    startup_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph_with_persistence.invoke({"messages": []}, startup_config)

if "payment_nonce" not in st.session_state:
    st.session_state.payment_nonce = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    if st.button("🆕 Start New Order", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")


config = {
    "recursion_limit": 100,
    "configurable": {"thread_id": st.session_state.thread_id},
}

# ── Payment server URL ────────────────────────────────────────────────────────
# Set PAYMENT_SERVER_URL in your .env to point at your FastAPI server.
# e.g. PAYMENT_SERVER_URL=https://xxxx.ngrok-free.app/pay  (if using a second tunnel)
# or   PAYMENT_SERVER_URL=http://localhost:8502/pay         (if on same machine, no HTTPS needed for the redirect target)
PAYMENT_SERVER_URL = os.environ.get("PAYMENT_SERVER_URL", "http://localhost:8502/pay")

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    return ""


def get_last_ai_text(messages: list) -> str | None:
    for msg in reversed(messages):
        if msg.type == "ai":
            text = extract_text(msg.content)
            if text.strip():
                return text
    return None


# ── Load current graph state ──────────────────────────────────────────────────
try:
    state = graph_with_persistence.get_state(config)
    messages = state.values.get("messages", [])
except Exception as exc:
    st.error(f"Failed to load conversation state: {exc}")
    st.stop()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in messages:
    if msg.type in ("tool", "system"):
        continue

    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(extract_text(msg.content))

    elif msg.type == "ai":
        text = extract_text(msg.content)
        if text.strip():
            with st.chat_message("assistant"):
                st.markdown(text)

# ── Detect interrupt type ─────────────────────────────────────────────────────
pending_interrupt = None
is_payment_interrupt = False
payment_total_cents = 0
payment_total_display = "$0.00"

if state.next and hasattr(state, 'tasks') and state.tasks:
    for task in state.tasks:
        if hasattr(task, 'interrupts') and task.interrupts:
            interrupt_val = task.interrupts[0].value
            if interrupt_val == "Waiting for user input":
                continue
            pending_interrupt = interrupt_val
            if "card" in interrupt_val.lower() or "payment" in interrupt_val.lower():
                is_payment_interrupt = True
                import re
                for line in interrupt_val.splitlines():
                    if "total" in line.lower():
                        match = re.search(r"\$?([\d]+\.[\d]{2})", line)
                        if match:
                            payment_total_display = f"${match.group(1)}"
                            payment_total_cents = int(float(match.group(1)) * 100)

# ── Capture nonce returned by Square payment page ─────────────────────────────
params = st.query_params
if "sq_nonce" in params and not st.session_state.payment_nonce:
    st.session_state.payment_nonce = params["sq_nonce"]
    st.query_params.clear()
    st.rerun()

# ── If we have a nonce, charge it and resume the graph ───────────────────────
if st.session_state.payment_nonce:
    nonce = st.session_state.payment_nonce
    st.session_state.payment_nonce = None

    from baristabot import charge_square
    with st.spinner("Processing your payment…"):
        result = charge_square(nonce, payment_total_cents)

    with st.spinner("Confirming order…"):
        try:
            for _ in graph_with_persistence.stream(
                Command(resume=result), config, stream_mode="values"
            ):
                pass
        except Exception as exc:
            st.error(f"Error resuming after payment: {exc}")
            st.stop()
    st.rerun()

# ── Payment interrupt: show a button that opens the Square payment page ───────
elif is_payment_interrupt:
    with st.chat_message("assistant"):
        st.markdown(pending_interrupt)

    # Build the payment URL with total info so the page can display it
    import urllib.parse
    pay_url = (
        f"{PAYMENT_SERVER_URL}"
        f"?total={urllib.parse.quote(payment_total_display)}"
        f"&cents={payment_total_cents}"
    )

    st.markdown("")
    st.link_button(
        f"💳 Pay {payment_total_display}",
        pay_url,
        use_container_width=True,
        type="primary",
    )
    st.caption("You'll be returned here automatically after payment.")

# ── Non-payment interrupts ────────────────────────────────────────────────────
elif pending_interrupt:
    with st.chat_message("assistant"):
        st.markdown(pending_interrupt)

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message here...", disabled=is_payment_interrupt):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Brewing your response…"):
            try:
                if state.next:
                    stream_input = Command(resume=prompt)
                else:
                    stream_input = {"messages": [("user", prompt)]}

                for _ in graph_with_persistence.stream(
                    stream_input, config, stream_mode="values"
                ):
                    pass

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

    st.rerun()
import uuid
 
import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command
 
from baristabot import graph_with_persistence
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BaristaBot Cafe", page_icon="☕", layout="centered")
 
st.title("☕ BaristaBot Cafe")
 
# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
    # NEW: Boot up the graph immediately on first load to trigger the Welcome Message
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    """Safely extract text whether content is a string or a list of content blocks."""
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
    """Return the text of the most recent AI message that has visible content."""
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
    # Hide background tool/system thoughts from the UI
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
                # Extract total amount from the interrupt message
                for line in interrupt_val.splitlines():
                    if "total" in line.lower():
                        import re
                        match = re.search(r"\$?([\d]+\.[\d]{2})", line)
                        if match:
                            payment_total_display = f"${match.group(1)}"
                            payment_total_cents = int(float(match.group(1)) * 100)


# ── If we have a nonce from the Square component, charge it ──────────────────
if st.session_state.payment_nonce:
    nonce = st.session_state.payment_nonce
    st.session_state.payment_nonce = None 

    from baristabot import charge_square
    with st.spinner("Processing your payment with Square..."):
        result = charge_square(nonce, payment_total_cents)

    with st.spinner("Confirming order..."):
        try:
            for _ in graph_with_persistence.stream(
                Command(resume=result), config, stream_mode="values"
            ):
                pass
        except Exception as exc:
            st.error(f"Error resuming after payment: {exc}")
            st.stop()
    st.rerun()

# ── Square payment dialog ─────────────────────────────────────────────────────
elif is_payment_interrupt:

    import os

    DEV_MODE = os.environ.get("DEV_MODE", "true").lower() == "true"



    if DEV_MODE:

        # Square Web Payments SDK requires HTTPS, which isn't available in local dev.

        # In DEV_MODE we skip tokenization and use Square's standard sandbox test nonce

        # (cnon:card-nonce-ok) which always succeeds in the sandbox environment.

        @st.dialog("💳 Payment method")

        def show_dev_payment_dialog():

            st.caption("🧪 Dev mode — Square SDK requires HTTPS (not available locally).")

            st.markdown(f"**Total: {payment_total_display}**")

            st.markdown("")

            st.info("Simulates a successful payment using Square's sandbox test nonce.")

            if st.button(f"Pay {payment_total_display} (Simulated)", use_container_width=True, type="primary"):

                st.session_state.payment_nonce = "cnon:card-nonce-ok"

                st.rerun()



        show_dev_payment_dialog()



    else:

        # Production path: Square Web Payments SDK (requires HTTPS)

        @st.dialog("💳 Payment method")

        def show_square_dialog():

            from baristabot import SQUARE_APP_ID, SQUARE_LOCATION_ID



            st.caption("This will be used to pay for your order.")

            st.markdown(f"**Total: {payment_total_display}**")

            st.markdown("")

            square_html = f"""<!DOCTYPE html>

<html>

<head>

<script type="text/javascript" src="https://sandbox.web.squarecdn.com/v1/square.js"></script>

<style>

  * {{ box-sizing: border-box; font-family: sans-serif; }}

  body {{ margin: 0; padding: 0; }}

  #card-container {{ margin: 8px 0 16px 0; min-height: 89px; }}

  #pay-btn {{

    width: 100%; padding: 12px; background: #3b82f6; color: white;

    border: none; border-radius: 8px; font-size: 15px; font-weight: 600;

    cursor: pointer; transition: background 0.2s;

  }}

  #pay-btn:hover:not(:disabled) {{ background: #2563eb; }}

  #pay-btn:disabled {{ background: #93c5fd; cursor: not-allowed; }}

  #status {{ margin-top: 10px; font-size: 13px; color: #374151; text-align: center; }}

</style>

</head>

<body>

<div id="card-container"></div>

<button id="pay-btn" disabled>Loading…</button>

<div id="status"></div>

<script>

  function waitForSquare(retries) {{

    if (window.Square) {{

      initSquare();

    }} else if (retries > 0) {{

      setTimeout(() => waitForSquare(retries - 1), 300);

    }} else {{

      document.getElementById("status").innerText = "⚠️ Square SDK failed to load.";

    }}

  }}

  async function initSquare() {{

    try {{

      const payments = window.Square.payments("{SQUARE_APP_ID}", "{SQUARE_LOCATION_ID}");

      const card = await payments.card();

      await card.attach("#card-container");

      const btn = document.getElementById("pay-btn");

      btn.disabled = false;

      btn.innerText = "Pay {payment_total_display}";

      btn.addEventListener("click", async () => {{

        btn.disabled = true;

        btn.innerText = "Processing…";

        document.getElementById("status").innerText = "";

        try {{

          const result = await card.tokenize();

          if (result.status === "OK") {{

            document.getElementById("status").innerText = "✅ Card tokenized — confirming payment…";

            const url = new URL(window.parent.location.href);

            url.searchParams.set("sq_nonce", result.token);

            window.parent.location.href = url.toString();

          }} else {{

            const errs = result.errors.map(e => e.message).join(", ");

            document.getElementById("status").innerText = "❌ " + errs;

            btn.disabled = false;

            btn.innerText = "Pay {payment_total_display}";

          }}

        }} catch(err) {{

          document.getElementById("status").innerText = "❌ " + err.message;

          btn.disabled = false;

          btn.innerText = "Pay {payment_total_display}";

        }}

      }});

    }} catch(err) {{

      document.getElementById("status").innerText = "❌ Init error: " + err.message;

    }}

  }}

  waitForSquare(10);

</script>

</body>

</html>"""

            components.html(square_html, height=320)

            st.caption("🧪 Sandbox — use test card `4111 1111 1111 1111`, any future date, any CVV.")



        # Capture nonce from query params if Square redirected back with it

        params = st.query_params

        if "sq_nonce" in params:

            st.session_state.payment_nonce = params["sq_nonce"]

            st.query_params.clear()

            st.rerun()

        else:

            show_square_dialog()



# ── Non-payment interrupts as chat messages ───────────────────────────────────
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
                # If the graph is paused at an interrupt(), resume it; otherwise start fresh.
                if state.next:
                    stream_input = Command(resume=prompt)
                else:
                    stream_input = {"messages": [("user", prompt)]}
 
                # Simply consume the stream. We don't need to manually extract the final
                # message here anymore, because st.rerun() will draw it perfectly from the top!
                for _ in graph_with_persistence.stream(
                    stream_input, config, stream_mode="values"
                ):
                    pass
 
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()
 
    # Refresh so the top-level render loop draws the new messages AND any new interrupts!
    st.rerun()

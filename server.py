"""server.py — FastAPI backend for BaristaBot"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.types import Command

from baristabot import graph_with_persistence, charge_square

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


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


def get_messages(thread_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_with_persistence.get_state(config)
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        if msg.type == "system":
            continue
        text = extract_text(msg.content)
        if msg.type == "ai" and text.strip():
            result.append({"role": "assistant", "text": text})
        elif msg.type == "tool" and text.strip():
            result.append({"role": "assistant", "text": text})
        elif msg.type == "human":
            result.append({"role": "user", "text": text})
    return result


def get_interrupt(thread_id: str) -> dict | None:
    """Return interrupt info if the graph is paused."""
    import re
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_with_persistence.get_state(config)
    if not state.next:
        return None
    if not hasattr(state, "tasks") or not state.tasks:
        return None
    for task in state.tasks:
        if not hasattr(task, "interrupts") or not task.interrupts:
            continue
        val = task.interrupts[0].value
        if val == "Waiting for user input":
            return None
        is_payment = "card" in val.lower() or "payment" in val.lower()
        total_cents = 0
        total_display = "$0.00"
        if is_payment:
            for line in val.splitlines():
                if "total" in line.lower():
                    match = re.search(r"\$?([\d]+\.[\d]{2})", line)
                    if match:
                        total_display = f"${match.group(1)}"
                        total_cents = int(float(match.group(1)) * 100)
        return {
            "message": val,
            "is_payment": is_payment,
            "total_cents": total_cents,
            "total_display": total_display,
        }
    return None


def run_graph(thread_id: str, stream_input):
    config = {"recursion_limit": 100, "configurable": {"thread_id": thread_id}}
    for _ in graph_with_persistence.stream(stream_input, config, stream_mode="values"):
        pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")


class NewThreadResponse(BaseModel):
    thread_id: str
    messages: list[dict]
    interrupt: dict | None


@app.post("/thread/new")
def new_thread() -> NewThreadResponse:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph_with_persistence.invoke({"messages": []}, config)
    return NewThreadResponse(
        thread_id=thread_id,
        messages=get_messages(thread_id),
        interrupt=get_interrupt(thread_id),
    )


class StateResponse(BaseModel):
    messages: list[dict]
    interrupt: dict | None


@app.get("/thread/{thread_id}/state")
def thread_state(thread_id: str) -> StateResponse:
    return StateResponse(
        messages=get_messages(thread_id),
        interrupt=get_interrupt(thread_id),
    )


class ChatRequest(BaseModel):
    message: str


@app.post("/thread/{thread_id}/chat")
def chat(thread_id: str, req: ChatRequest) -> StateResponse:
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_with_persistence.get_state(config)
    if state.next:
        stream_input = Command(resume=req.message)
    else:
        stream_input = {"messages": [("user", req.message)]}
    run_graph(thread_id, stream_input)
    return StateResponse(
        messages=get_messages(thread_id),
        interrupt=get_interrupt(thread_id),
    )


class PaymentRequest(BaseModel):
    nonce: str
    amount_cents: int


@app.post("/thread/{thread_id}/pay")
def pay(thread_id: str, req: PaymentRequest) -> StateResponse:
    result = charge_square(req.nonce, req.amount_cents)
    run_graph(thread_id, Command(resume=result))
    return StateResponse(
        messages=get_messages(thread_id),
        interrupt=get_interrupt(thread_id),
    )


class EnvResponse(BaseModel):
    square_app_id: str
    square_location_id: str


@app.get("/config")
def get_config() -> EnvResponse:
    app_id = os.environ.get("SQUARE_APP_ID", "")
    location_id = os.environ.get("SQUARE_LOCATION_ID", "")
    if not app_id or not location_id:
        raise HTTPException(status_code=500, detail="Square credentials not configured.")
    return EnvResponse(square_app_id=app_id, square_location_id=location_id)
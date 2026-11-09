
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sqlite3
import time
import uuid
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm
from policy_rag import retrieve, build_retriever

DB_FILE = ROOT / "data" / "instamart.db"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

RETRIEVER = build_retriever()

# Tool specifications use the common JSON function-schema format.
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order's status, amount and delivery time in the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve the most relevant policy chunks for a policy question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the refund for a completed order delivered more than 45 minutes after ordering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]},
                },
                "required": ["order_amount", "delivery_minutes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved automatic refund of INR 1000 or less in the simulated ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund above INR 1000 for human approval without issuing it automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
                "additionalProperties": False,
            },
        },
    },
]


def get_order(order_id):
    with sqlite3.connect(DB_FILE) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            LIMIT 1
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": (
            float(row["order_amount"])
            if row["order_amount"] is not None else None
        ),
        "delivery_minutes": (
            float(row["delivery_minutes"])
            if row["delivery_minutes"] is not None else None
        ),
    }


def search_policy(query):
    return {"results": retrieve(query, k=3, retriever=RETRIEVER)}


def calculate_refund(order_amount, delivery_minutes):
    if order_amount is None or delivery_minutes is None:
        return {"refund_inr": 0.0}

    refund = float(order_amount) if float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def _record_ledger(tool_name, order_id, refund_inr):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": str(order_id),
        "refund_inr": round(float(refund_inr), 2),
    }
    with LEDGER_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    amount = round(float(refund_inr), 2)
    if amount <= 0 or amount > 1000:
        return {"error": "automatic refund must be greater than 0 and at most INR 1000"}
    _record_ledger("issue_refund", order_id, amount)
    return {"refund_inr": amount}


def escalate_to_human(order_id, refund_inr):
    amount = round(float(refund_inr), 2)
    if amount <= 1000:
        return {"error": "only refunds above INR 1000 should be escalated"}
    _record_ledger("escalate_to_human", order_id, amount)
    return {"refund_inr": amount}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

SYSTEM_MESSAGE = (
    "AGENT: You are a refund-support agent. Use tools to look up orders, "
    "check policy, calculate the refund, and issue or escalate only when allowed. "
    "Only completed orders qualify. Delivery must exceed 45 minutes for a refund. "
    "Never issue an automatic refund above INR 1000; escalate it to a human."
)

# Used only by the required crash-and-resume demonstration.
CRASH_AFTER_TOOL_COUNT = None


class SimulatedCrash(RuntimeError):
    def __init__(self, run_id):
        super().__init__(f"Simulated crash after checkpoint for run {run_id}")
        self.run_id = run_id


def _save_checkpoint(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{state['run_id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _normal_status(final_answer):
    if not final_answer or ":" not in final_answer:
        return "UNKNOWN"
    return final_answer.split(":", 1)[0].strip()


def _fallback(state, status, message):
    return {
        "status": status,
        "final_answer": f"{status}: {message} Please hand this case to a human support agent.",
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "run_id": state["run_id"],
    }


def _continue_agent(state, max_iterations=6, token_limit=6000,
                    time_limit_seconds=10):
    started = time.monotonic()

    while state["llm_calls"] < max_iterations:
        elapsed = time.monotonic() - started
        if elapsed >= time_limit_seconds:
            return _fallback(
                state, "FALLBACK_TIMEOUT",
                "The agent reached its time limit.",
            )

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        used = int(response.usage.total_tokens)
        state["tokens_used"] += used
        message = response.choices[0].message

        # Do not execute a requested tool if this response exceeded the budget.
        if state["tokens_used"] > token_limit:
            return _fallback(
                state, "FALLBACK_TOKEN_LIMIT",
                "The model token budget was exceeded before the requested tool ran.",
            )

        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            final_answer = message.content or "UNKNOWN: The model returned no answer."
            return {
                "status": _normal_status(final_answer),
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "run_id": state["run_id"],
            }

        assistant_calls = []
        for call in tool_calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assistant_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": arguments,
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        for call in assistant_calls:
            tool_name = call["name"]
            arguments = call["arguments"]

            if tool_name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {tool_name}"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as exc:
                    observation = {"error": f"{type(exc).__name__}: {exc}"}

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": tool_name,
                "content": json.dumps(observation, ensure_ascii=False),
            })

            state["steps"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            state["executed_tools"].append(tool_name)
            state["step_number"] += 1

            # Save the complete conversation after every tool observation.
            _save_checkpoint(state)

            if (
                CRASH_AFTER_TOOL_COUNT is not None
                and len(state["executed_tools"]) == CRASH_AFTER_TOOL_COUNT
            ):
                raise SimulatedCrash(state["run_id"])

    return _fallback(
        state, "FALLBACK_MAX_ITERATIONS",
        f"The agent reached the maximum of {max_iterations} model calls without a final answer.",
    )


def run_agent(user_message, scenario_id, max_iterations=6,
              token_limit=6000, time_limit_seconds=10):
    run_id = f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    state = {
        "run_id": run_id,
        "scenario_id": str(scenario_id),
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": str(user_message)},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "llm_calls": 0,
        "steps": [],
        "executed_tools": [],
    }

    return _continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


def resume_agent(run_id):
    checkpoint = STATE_DIR / f"{run_id}.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    return _continue_agent(state)


def _append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_scenarios():
    return pd.read_csv(ROOT / "data" / "raw" / "agent_scenarios.csv").fillna("")



def _extract_refund_amount(result):
    # Prefer the actual refund amount recorded by the issuing/escalation tool.
    for step in reversed(result["steps"]):
        if step["tool_name"] in {"issue_refund", "escalate_to_human"}:
            observation = step["observation"]
            if isinstance(observation, dict) and "refund_inr" in observation:
                return float(observation["refund_inr"])

    # Otherwise, use the amount calculated by the refund tool.
    for step in reversed(result["steps"]):
        if step["tool_name"] == "calculate_refund":
            observation = step["observation"]
            if isinstance(observation, dict) and "refund_inr" in observation:
                return float(observation["refund_inr"])

    # For cases such as NOT_FOUND, the refund amount is zero.
    return 0.0



def _record_run(scenario_id, result):
    record = {
        "scenario_id": str(scenario_id),
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }
    _append_jsonl(AGENT_RUNS_FILE, record)


def run_scenario_batch():
    rows = _read_scenarios()
    results = []

    # Start a fresh output file for this batch, avoiding duplicate batch rows.
    AGENT_RUNS_FILE.write_text("", encoding="utf-8")

    for _, row in rows.iterrows():
        result = run_agent(row["message"], row["scenario_id"])
        _record_run(row["scenario_id"], result)

        expected_amount = float(row["expected_refund_inr"])
        actual_amount = _extract_refund_amount(result)
        results.append({
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "refund_amount": actual_amount,
            "expected_refund_inr": expected_amount,
            "amount_matches": abs(actual_amount - expected_amount) <= 0.01,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "run_id": result["run_id"],
        })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "agent_scenario_summary.csv", index=False)
    print("\nScenario results:")
    print(summary.to_string(index=False))
    return summary


def run_safety_demos():
    scenarios = _read_scenarios()
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]

    token_demo = run_agent(
        s1["message"], "S1_TOKEN_LIMIT",
        token_limit=200,
    )
    print("\nS1 token-limit demo:", token_demo["status"])
    print(token_demo["final_answer"])

    timeout_demo = run_agent(
        s1["message"], "S1_TIMEOUT",
        time_limit_seconds=0,
    )
    print("\nS1 timeout demo:", timeout_demo["status"])
    print(timeout_demo["final_answer"])

    return token_demo, timeout_demo


def run_crash_resume_demo():
    global CRASH_AFTER_TOOL_COUNT

    scenarios = _read_scenarios()
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]
    CRASH_AFTER_TOOL_COUNT = 2

    try:
        run_agent(s1["message"], "S1_RESUME_DEMO")
        raise AssertionError("Expected the simulated crash did not occur.")
    except SimulatedCrash as crash:
        run_id = crash.run_id
        print("\nSimulated crash after checkpoint:", run_id)
    finally:
        CRASH_AFTER_TOOL_COUNT = None

    # Resume from the saved conversation; earlier tools must not be replayed.
    resumed = resume_agent(run_id)
    state = json.loads(
        (STATE_DIR / f"{run_id}.json").read_text(encoding="utf-8")
    )

    # Checkpoint is refreshed after observations; the returned result is authoritative.
    tools = [step["tool_name"] for step in resumed["steps"]]
    print("Tools executed across crash and resume:", tools)
    print("Resumed status:", resumed["status"])
    print("Resumed answer:", resumed["final_answer"])

    expected = [
        "get_order", "search_policy", "calculate_refund", "issue_refund"
    ]
    if tools != expected:
        raise AssertionError(f"Unexpected resumed tool sequence: {tools}")
    if resumed["status"] != "REFUND":
        raise AssertionError(f"Expected REFUND after resume, got {resumed['status']}")
    if abs(_extract_refund_amount(resumed) - 128.25) > 0.01:
        raise AssertionError("Resumed refund amount did not match INR 128.25")

    _record_run("S1_RESUME_DEMO", resumed)
    return resumed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print("Running all seven agent scenarios...")
    summary = run_scenario_batch()

    # S1 must use five model calls; S7 must use all six calls and fall back.
    s1 = summary.loc[summary["scenario_id"] == "S1"].iloc[0]
    s7 = summary.loc[summary["scenario_id"] == "S7"].iloc[0]
    if int(s1["llm_calls"]) != 5:
        raise AssertionError(f"S1 should use 5 model calls, got {s1['llm_calls']}")
    if s7["status"] != "FALLBACK_MAX_ITERATIONS" or int(s7["llm_calls"]) != 6:
        raise AssertionError("S7 did not finish six calls and stop by iteration limit.")

    run_safety_demos()
    run_crash_resume_demo()

    print("\nAgent outputs saved:")
    print(AGENT_RUNS_FILE)
    print(LEDGER_FILE)
    print(STATE_DIR)
    print(OUTPUT_DIR / "agent_scenario_summary.csv")


if __name__ == "__main__":
    main()

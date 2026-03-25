import os
from sys import audit
from unittest import result
from websockets import route
from websockets import route
import yaml
import re
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Any, Dict, TypedDict, Final
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------------------
# Typed state for multi-agent graph
# -------------------------------------------------------------------
class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user

# -------------------------------------------------------------------
# Prompt Manager
# -------------------------------------------------------------------
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class PromptManager:
    """Manages versioned prompts stored as YAML files."""

    def __init__(self):
        # Always use the "prompts" folder
        self.prompts_dir = Path("prompts")
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def load_prompt(self, agent_name: str) -> dict:
        """Load a prompt YAML file by agent name (flat file style)."""
        prompt_file = self.prompts_dir / f"{agent_name}.yaml"

        if not prompt_file.exists():
            raise ValueError(f"Prompt not found: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8-sig") as f:
            prompt_data = yaml.safe_load(f)

        if not isinstance(prompt_data, dict):
            raise ValueError(f"Prompt file {prompt_file} is empty or invalid YAML")

        # Add metadata
        prompt_data["loaded_at"] = datetime.utcnow().isoformat()
        prompt_data["file_path"] = str(prompt_file)
        return prompt_data

# -------------------------------------------------------------------
# Supervisor Node
# -------------------------------------------------------------------
VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_node(state: MultiAgentState) -> dict:
    prompt_manager = PromptManager()
    supervisor_prompt = prompt_manager.load_prompt("supervisor_v1")["system"]

    messages = [
        SystemMessage(content=supervisor_prompt),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}

# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------
def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")

# -------------------------------------------------------------------
# Specialist Agents
# -------------------------------------------------------------------
def orders_agent_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content="You are the orders support agent. Handle order-related queries."),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    
    return {
        "agent_used": "orders_agent_node",
        "response": getattr(response, "content", None) or response[0].content
    }


def billing_agent_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content="You are the billing support agent. Handle billing-related queries."),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    state["agent_used"] = "billing_agent_node"
    state["final_response"] = response.content
    return state


def technical_agent_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content="You are the technical support agent. Handle technical-related queries."),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    print("DEBUG type:", type(response))
    print("DEBUG repr:", response)

    
def subscription_agent_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content="You are the subscription support agent. Handle subscription-related queries."),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "subscription_agent_node",
        "response": getattr(response, "content", None) or response[0].content
    }

def general_agent_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content="You are the general support agent. Handle general queries."),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "general_agent_node",
        "response": getattr(response, "content", None) or response[0].content
    }

# -------------------------------------------------------------------
# Response Synthesizer
# -------------------------------------------------------------------
def synthesize_response_node(state: MultiAgentState) -> dict:
    final_text = f"Final response based on {state['agent_used']}: {state['specialist_result']}"
    return {"final_response": final_text}

# -------------------------------------------------------------------
# Graph Builder
# -------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")

    workflow.add_conditional_edges("supervisor_node", route_to_specialist)

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()

# -------------------------------------------------------------------
# Agent Handoff
# -------------------------------------------------------------------
@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str   # "low" | "normal" | "high"
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )

# -------------------------------------------------------------------
# Guardrails
# -------------------------------------------------------------------
INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]

def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input

# -------------------------------------------------------------------
# Audit Logging
# -------------------------------------------------------------------
@dataclass
class SessionAuditLog:
    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(self, agent: str, action: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }

def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    for request in [
        "i have issue with my billing statement,plz help!",
        "i wanted to replace my order but the website is not working",
    ]:
        safe_text = guard_request(request)
        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }
        result = graph.invoke(state)

        print("Request:", request)
        print(f"Route: {result.get('route', '')} Agent used: {result.get('agent_used', '')}")
        print(f"Final: Final response based on {result.get('agent_used', '')}: {result.get('final_response', '')}")

        print("---")

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)

if __name__ == "__main__":
    main()
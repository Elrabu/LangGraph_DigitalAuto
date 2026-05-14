from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json, re
from graph.state import WiperState
from langchain_core.callbacks import BaseCallbackHandler

class LLMTap(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("═══ LLM CALL ═══")
        for msg_list in messages:
            for m in msg_list:
                print(f"[{m.type.upper()}] {m.content}")
        print("════════════════")

    def on_llm_end(self, response, **kwargs):
        for gen in response.generations:
            for g in gen:
                print(f"[LLM REPLY] {g.text}")
        print("════════════════\n")

llm = ChatOllama(
    model="llama3.1:8b",
    #callbacks=[LLMTap()],
    temperature=0.0,
    base_url="http://localhost:11434"
)


 
SAFETY_PROMPT = """You are a vehicle Safety Agent. Assess risk based on VSS signals 
and assess whether a safety risk exists IN THIS SPECIFIC SITUATION.

Important: Base your assessment STRICTLY on the values provided in the user message.
Do NOT assume any condition that is not explicitly stated.

Rules of thumb (apply only if conditions match):
- If hood_is_open=True AND wiper_mode != OFF → HIGH risk (mechanical collision).
- If hood_is_open=False → no hood-related risk.
- Otherwise assess normally. 

Given the state, return a JSON object:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "assessment": "<one-sentence reasoning>",
  "recommended_action": "STOP_WIPER" | "KEEP_WIPER" | "REDUCE_WIPER"
}
 
Return ONLY valid JSON, nothing else.
"""
 
def safety_node(state: WiperState) -> WiperState:
    user = (
        f"hood_is_open={state['hood_is_open']}, "
        f"current_wiper_mode={state['current_wiper_mode']}, "
        f"vehicle_speed={state['vehicle_speed']} km/h"
    )
    response = llm.invoke([
        SystemMessage(content=SAFETY_PROMPT),
        HumanMessage(content=user)
    ]).content
 
    match = re.search(r'\{.*\}', response, re.DOTALL)
    data = json.loads(match.group(0))
 
    state["safety_risk_level"]  = data["risk_level"]
    state["safety_assessment"]  = data["assessment"]
    state["decided_action"]     = data["recommended_action"]
    state.setdefault("reasoning_log", []).append(
        f"[Safety] {data['risk_level']}: {data['assessment']}"
    )
    return state
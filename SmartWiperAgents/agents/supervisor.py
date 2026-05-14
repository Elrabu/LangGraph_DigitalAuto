# supervisor.py — Regel-basiert, kein LLM-Halluzinations-Risiko
from graph.state import WiperState

def supervisor_node(state: WiperState) -> WiperState:
    # Input defaults (sensor values)
    state.setdefault("hood_is_open",       False)
    state.setdefault("current_wiper_mode", "OFF")
    state.setdefault("vehicle_speed",      0.0)

    # Working memory defaults
    state.setdefault("reasoning_log",      [])
    state.setdefault("safety_assessment",  None)
    state.setdefault("safety_risk_level",  None)
    state.setdefault("decided_action",     None)

    if state.get("safety_assessment") is None:
        decision = "safety_agent"
    elif state.get("decided_action") is None:
        decision = "actuator_agent"
    else:
        decision = "END"

    state["next_agent"] = decision
    state["reasoning_log"].append(f"[Supervisor] → {decision}")
    return state

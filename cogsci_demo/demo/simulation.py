"""Run booth simulations and format comparison results."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from demo.constants import (
    DATA_DIR,
    DRUG_DISPLAY,
    DRUG_PICKS,
    FACTOR_PICKS,
    SCENARIO_NAME,
    SELECTABLE_FACTORS,
    SIM_DRUG_NUMERIC,
    SIMULATION_WEEKS,
)
from demo.learner_agent import ATTENTION_WEIGHTS_DIR, create_booth_learner, train_booth_learner
from demo.visitor_agent import VisitorAttentionAgent
from demo.weights import build_visitor_weights, top_factor_labels
from src.core.weekly_decisions import (
    _load_scenario_config,
    create_state_from_config,
    make_weekly_decisions,
)

RESULTS_THESIS_MAIN = "The first decision is where to look."
RESULTS_THESIS_SUB = (
    "Attention weights turn inventory signals into urgency scores"
    "highest-urgency drugs received deep planning each week."
)


def _load_scenario_drug_names() -> list[str]:
    from demo.constants import PROJECT_ROOT

    scenario_path = PROJECT_ROOT / "configs" / "scenarios" / "booth" / "demo_6drug.json"
    with open(scenario_path, "r") as handle:
        cfg = json.load(handle)
    return [drug["name"] for drug in cfg["drugs"]]


def get_scenario_preview() -> dict[str, Any]:
    drug_names = _load_scenario_drug_names()
    drugs = []
    for name in drug_names:
        display = DRUG_DISPLAY.get(name, {})
        numeric = SIM_DRUG_NUMERIC.get(name, {})
        runway = round(numeric["qoh"] / numeric["utz"], 2) if numeric.get("utz") else None
        drugs.append({
            "id": name,
            "label": name.replace("Drug ", ""),
            "runway": runway,
            **display,
        })
    return {
        "scenario": SCENARIO_NAME,
        "weeks": SIMULATION_WEEKS,
        "drug_picks": DRUG_PICKS,
        "factor_picks": FACTOR_PICKS,
        "drugs": drugs,
        "factors": [
            {"id": key, "label": label}
            for key, label in SELECTABLE_FACTORS.items()
        ],
    }


def _focus_drug_names(agent, drug_names: list[str]) -> list[str]:
    return [drug_names[i] for i in agent.last_focus_set if i < len(drug_names)]


ACTION_LABELS = {
    "WAIT_MONITOR": "Monitor",
    "AUDIT_QOH_AND_UTZ": "Audit supply",
    "IMPLEMENT_SOFT_LMA": "Soft demand reduction",
    "IMPLEMENT_HARD_LMA": "Hard demand reduction",
    "REMOVE_LMA": "Lift demand limits",
    "QUERY_ALTERNATE_SUPPLIER": "Query alt supplier",
    "CONTACT_MANUFACTURER_DIRECT": "Contact manufacturer",
    "SWITCH_SUPPLIER": "Switch supplier",
    "SWITCH_BACK_TO_PRIMARY": "Return to primary",
    "REQUEST_FROM_RESERVE_WAREHOUSE": "Reserve warehouse",
    "REQUEST_LOAN_FROM_OTHER_HOSPITALS": "Hospital loan",
    "GREY_MARKET_PURCHASE": "Grey market buy",
}


def _format_actions(raw_actions: list) -> list[str]:
    labels: list[str] = []
    for action in raw_actions or []:
        kind = action.kind if hasattr(action, "kind") else action
        name = kind.name if hasattr(kind, "name") else str(kind)
        label = ACTION_LABELS.get(name, name.replace("_", " ").title())
        if label not in labels:
            labels.append(label)
    return labels or ["Monitor"]


def _is_stockout(drug) -> bool:
    return drug.quantity_on_hand <= 0.0 or drug.runway_weeks < 1.0


def _explain_stockout(
    drug_name: str,
    *,
    focused: bool,
    action_labels: list[str],
    runway_start: float,
    runway_end: float,
) -> str:
    short = drug_name.replace("Drug ", "")
    only_monitor = action_labels == ["Monitor"]

    if not focused:
        if runway_start < 2.0:
            return (
                f"{short} had only {runway_start:.1f} wk runway but was outside your focus set, "
                f"so no actions were taken."
            )
        return (
            f"{short} was outside your focus set. Runway fell from {runway_start:.1f} to "
            f"{runway_end:.1f} wk with no interventions."
        )

    if only_monitor:
        return (
            f"{short} was in focus but only monitored. Runway fell from {runway_start:.1f} to "
            f"{runway_end:.1f} wk."
        )

    acts = ", ".join(action_labels)
    return (
        f"{short} was in focus ({acts}), but runway still fell from {runway_start:.1f} to "
        f"{runway_end:.1f} wk."
    )


def _build_drug_details(
    drug_names: list[str],
    focus_names: set[str],
    metrics: dict[str, Any],
    prev_true_state,
    next_true_state,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    stockout_drugs: list[str] = []

    for drug_name in drug_names:
        prev_drug = prev_true_state.drugs[drug_name]
        next_drug = next_true_state.drugs[drug_name]
        dm = metrics.get(drug_name, {})
        focused = drug_name in focus_names
        if dm:
            action_labels = _format_actions(dm.get("actions", []))
        elif focused:
            action_labels = ["Monitor"]
        else:
            action_labels = ["No actions taken"]
        stockout = _is_stockout(next_drug)
        runway_start = round(float(prev_drug.runway_weeks), 2)
        runway_end = round(float(next_drug.runway_weeks), 2)

        if stockout:
            stockout_drugs.append(drug_name)

        if not (focused or stockout):
            continue

        entry: dict[str, Any] = {
            "drug": drug_name,
            "label": drug_name.replace("Drug ", ""),
            "focused": focused,
            "actions": action_labels,
            "runway_start": runway_start,
            "runway_end": runway_end,
            "stockout": stockout,
        }
        if stockout:
            entry["stockout_reason"] = _explain_stockout(
                drug_name,
                focused=focused,
                action_labels=action_labels,
                runway_start=runway_start,
                runway_end=runway_end,
            )
        details.append(entry)

    return details


def _serialize_week(
    week: int,
    agent,
    drug_names: list[str],
    metrics: dict[str, Any],
    elapsed_ms: float,
    prev_true_state,
    next_true_state,
) -> dict[str, Any]:
    focus_names = set(_focus_drug_names(agent, drug_names))
    stockouts = metrics.get("stockouts", {})
    stockout_drugs = [name for name, hit in stockouts.items() if hit] if isinstance(stockouts, dict) else []
    runway_snapshot = {}
    if isinstance(metrics, dict):
        for drug_name in drug_names:
            drug_metrics = metrics.get(drug_name)
            if isinstance(drug_metrics, dict) and "runway_next_week" in drug_metrics:
                runway_snapshot[drug_name] = round(float(drug_metrics["runway_next_week"]), 2)
            elif drug_name in next_true_state.drugs:
                runway_snapshot[drug_name] = round(float(next_true_state.drugs[drug_name].runway_weeks), 2)

    drug_details = _build_drug_details(
        drug_names, focus_names, metrics, prev_true_state, next_true_state
    )
    stockout_insights = [d["stockout_reason"] for d in drug_details if d.get("stockout_reason")]

    return {
        "week": week,
        "focused_drugs": _focus_drug_names(agent, drug_names),
        "total_reward": round(float(metrics.get("total_reward", 0.0)), 2),
        "stockouts": stockout_drugs,
        "stockout_insights": stockout_insights,
        "drug_details": drug_details,
        "runways": runway_snapshot,
        "decision_time_ms": round(elapsed_ms, 1),
    }


def run_policy_simulation(agent) -> dict[str, Any]:
    os.environ["OVERRIDE_RNG_SEED"] = "42"
    drug_names = _load_scenario_drug_names()
    weeks = []
    total_reward = 0.0
    total_stockouts = 0
    prev_true_state = None
    prev_belief_state = None

    if hasattr(agent, "start_new_episode"):
        agent.start_new_episode()

    scenario_config = _load_scenario_config(SCENARIO_NAME)

    for week in range(SIMULATION_WEEKS):
        if week == 0 or prev_true_state is None:
            week_start_state = create_state_from_config(
                rng_seed=scenario_config["rng_seed"],
                suppliers_cfg=scenario_config["suppliers_cfg"],
                drugs_cfg=scenario_config["drugs_cfg"],
            )
        else:
            week_start_state = prev_true_state

        start = time.perf_counter()
        true_state, belief_state, metrics = make_weekly_decisions(
            agent,
            SCENARIO_NAME,
            week,
            prev_true_state,
            prev_belief_state,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_reward += float(metrics.get("total_reward", 0.0))
        stockouts = metrics.get("stockouts", {})
        if isinstance(stockouts, dict):
            total_stockouts += sum(1 for hit in stockouts.values() if hit)
        weeks.append(
            _serialize_week(
                week, agent, drug_names, metrics, elapsed_ms, week_start_state, true_state
            )
        )
        prev_true_state = true_state
        prev_belief_state = belief_state

    result = {
        "weeks": weeks,
        "summary": {
            "total_reward": round(total_reward, 2),
            "avg_reward_per_week": round(total_reward / SIMULATION_WEEKS, 2),
            "total_stockouts": total_stockouts,
            "final_focus": weeks[-1]["focused_drugs"] if weeks else [],
            "initial_focus": weeks[0]["focused_drugs"] if weeks else [],
        },
    }
    if hasattr(agent, "attention_weights"):
        result["attention_weights"] = agent.attention_weights.copy()
        result["top_factors"] = top_factor_labels(agent.attention_weights)
    return result


def _focus_set_equal(a: list[str], b: list[str]) -> bool:
    return set(a) == set(b)


def _format_drug_list(drugs: list[str]) -> str:
    return ", ".join(d.replace("Drug ", "") for d in drugs)


def _find_learner_focus_week(
    learner_weeks: list[dict],
    drug: str,
    through_week_idx: int,
) -> tuple[int | None, dict | None]:
    for w in reversed(learner_weeks[: through_week_idx + 1]):
        if drug not in w.get("focused_drugs", []):
            continue
        detail = next(
            (d for d in w.get("drug_details", []) if d["drug"] == drug),
            None,
        )
        return w["week"] + 1, detail
    return None, None


def _stockout_highlights(visitor_result: dict, learner_result: dict) -> list[dict[str, Any]]:
    """Contrast stockout weeks with what each policy did on those drugs."""
    highlights: list[dict[str, Any]] = []
    for vw, lw in zip(visitor_result["weeks"], learner_result["weeks"]):
        week_num = vw["week"] + 1
        for detail in vw.get("drug_details", []):
            if not detail.get("stockout"):
                continue
            drug = detail["drug"]
            short = drug.replace("Drug ", "")
            learner_detail = next(
                (d for d in lw.get("drug_details", []) if d["drug"] == drug),
                None,
            )
            learner_rw = lw.get("runways", {}).get(drug)
            contrast = None
            if learner_detail and learner_detail.get("focused"):
                acts = ", ".join(learner_detail.get("actions", []))
                contrast = f"Learner had {short} in focus ({acts})."
            elif drug not in lw.get("stockouts", []):
                parts: list[str] = []
                if learner_rw is not None:
                    parts.append(
                        f"Learner's {short} ended at {learner_rw:.1f} wk runway (no stockout)."
                    )
                else:
                    parts.append(f"Learner avoided a stockout on {short} that week.")
                focus_week, focus_detail = _find_learner_focus_week(
                    learner_result["weeks"], drug, vw["week"]
                )
                if focus_week is not None and focus_week < week_num and focus_detail:
                    acts = ", ".join(focus_detail.get("actions", []))
                    parts.append(
                        f"Earlier (week {focus_week}) Learner focused {short} ({acts})."
                    )
                contrast = " ".join(parts)
            highlights.append({
                "week": week_num,
                "drug": short,
                "your_reason": detail.get("stockout_reason", ""),
                "learner_contrast": contrast,
                "your_runway_end": detail.get("runway_end"),
                "learner_runway_end": learner_rw,
            })
    return highlights


def _count_focus_divergence(visitor_weeks: list[dict], learner_weeks: list[dict]) -> int:
    if len(visitor_weeks) <= 1:
        return 0
    return sum(
        1
        for vw, lw in zip(visitor_weeks[1:], learner_weeks[1:])
        if set(vw.get("focused_drugs", [])) != set(lw.get("focused_drugs", []))
    )


def _build_focus_matrix(weeks: list[dict], drug_names: list[str]) -> list[list[bool]]:
    matrix: list[list[bool]] = []
    for week in weeks:
        focused = set(week.get("focused_drugs", []))
        matrix.append([name in focused for name in drug_names])
    return matrix


def _build_result_card(
    visitor_result: dict,
    learner_result: dict,
    selected_drugs: list[str],
    selected_factors: list[str],
) -> dict[str, Any]:
    """Wordle-style shareable summary: visual data + one-line insight."""
    v_summary = visitor_result["summary"]
    l_summary = learner_result["summary"]
    v_reward = float(v_summary["avg_reward_per_week"])
    l_reward = float(l_summary["avg_reward_per_week"])
    v_stockouts = int(v_summary["total_stockouts"])
    l_stockouts = int(l_summary["total_stockouts"])
    reward_delta = round(l_reward - v_reward, 2)
    v_weeks = visitor_result["weeks"]
    l_weeks = learner_result["weeks"]
    drug_names = _load_scenario_drug_names()
    drug_labels = [n.replace("Drug ", "") for n in drug_names]
    factor_labels = [SELECTABLE_FACTORS.get(f, f) for f in selected_factors]
    initial = _format_drug_list(v_summary.get("initial_focus", []))
    focus_diverged = _count_focus_divergence(v_weeks, l_weeks)

    if v_stockouts == l_stockouts == 0:
        if reward_delta > 8:
            title = "SAME SUPPLY, DIFFERENT ATTENTION"
            subtitle = (
                f"No stockouts either way. The Learner scored {l_reward}/wk vs your {v_reward}/wk "
                f"by shifting urgency toward different drugs after week 1."
            )
        elif reward_delta < -8:
            title = "YOUR ATTENTION PAID OFF"
            subtitle = (
                f"No stockouts either way. You scored {v_reward}/wk vs the Learner's {l_reward}/wk."
            )
        else:
            title = "ATTENTION SHAPED THE PATH"
            subtitle = (
                f"Both kept supply over {SIMULATION_WEEKS} weeks with similar reward "
                f"({v_reward} vs {l_reward}/wk). The spotlight still moved differently."
            )
    elif l_stockouts < v_stockouts:
        title = "DIFFERENT SPOTLIGHT, FEWER STOCKOUTS"
        subtitle = (
            f"The Learner had {l_stockouts} drug-weeks in stockout vs your {v_stockouts}. "
            f"Attention (which signals counted as urgent) made the difference."
        )
    elif v_stockouts < l_stockouts:
        title = "YOU READ THE CRISIS WELL"
        subtitle = (
            f"Fewer stockouts on your run ({v_stockouts} vs {l_stockouts} drug-weeks)."
        )
    else:
        title = "SAME STOCKOUTS, DIFFERENT PATHS"
        subtitle = f"Reward: {v_reward}/wk (you) vs {l_reward}/wk (Learner)."

    return {
        "tagline": "The first decision is where to look.",
        "title": title,
        "subtitle": subtitle,
        "your_score": v_reward,
        "learner_score": l_reward,
        "score_label": "avg reward / week",
        "reward_delta": reward_delta,
        "your_stockouts": v_stockouts,
        "learner_stockouts": l_stockouts,
        "initial_focus": initial,
        "your_factors": factor_labels,
        "your_drugs": [d.replace("Drug ", "") for d in selected_drugs],
        "your_drugs_display": _format_drug_list(selected_drugs),
        "reward_series": {
            "you": [float(w.get("total_reward", 0)) for w in v_weeks],
            "learner": [float(w.get("total_reward", 0)) for w in l_weeks],
        },
        "focus_matrix": {
            "you": _build_focus_matrix(v_weeks, drug_names),
            "learner": _build_focus_matrix(l_weeks, drug_names),
        },
        "drug_labels": drug_labels,
        "weeks_count": SIMULATION_WEEKS,
        "focus_diverged_weeks": focus_diverged,
        "share_text": (
            f"Attention Demo | Started on {initial} | "
            f"{v_reward}/wk vs Learner {l_reward}/wk | "
            f"The first decision is where to look."
        ),
        "both_avoided_stockouts": v_stockouts == l_stockouts == 0,
    }


def _build_takeaway(visitor_result: dict, learner_result: dict) -> dict[str, Any]:
    """Compact flags for UI helpers."""
    v = visitor_result["summary"]
    l = learner_result["summary"]
    return {
        "both_avoided_stockouts": int(v["total_stockouts"]) == int(l["total_stockouts"]) == 0,
        "your_reward": float(v["avg_reward_per_week"]),
        "learner_reward": float(l["avg_reward_per_week"]),
    }


def _interpretive_line(visitor_factors: list[str], visitor_result: dict, learner_result: dict) -> str:
    factor_labels = [SELECTABLE_FACTORS.get(f, f) for f in visitor_factors]
    visitor_focus = visitor_result["summary"]["initial_focus"]
    learner_focus = learner_result["summary"]["initial_focus"]
    visitor_stockouts = visitor_result["summary"]["total_stockouts"]
    learner_stockouts = learner_result["summary"]["total_stockouts"]
    learner_top = learner_result.get("top_factors", [])[:2]
    learner_later = learner_result["summary"].get("final_focus", [])

    your_lead = factor_labels[0] if factor_labels else "your top signal"
    learner_lead = learner_top[0] if learner_top else "shortage history"
    your_drugs = _format_drug_list(visitor_focus)
    learner_drugs = _format_drug_list(learner_focus)
    same_week1 = _focus_set_equal(visitor_focus, learner_focus)

    if learner_stockouts < visitor_stockouts:
        if same_week1:
            return (
                f"Same opening spotlight ({your_drugs}). Your weights stayed fixed on {your_lead}; "
                f"the Learner shifted toward {learner_lead} and ended on "
                f"{_format_drug_list(learner_later)} and fewer stockouts followed."
            )
        return (
            f"You weighted {your_lead} highest; the Learner weighted {learner_lead} highest. "
            f"Deep planning landed on {your_drugs} vs {learner_drugs} and the Learner's "
            f"spotlight avoided more stockouts."
        )
    if same_week1:
        return (
            f"Same week-1 spotlight ({your_drugs}), but different urgency signals: "
            f"{your_lead} for you, {learner_lead} for the Learner."
        )
    return (
        f"You weighted {your_lead} highest; the Learner weighted {learner_lead} highest. "
        f"That sent deep planning to {your_drugs} vs {learner_drugs} in week 1."
    )


COMPARISON_RULES = {
    "horizon_weeks": SIMULATION_WEEKS,
    "your_policy": (
        "Week 1 uses your two drug picks as the focus set. After that, top-two focus "
        "each week from your three factor weights (runway always included)."
    ),
    "learner_agent": (
        "LearnedAttentionAgent from the paper: REINFORCE-updated attention weights, "
        "top-two focus set recomputed each week from urgency scores."
    ),
}


def run_comparison(
    selected_drugs: list[str],
    selected_factors: list[str],
) -> dict[str, Any]:
    drug_names = _load_scenario_drug_names()
    for drug in selected_drugs:
        if drug not in drug_names:
            raise ValueError(f"Unknown drug: {drug}")
    if len(selected_drugs) != DRUG_PICKS:
        raise ValueError(f"Exactly {DRUG_PICKS} drugs must be selected")
    if len(selected_factors) != FACTOR_PICKS:
        raise ValueError(f"Exactly {FACTOR_PICKS} factors must be selected")

    visitor_weights = build_visitor_weights(selected_factors)
    visitor_agent = VisitorAttentionAgent(
        attention_weights=visitor_weights,
        initial_focus_drugs=selected_drugs,
    )
    visitor_result = run_policy_simulation(visitor_agent)

    learner_agent = create_booth_learner(train_during_run=True)
    learner_result = run_policy_simulation(learner_agent)
    learner_weights = learner_result.get("attention_weights", learner_agent.attention_weights.copy())

    return {
        "session_id": str(uuid.uuid4()),
        "simulation_weeks": SIMULATION_WEEKS,
        "selected_drugs": selected_drugs,
        "selected_factors": selected_factors,
        "thesis_main": RESULTS_THESIS_MAIN,
        "thesis_sub": RESULTS_THESIS_SUB,
        "interpretation": _interpretive_line(selected_factors, visitor_result, learner_result),
        "result_card": _build_result_card(
            visitor_result, learner_result, selected_drugs, selected_factors
        ),
        "takeaway": _build_takeaway(visitor_result, learner_result),
        "stockout_highlights": _stockout_highlights(visitor_result, learner_result),
        "comparison_rules": COMPARISON_RULES,
        "your_policy": {
            "label": "Your Attention Policy",
            "weights": visitor_weights,
            "top_factors": top_factor_labels(visitor_weights),
            "selected_drugs": selected_drugs,
            **visitor_result,
        },
        "learner_agent": {
            "label": "Learner Agent",
            "weights": learner_weights,
            "top_factors": learner_result.get("top_factors", top_factor_labels(learner_weights)),
            **learner_result,
        },
    }


def train_and_save_learner(training_weeks: int = 25) -> dict[str, Any]:
    """Train LearnedAttentionAgent on booth scenario; persist weights only."""
    learner = train_booth_learner(training_weeks=training_weeks)
    weights_path = ATTENTION_WEIGHTS_DIR / "learned_attention_weights.json"
    return {
        "training_weeks": training_weeks,
        "weights_path": str(weights_path),
        "attention_weights": learner.attention_weights.copy(),
    }

"""Experiment Runner"""

import json
import os
import time
from datetime import datetime

from src.core.weekly_decisions import make_weekly_decisions

def run_experiment(agent, scenario, weeks, outdir: str = "results", run_name: str = "run_1"):
    """Experiment Runner"""
    agent_name = agent.name if hasattr(agent, 'name') else str(agent)
    run_id = f"{agent_name}_{scenario}"
    results_dir = os.path.join(outdir, run_name, run_id)
    os.makedirs(results_dir, exist_ok=True)
    weekly_path = os.path.join(results_dir, "weekly.jsonl")
    summary_path = os.path.join(results_dir, "summary.json")
    total_reward = 0.0
    total_actions = 0
    total_stockouts = 0
    decision_times = []
    per_drug_rewards = {}
    all_actions = []
    metadata = {
        "agent": agent_name,
        "scenario": scenario,
        "weeks": weeks,
        "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }
    prev_true_state = None
    prev_belief_state = None
    if hasattr(agent, 'start_new_episode'):
        agent.start_new_episode()
    with open(weekly_path, "w") as wf:
        for week in range(weeks):
            start_time = time.perf_counter()
            state, beliefs, metrics = make_weekly_decisions(agent, scenario, week, prev_true_state, prev_belief_state)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            prev_true_state = state
            prev_belief_state = beliefs
            if "total_reward" in metrics and metrics["total_reward"] is not None:
                total_reward += metrics["total_reward"]
                total_actions += metrics.get("total_actions", 0)
                total_stockouts += _count_stockouts(metrics)
                decision_times.append(elapsed_ms)
                for drug_name, drug_metrics in metrics.items():
                    if isinstance(drug_metrics, dict) and "reward" in drug_metrics:
                        if drug_name not in per_drug_rewards:
                            per_drug_rewards[drug_name] = []
                        per_drug_rewards[drug_name].append(drug_metrics["reward"])
                    if isinstance(drug_metrics, dict) and "actions" in drug_metrics:
                        drug_actions = drug_metrics["actions"]
                        if drug_actions and drug_actions != ['WAIT_MONITOR']:
                            action_strings = [
                                action.kind.name if hasattr(action, 'kind') and hasattr(action.kind, 'name') else str(action)
                                for action in drug_actions
                            ]
                            all_actions.extend(action_strings)
                record = {
                    "week": week,
                    "agent": agent_name,
                    "scenario": scenario,
                    "decision_time_ms": round(elapsed_ms, 3),
                    **_serialize_metrics(metrics),
                }
                # Log actual attention focus-set size (not action-count proxy).
                if hasattr(agent, "last_focus_set") and getattr(agent, "last_focus_set", None) is not None:
                    record["focus_set_size"] = len(agent.last_focus_set)
                wf.write(json.dumps(record) + "\n")
    avg_decision_ms = sum(decision_times) / len(decision_times) if decision_times else 0.0
    per_drug_summary = {}
    for drug_name, weekly_rewards in per_drug_rewards.items():
        per_drug_summary[drug_name] = {
            "total_reward": round(sum(weekly_rewards), 2),
            "avg_reward_per_week": round(sum(weekly_rewards) / len(weekly_rewards), 2)
        }
    summary = {
        **metadata,
        "total_reward": round(total_reward, 2),
        "avg_reward_per_week": round(total_reward / weeks, 2),
        "total_actions": total_actions,
        "total_stockouts": total_stockouts,
        "avg_decision_time_ms": round(avg_decision_ms, 3),
        "per_drug_rewards": per_drug_summary,
        "actions": all_actions,
    }
    if hasattr(agent, 'get_learning_stats'):
        summary["rl_learning_stats"] = agent.get_learning_stats()
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    return summary

def _count_stockouts(metrics):
    """Stouckout Counter"""
    if "stockouts" in metrics and isinstance(metrics["stockouts"], dict):
        return sum(1 for stockout in metrics["stockouts"].values() if stockout)
    elif "total_stockouts" in metrics:
        return int(metrics["total_stockouts"])
    return 0

def _serialize_metrics(metrics):
    """Metrics Serializer"""
    serializable = {}
    for key, value in metrics.items():
        if key in ["actions", "total_reward", "total_stockouts", "total_actions", "stockouts"]:
            serializable[key] = value
        elif isinstance(value, dict):
            drug_metrics = {}
            for drug_key, drug_value in value.items():
                if drug_key == "actions" and isinstance(drug_value, list):
                    drug_metrics[drug_key] = [
                        action.kind.name if hasattr(action, 'kind') and hasattr(action.kind, 'name') else str(action) 
                        for action in drug_value
                    ]
                else:
                    drug_metrics[drug_key] = drug_value
            serializable[key] = drug_metrics
        else:
            serializable[key] = value
    return serializable

    
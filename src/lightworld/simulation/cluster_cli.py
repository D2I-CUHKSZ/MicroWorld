"""Interactive cluster-method selection helpers for terminal runs."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from lightworld.simulation.cluster_flags import resolve_cluster_feature_flags


CLUSTER_METHOD_THRESHOLD = "threshold"
CLUSTER_METHOD_LLM_KEYWORD = "llm_keyword"

_PROMPT_LABELS = {
    CLUSTER_METHOD_THRESHOLD: "threshold_only",
    CLUSTER_METHOD_LLM_KEYWORD: "llm_keyword_consistency",
}


def detect_cluster_method(config: Dict[str, Any]) -> Optional[str]:
    topo_cfg = config.get("topology_aware", {}) or {}
    if not isinstance(topo_cfg, dict):
        return None

    threshold_enabled, llm_enabled = resolve_cluster_feature_flags(topo_cfg)
    if llm_enabled:
        return CLUSTER_METHOD_LLM_KEYWORD
    if threshold_enabled:
        return CLUSTER_METHOD_THRESHOLD
    return None


def cluster_method_payload(method: str) -> Dict[str, Any]:
    if method == CLUSTER_METHOD_THRESHOLD:
        return {
            "threshold_cluster_enabled": True,
            "llm_keyword_cluster_enabled": False,
            "cluster_mode": "threshold_only",
        }
    if method == CLUSTER_METHOD_LLM_KEYWORD:
        return {
            "threshold_cluster_enabled": False,
            "llm_keyword_cluster_enabled": True,
            "cluster_mode": "llm_keyword_consistency",
        }
    raise ValueError(f"Unsupported cluster method: {method}")


def apply_cluster_method_to_topology_config(topo_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    updated = dict(topo_cfg or {})
    updated.update(cluster_method_payload(method))
    return updated


def apply_cluster_method_to_simulation_config(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    topo_cfg = config.get("topology_aware", {}) or {}
    if not isinstance(topo_cfg, dict):
        topo_cfg = {}
    topo_cfg = apply_cluster_method_to_topology_config(topo_cfg, method)
    config["topology_aware"] = topo_cfg
    return topo_cfg


def apply_cluster_method_to_full_run_config(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    simulation_cfg = config.get("simulation", {}) or {}
    if not isinstance(simulation_cfg, dict):
        simulation_cfg = {}

    overrides = simulation_cfg.get("config_overrides", {}) or {}
    if not isinstance(overrides, dict):
        overrides = {}

    topo_cfg = overrides.get("topology_aware", {}) or {}
    if not isinstance(topo_cfg, dict):
        topo_cfg = {}

    overrides["topology_aware"] = apply_cluster_method_to_topology_config(topo_cfg, method)
    simulation_cfg["config_overrides"] = overrides
    config["simulation"] = simulation_cfg
    return overrides["topology_aware"]


def maybe_prompt_cluster_method(requested_method: Optional[str], current_method: Optional[str]) -> Optional[str]:
    if requested_method:
        return requested_method
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return current_method

    default_method = current_method or CLUSTER_METHOD_THRESHOLD
    default_choice = "1" if default_method == CLUSTER_METHOD_THRESHOLD else "2"

    print("Select the cluster method for this run:", flush=True)
    print("  1) threshold_only", flush=True)
    print("     Groups by opinion/influence/structure thresholds. Faster and more stable.", flush=True)
    print("  2) llm_keyword_consistency", flush=True)
    print("     Additionally uses LLM for keyword semantic consistency grouping. Stronger semantics but slower.", flush=True)

    while True:
        raw = input(f"Enter 1 or 2 [{default_choice}]: ").strip().lower()
        if not raw:
            return default_method
        if raw in {"1", "threshold", "threshold_only"}:
            return CLUSTER_METHOD_THRESHOLD
        if raw in {"2", "llm", "llm_keyword", "llm_keyword_consistency"}:
            return CLUSTER_METHOD_LLM_KEYWORD
        print("Invalid input. Please enter 1 or 2.", flush=True)


def describe_cluster_method(method: Optional[str]) -> str:
    if not method:
        return "disabled"
    return _PROMPT_LABELS.get(method, method)

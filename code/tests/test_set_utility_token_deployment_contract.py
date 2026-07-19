from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_set_utility_token_predictor_v2.json"


def test_token_predictor_deployment_latency_contract_is_frozen() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))[
        "deployment_evaluation"
    ]
    assert contract["primary_profile"] == "warm_shared_encoder"
    assert contract["warm_profile"] == {
        "event_embedding_computed_once_at_arrival": True,
        "full_event_source_tokens_retained_online": False,
        "current_query_source_tokens_shared_with_action_policy": True,
        "latency_interval": (
            "shared_query_tokens_ready_to_selected_subset_ready"
        ),
    }
    assert contract["acceptance"] == {
        "metric": (
            "warm_selector_p95_seconds_divided_by_"
            "action_policy_forward_p95_seconds"
        ),
        "maximum_ratio": 0.1,
        "held_out_utility_improvement_required": True,
    }
    assert contract["required_secondary_profile"] == "cold_standalone_encoder"
    assert contract["required_latency_quantiles"] == [0.5, 0.95]
    assert contract["exact_subset_oracle_is_offline_only"] is True
    assert contract["ranking_rule"] == "utility_latency_pareto_frontier"
    assert contract["failed_latency_variants_may_be_ablation_only"] is True


def test_token_predictor_deployment_contract_covers_all_online_costs() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))[
        "deployment_evaluation"
    ]
    assert set(contract["required_cost_metrics"]) == {
        "event_ingest_latency_seconds",
        "query_selector_latency_seconds",
        "end_to_end_policy_step_latency_seconds",
        "persistent_bytes_per_event",
        "peak_gpu_memory_bytes",
        "subset_score_count",
    }

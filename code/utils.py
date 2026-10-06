import numpy as np


def load_model_state(model, state):
    """Load paper checkpoints, discarding only legacy inactive P2 parameters.

    All active parameter names and shapes must match. Returns whether legacy
    parameters were removed, in which case an old optimizer must be reset.
    """
    legacy_names = {
        "p2_freq_low_raw",
        "p2_freq_gap_raw",
        "p2_hpair_scale_raw",
        "p2_tpair_scale_raw",
        "p2_qpair_scale_raw",
    }
    legacy = {key for key in state if key.rsplit(".", 1)[-1] in legacy_names}
    model.load_state_dict(
        {key: value for key, value in state.items() if key not in legacy}, strict=True
    )
    return bool(legacy)


def parse_idf_component_flags(idf_components, disable_idf):
    """Select the head, tail, and qualifier pair routes for IDF ablations."""
    components = set(idf_components.lower())
    invalid = components.difference("htq")
    if invalid:
        raise ValueError(f"Invalid IDF components: {sorted(invalid)}; use h, t, q.")
    if disable_idf:
        return False, False, False, "none"
    flags = tuple(route in components for route in "htq")
    description = ",".join(
        name for enabled, name in zip(flags, ("hpair", "tpair", "qpair")) if enabled
    )
    return *flags, description or "none"


def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    return rank


def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1

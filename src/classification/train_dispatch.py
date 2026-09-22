"""
Single entrypoint that selects the training arm from the config.

Before 2026-09-21 the arm was chosen by *which script you ran*:
`scripts/run_train.py` imported the single-view trainer, `run_train_dual_stream.py`
imported the dual-stream trainer, and each hardcoded its own default config. The
`use_multi_view_distillation` key existed in 19 configs but nothing read it, and it
disagreed with reality in 13 of them -- `configs/default.yaml` set it to `true` while
being consumed by the single-view trainer.

Routing through this module makes the key load-bearing, so the two arms really are one
config pair differing in one flag, and a config can no longer describe an arm it did not
train.
"""

from __future__ import annotations

from src.classification.train import TrainOutput
from src.config.load_config import load_config

TEACHER_KEYS = ("dual_stream", "multiview_distillation")


def resolve_arm(cfg: dict) -> str:
    """
    Return "single_stream" or "dual_stream", refusing configs that contradict themselves.

    The flag and the presence of a teacher hyperparameter block must agree. A config
    asking for distillation without teacher settings, or carrying teacher settings it
    never uses, is a mistake rather than a default worth silently resolving.
    """
    cls_cfg = cfg.get("classification", {}) or {}
    flag = bool(cls_cfg.get("use_multi_view_distillation", False))
    has_teacher_block = any(k in cls_cfg for k in TEACHER_KEYS)

    if flag and not has_teacher_block:
        raise ValueError(
            "use_multi_view_distillation is true but no classification.dual_stream block "
            "is present, so the teacher has no hyperparameters."
        )
    if has_teacher_block and not flag:
        raise ValueError(
            "classification.dual_stream is present but use_multi_view_distillation is "
            "false, so the teacher settings would be silently ignored."
        )
    return "dual_stream" if flag else "single_stream"


def train_from_config(config_path: str) -> TrainOutput:
    arm = resolve_arm(load_config(config_path))
    print(f"[dispatch] use_multi_view_distillation selects the {arm} arm")

    if arm == "dual_stream":
        from src.classification.train_dual_stream import train_from_config as _train
    else:
        from src.classification.train import train_from_config as _train
    return _train(config_path)

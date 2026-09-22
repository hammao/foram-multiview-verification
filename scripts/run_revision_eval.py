"""Post-retrain evaluation pipeline. Run only after both frozen checkpoints exist."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["single_stream", "dual_stream", "both"], default="both")
    args = ap.parse_args()
    py = sys.executable
    arms = ["single_stream", "dual_stream"] if args.arm == "both" else [args.arm]

    for arm in arms:
        cfg = f"configs/train_{arm}.yaml"
        ckpt = ROOT / f"checkpoints_{arm}" / "classifier.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing frozen checkpoint: {ckpt}")
        run([py, "scripts/select_best_checkpoint_by_val.py", "--config", cfg, "--apply"])
        # Calibrate before evaluating: run_eval.py reads checkpoints_<arm>/temperature.json and
        # falls back to T=1.0 with null calibrated metrics if the file does not exist yet.
        run(
            [
                py,
                "scripts/calibrate_temperature.py",
                "--config",
                cfg,
                "--split",
                "val",
                "--out",
                f"checkpoints_{arm}/temperature.json",
                "--checkpoints-dir",
                f"checkpoints_{arm}",
            ]
        )
        run([py, "scripts/run_eval.py", "--config", cfg])
        run(
            [
                py,
                "scripts/export_embedding_pca_data.py",
                "--config",
                cfg,
            ]
        )
        run(
            [
                py,
                "scripts/analyze_rarity_and_helegans.py",
                "--eval-report",
                f"processed/manifest/eval_report_{arm}.json",
                "--out-dir",
                f"processed/manifest/probes/{arm}",
            ]
        )
        run(
            [
                py,
                "scripts/eval_audit_once.py",
                "--config",
                cfg,
                "--out-prefix",
                f"processed/manifest/verified/audit_once_{arm}",
            ]
        )
        run(
            [
                py,
                "scripts/noise_test.py",
                "--config",
                cfg,
                "--n",
                "0",
                "--output-dir",
                f"processed/noise_tests/{arm}",
            ]
        )
        run(
            [
                py,
                "scripts/generate_report_pack.py",
                "--config",
                cfg,
                "--with-model-diagnostics",
            ]
        )


if __name__ == "__main__":
    main()

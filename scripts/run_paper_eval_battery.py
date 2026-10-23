#!/usr/bin/env python3
"""Paper evaluation battery for the 50-species / 140-shell revision (frozen A2 split).

Per arm, in this order:
  1. refuse unless run_status.json says complete and selection used validation only
  2. temperature scaling fitted on validation            (scripts/calibrate_temperature.py)
  3. score val + test, specimen majority, per-species, confusions, coverage, error analysis
                                                         (scripts/paper_score_and_summarize.py)
  4. Leica audit overlap (39 images) scored once          (scripts/eval_audit_once.py)
  5. coverage-accuracy curve on the audit overlap         (scripts/audit_confidence_curve.py)
  6. Gaussian / salt-and-pepper noise on the full test split (scripts/noise_test.py)

Everything lands in reports/experiments/2026-10_drive140_paper/<arm>/ with a battery_manifest.json
that records the SHA-256 of the frozen split files, the audit list, and the deployed checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "experiments" / "2026-10_drive140_paper"
SPLIT = ROOT / "processed" / "drive140" / "manifest" / "splits_specimen_tvt"
ARMS = {
    "single_stream": "configs/train_single_stream_drive140_paper.yaml",
    "dual_stream": "configs/train_dual_stream_drive140_paper.yaml",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  rc={p.returncode}  {' '.join(cmd[:2])}")
    return int(p.returncode)


def audit_labels_csv() -> Path:
    """Derive the labels CSV the audit scripts expect from the frozen overlap list."""
    out = REPORT / "audit_overlap_39_labels.csv"
    if out.exists():
        return out
    ov = pd.read_csv(SPLIT / "audit_overlap_39.csv")
    ov.rename(columns={"audit_image_path": "image_path"})[["image_path", "true_species", "difficulty_stratum"]].to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip steps whose outputs already exist (the audit is then not re-scored).")
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    labels_csv = audit_labels_csv()
    frozen = {p.name: sha256(p) for p in sorted(SPLIT.iterdir())}
    prov = json.loads((SPLIT / "split_provenance.json").read_text(encoding="utf-8"))
    for name, digest in prov["file_sha256"].items():
        if frozen.get(name) != digest:
            raise SystemExit(f"Frozen file {name} changed since pre-registration; refusing to evaluate")

    for arm in args.arms:
        config = ARMS[arm]
        ckpt_dir = ROOT / "models" / "v3_expansion" / "20261016_drive140_paper" / arm
        status_path = ckpt_dir / "run_status.json"
        if not status_path.exists():
            print(f"[{arm}] no run_status.json yet; skipping")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            print(f"[{arm}] status {status.get('status')!r}; skipping")
            continue
        sel = json.loads((ckpt_dir / "selection_provenance.json").read_text(encoding="utf-8"))
        if sel.get("audit_cohort_used") or sel.get("test_split_used"):
            raise SystemExit(f"[{arm}] selection provenance violates the firewall")

        out = REPORT / arm
        out.mkdir(parents=True, exist_ok=True)
        log = out / "battery.log"
        mode = "a" if args.resume and log.exists() else "w"
        with log.open(mode, encoding="utf-8") as fh:
            fh.write(f"battery start {datetime.now(timezone.utc).isoformat()} resume={args.resume}\n")
        print(f"[{arm}] epoch {status.get('selected_epoch')} of {status.get('last_epoch')}")

        def done(*paths: Path) -> bool:
            return args.resume and all(p.exists() for p in paths)

        if not done(ckpt_dir / "temperature.json"):
            run(["python3", "scripts/calibrate_temperature.py", "--config", config, "--split", "val"], log)
        if not done(out / "summary.json"):
            run(["python3", "scripts/paper_score_and_summarize.py", "--config", config, "--out-dir", str(out)], log)
        if not done(out / "audit_once_39_summary.json"):
            run(["python3", "scripts/eval_audit_once.py", "--config", config, "--labels-csv", str(labels_csv), "--out-prefix", str(out / "audit_once_39")], log)
        if not done(out / "audit_confidence_39_summary.json"):
            run(["python3", "scripts/audit_confidence_curve.py", "--config", config, "--labels-csv", str(labels_csv), "--out-prefix", str(out / "audit_confidence_39")], log)
        if not args.skip_noise and not (args.resume and list((out / "noise").glob("noise_test_*/results.csv"))):
            for empty in (out / "noise").glob("noise_test_*"):
                if empty.is_dir() and not any(empty.iterdir()):
                    empty.rmdir()
            run(["python3", "scripts/noise_test.py", "--config", config, "--from-csv", str(SPLIT / "test.csv"), "--n", "0", "--output-dir", str(out / "noise")], log)

        manifest = {
            "arm": arm,
            "config": config,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256(ckpt_dir / "classifier.pt"),
            "selected_epoch": status.get("selected_epoch"),
            "epochs_run": status.get("last_epoch"),
            "selection_provenance": sel,
            "frozen_split_sha256": frozen,
            "audit_labels_csv": str(labels_csv),
            "audit_scored_once": True,
        }
        (out / "battery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{arm}] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

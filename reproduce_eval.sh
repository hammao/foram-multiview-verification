#!/usr/bin/env bash
#
# Regenerate the committed evaluation outputs from the released model weights.
#
# Downloads both checkpoints from the GitHub release, verifies their SHA-256 sums, re-runs
# evaluation on the held-out-view test split and on the locked audit cohort, and diffs the fresh
# numbers against the files in results/. Training is not re-run; this reproduces evaluation only.
#
#   ./reproduce_eval.sh                  # both arms
#   ./reproduce_eval.sh single_stream    # one arm
#   SKIP_DOWNLOAD=1 ./reproduce_eval.sh  # use checkpoints already in checkpoints/<arm>/
#
# Needs ~2 GB of disk. A few minutes on a GPU, roughly half an hour on CPU.

set -euo pipefail
cd "$(dirname "$0")"

RELEASE_TAG="${RELEASE_TAG:-v1.0.0}"
REPO="${REPO:-hammao/foram-multiview-verification}"
BASE_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"
ARMS=("${@:-single_stream dual_stream}")
read -r -a ARMS <<< "${ARMS[*]}"

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

require python3
require curl

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

fetch_checkpoint() {
  local arm="$1" dest="checkpoints/${arm}/classifier.pt"
  mkdir -p "checkpoints/${arm}"
  if [[ -f "$dest" ]]; then
    echo "already present: $dest"
  elif [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    echo "SKIP_DOWNLOAD=1 but $dest is missing" >&2; exit 1
  else
    echo "downloading ${arm} checkpoint (~350 MB)"
    curl -fL --progress-bar -o "$dest" "${BASE_URL}/classifier_${arm}.pt"
  fi

  # CHECKSUMS.sha256 in the release notes covers the assets; the copy here covers the repo.
  local expected
  expected=$(grep -E "classifier_${arm}\.pt$" release_checksums.sha256 2>/dev/null | awk '{print $1}' || true)
  if [[ -n "$expected" ]]; then
    local actual; actual=$(sha256 "$dest")
    if [[ "$actual" != "$expected" ]]; then
      echo "checksum mismatch for $dest" >&2
      echo "  expected $expected" >&2
      echo "  actual   $actual" >&2
      exit 1
    fi
    echo "checksum ok"
  else
    echo "no recorded checksum for ${arm}; skipping verification"
  fi

  # The label map and the deployed epoch's metadata travel with the repository, not the release.
  [[ -f "checkpoints/${arm}/label_map.json" ]] || cp "results/${arm}/label_map.json" "checkpoints/${arm}/label_map.json"
  [[ -f "checkpoints/${arm}/temperature.json" ]] || cp "results/${arm}/temperature.json" "checkpoints/${arm}/temperature.json"
}

# Point processed_dir at reproduced/<arm> so nothing overwrites the committed outputs. The config
# has to stay inside configs/ because the project root is resolved from that directory's parent.
write_scratch_config() {
  local arm="$1"
  python3 - "$arm" <<'PY'
import sys, yaml, pathlib
arm = sys.argv[1]
cfg = yaml.safe_load(pathlib.Path(f"configs/train_{arm}.yaml").read_text())
cfg["paths"]["processed_dir"] = f"reproduced/{arm}"
pathlib.Path(f"reproduced/{arm}/manifest/verified").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"configs/_reproduce_{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

compare() {
  local arm="$1"
  python3 compare_reproduced.py "$arm"
}

trap 'rm -f configs/_reproduce_*.yaml' EXIT

for arm in "${ARMS[@]}"; do
  say "${arm}: fetching weights"
  fetch_checkpoint "$arm"
  write_scratch_config "$arm"
  cfg="configs/_reproduce_${arm}.yaml"

  say "${arm}: fitting the temperature on the validation split"
  python3 scripts/calibrate_temperature.py --config "$cfg" --split val \
    --out "reproduced/${arm}/temperature.json" --checkpoints-dir "checkpoints/${arm}"
  cp "reproduced/${arm}/temperature.json" "checkpoints/${arm}/temperature.json"

  say "${arm}: evaluating the held-out-view test split"
  python3 scripts/run_eval.py --config "$cfg"

  say "${arm}: scoring the locked audit cohort"
  python3 scripts/eval_audit_once.py --config "$cfg" \
    --labels-csv data/audit/verified_labels_audit_v2.csv \
    --out-prefix "reproduced/${arm}/audit_once"

  say "${arm}: comparing against the committed outputs"
  compare "$arm"
done

say "done"
echo "Fresh outputs are under reproduced/. The committed ones are untouched in results/."

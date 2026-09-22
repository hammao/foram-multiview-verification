from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

SYNONYMS_CSV = Path(__file__).resolve().parents[2] / "configs" / "species_synonyms.csv"
SPECIES_CSV = Path(__file__).resolve().parents[2] / "configs" / "species.csv"


def load_species_synonyms(path: Path | None = None) -> dict[str, str]:
    """Map legacy species names to the accepted name used in the model label space."""
    src = path if path is not None else SYNONYMS_CSV
    if not src.exists():
        return {}
    table = pd.read_csv(src)
    return {
        str(row["legacy_name"]).strip(): str(row["accepted_species"]).strip()
        for _, row in table.iterrows()
    }


def load_label_space(path: Path | None = None) -> set[str]:
    src = path if path is not None else SPECIES_CSV
    if not src.exists():
        return set()
    return {str(name).strip() for name in pd.read_csv(src)["species"]}


@dataclass(frozen=True)
class LockedAuditManifestResult:
    manifest_csv: Path
    locked_labels_csv: Path
    n_total: int
    n_included: int
    n_normal: int
    n_difficult: int
    n_excluded: int


@dataclass(frozen=True)
class VerifiedImageRecord:
    path: Path
    true_species: str


def _normalize_path(path_value: str) -> str:
    return str(Path(str(path_value)).expanduser().resolve())


def _genus(species_name: str) -> str:
    parts = str(species_name).strip().split()
    return parts[0] if parts else ""


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = float(successes) / float(total)
    denom = 1.0 + (z**2) / float(total)
    center = (phat + (z**2) / (2.0 * float(total))) / denom
    margin = (
        z
        * (
            (phat * (1.0 - phat) / float(total))
            + ((z**2) / (4.0 * (float(total) ** 2)))
        )
        ** 0.5
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def load_image_basenames(root: Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> set[str]:
    return set(load_image_paths_by_basename(root, suffixes=suffixes).keys())


def load_image_paths_by_basename(root: Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> dict[str, Path]:
    wanted = {str(s).lower() for s in suffixes}
    mapping: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or ((path.suffix or "").lower() not in wanted):
            continue
        if path.name in mapping:
            raise ValueError(f"Duplicate basename in audit stratum directory {root}: {path.name}")
        mapping[path.name] = path.resolve()
    return mapping


def load_verified_records_by_basename(root: Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> dict[str, VerifiedImageRecord]:
    wanted = {str(s).lower() for s in suffixes}
    mapping: dict[str, VerifiedImageRecord] = {}
    for path in root.rglob("*"):
        if not path.is_file() or ((path.suffix or "").lower() not in wanted):
            continue
        if path.name in mapping:
            raise ValueError(f"Duplicate basename in verified reference directory {root}: {path.name}")
        mapping[path.name] = VerifiedImageRecord(
            path=path.resolve(),
            true_species=path.parent.name.replace("_", " ").strip(),
        )
    return mapping


def build_locked_audit_manifest(
    *,
    verified_labels: pd.DataFrame,
    normal_filenames: set[str],
    difficult_filenames: set[str],
    cohort_version: str,
    exclusion_reason: str = "not_in_locked_cohort",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_path", "true_species"}
    if not required.issubset(verified_labels.columns):
        missing = sorted(required - set(verified_labels.columns))
        raise ValueError(f"verified_labels missing required columns: {missing}")

    overlap = set(normal_filenames) & set(difficult_filenames)
    if overlap:
        overlap_list = ", ".join(sorted(overlap))
        raise ValueError(f"Ambiguous filename membership across strata: {overlap_list}")

    df = verified_labels.copy()
    df["image_path"] = df["image_path"].astype(str)
    df["image_path_abs"] = df["image_path"].map(_normalize_path)
    df["image_basename"] = df["image_path"].map(lambda p: Path(str(p)).name)

    def _difficulty_for_name(name: str) -> str:
        if name in normal_filenames:
            return "normal"
        if name in difficult_filenames:
            return "difficult"
        return ""

    df["difficulty_stratum"] = df["image_basename"].map(_difficulty_for_name)
    df["include_for_reporting"] = df["difficulty_stratum"].isin({"normal", "difficult"})
    df["cohort_version"] = str(cohort_version)
    df["inclusion_reason"] = df["include_for_reporting"].map(
        lambda include: "locked_prediction_audit_cohort" if bool(include) else str(exclusion_reason)
    )

    locked_labels = df[df["include_for_reporting"]].copy()
    return (df, locked_labels)


def write_locked_audit_manifest(
    *,
    verified_labels_csv: Path,
    normal_dir: Path,
    difficult_dir: Path,
    out_manifest_csv: Path,
    out_locked_labels_csv: Path,
    cohort_version: str,
    exclusion_reason: str = "not_in_locked_cohort",
    verified_root: Path | None = None,
    species_csv: Path | None = None,
    synonyms_csv: Path | None = None,
) -> LockedAuditManifestResult:
    verified = pd.read_csv(verified_labels_csv)
    normal_paths = load_image_paths_by_basename(normal_dir)
    difficult_paths = load_image_paths_by_basename(difficult_dir)
    verified_records = load_verified_records_by_basename(verified_root) if verified_root is not None else {}
    manifest, locked_labels = build_locked_audit_manifest(
        verified_labels=verified,
        normal_filenames=set(normal_paths.keys()),
        difficult_filenames=set(difficult_paths.keys()),
        cohort_version=cohort_version,
        exclusion_reason=exclusion_reason,
    )

    manifest["original_true_species"] = manifest["true_species"].astype(str)
    if verified_records:
        manifest["verified_image_path"] = manifest["image_basename"].map(
            lambda basename: str(verified_records[basename].path) if basename in verified_records else ""
        )
        manifest["true_species"] = manifest["image_basename"].map(
            lambda basename: verified_records[basename].true_species if basename in verified_records else ""
        )
        manifest.loc[manifest["true_species"].astype(str).eq(""), "true_species"] = manifest["original_true_species"]
        manifest["true_species_source"] = manifest["verified_image_path"].map(
            lambda path: "verified_validation_folder" if str(path).strip() else "verified_labels_csv"
        )

        # The directory names under verified_root predate the WoRMS harmonisation, so taking
        # them verbatim silently reintroduces five superseded names. Scored against a model
        # trained on accepted names those images can never be counted correct, which caps
        # audit top-1 at 29/45 and makes the metric a measure of label agreement rather than
        # of classification skill. See processed/manifest/verified/AUDIT_LABEL_TAXONOMY.md.
        synonyms = load_species_synonyms(synonyms_csv)
        if synonyms:
            renamed = manifest["true_species"].astype(str).map(lambda name: synonyms.get(name, name))
            manifest["true_species_source"] = manifest["true_species_source"].where(
                renamed.eq(manifest["true_species"]), "verified_validation_folder_synonymised"
            )
            manifest["true_species"] = renamed
    else:
        manifest["verified_image_path"] = ""
        manifest["true_species_source"] = "verified_labels_csv"

    label_space = load_label_space(species_csv)
    if label_space:
        off_label = manifest[manifest["include_for_reporting"] & ~manifest["true_species"].isin(label_space)]
        if not off_label.empty:
            offenders = ", ".join(sorted(set(off_label["true_species"].astype(str))))
            raise ValueError(
                "Audit labels outside the model label space, so these images could never be "
                f"scored correct: {offenders}. Add the mapping to configs/species_synonyms.csv."
            )

    if verified_root is not None:
        missing_verified = manifest[manifest["include_for_reporting"] & manifest["verified_image_path"].astype(str).eq("")]
        if not missing_verified.empty:
            missing = ", ".join(missing_verified["image_basename"].astype(str).tolist())
            raise ValueError(f"Included audit images missing from verified reference directory: {missing}")

    def _reporting_path(row: pd.Series) -> str:
        basename = str(row["image_basename"])
        difficulty = str(row["difficulty_stratum"])
        if difficulty == "normal" and basename in normal_paths:
            return str(normal_paths[basename])
        if difficulty == "difficult" and basename in difficult_paths:
            return str(difficult_paths[basename])
        linked_path = str(row.get("linked_path", "")).strip()
        if linked_path:
            return _normalize_path(linked_path)
        return str(row["image_path_abs"])

    manifest["reporting_image_path"] = manifest.apply(_reporting_path, axis=1)
    manifest["reporting_image_path_abs"] = manifest["reporting_image_path"].map(_normalize_path)
    locked_labels = manifest[manifest["include_for_reporting"]].copy()
    locked_labels["source_image_path"] = locked_labels["image_path"].astype(str)
    locked_labels["image_path"] = locked_labels["reporting_image_path"]
    locked_labels["image_path_abs"] = locked_labels["reporting_image_path_abs"]

    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    out_locked_labels_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_manifest_csv, index=False)
    locked_labels.to_csv(out_locked_labels_csv, index=False)

    n_normal = int((manifest["difficulty_stratum"] == "normal").sum())
    n_difficult = int((manifest["difficulty_stratum"] == "difficult").sum())
    n_included = int(manifest["include_for_reporting"].sum())

    return LockedAuditManifestResult(
        manifest_csv=out_manifest_csv.resolve(),
        locked_labels_csv=out_locked_labels_csv.resolve(),
        n_total=int(len(manifest)),
        n_included=n_included,
        n_normal=n_normal,
        n_difficult=n_difficult,
        n_excluded=int(len(manifest) - n_included),
    )


def summarize_audit_predictions(*, predictions: pd.DataFrame, audit_manifest: pd.DataFrame) -> pd.DataFrame:
    required_manifest = {"image_path", "true_species", "difficulty_stratum", "include_for_reporting"}
    required_predictions = {"image_path", "pred_1"}
    if not required_manifest.issubset(audit_manifest.columns):
        missing = sorted(required_manifest - set(audit_manifest.columns))
        raise ValueError(f"audit_manifest missing required columns: {missing}")
    if not required_predictions.issubset(predictions.columns):
        missing = sorted(required_predictions - set(predictions.columns))
        raise ValueError(f"predictions missing required columns: {missing}")

    manifest = audit_manifest.copy()
    path_column = "reporting_image_path" if "reporting_image_path" in manifest.columns else "image_path"
    manifest[path_column] = manifest[path_column].astype(str)
    manifest["image_path_abs"] = manifest[path_column].map(_normalize_path)
    manifest = manifest[manifest["include_for_reporting"]].copy()

    preds = predictions.copy()
    preds["image_path"] = preds["image_path"].astype(str)
    preds["image_path_abs"] = preds["image_path"].map(_normalize_path)
    for col in ("pred_2", "pred_3"):
        if col not in preds.columns:
            preds[col] = ""

    merged = manifest[
        ["image_path", "image_path_abs", "true_species", "difficulty_stratum", "cohort_version", "inclusion_reason"]
        if "cohort_version" in manifest.columns and "inclusion_reason" in manifest.columns
        else ["image_path", "image_path_abs", "true_species", "difficulty_stratum"]
    ].merge(
        preds[["image_path_abs", "pred_1", "pred_2", "pred_3"]],
        how="inner",
        on="image_path_abs",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "difficulty_stratum",
                "n_images",
                "top1_correct",
                "top1_acc",
                "top1_ci_low",
                "top1_ci_high",
                "top3_correct",
                "top3_acc",
                "genus_top1_correct",
                "genus_top1_acc",
            ]
        )

    merged["top1_correct"] = (merged["pred_1"].astype(str) == merged["true_species"].astype(str)).astype(int)
    merged["top3_correct"] = (
        merged[["pred_1", "pred_2", "pred_3"]].eq(merged["true_species"], axis=0).any(axis=1).astype(int)
    )
    merged["genus_top1_correct"] = (
        merged["pred_1"].astype(str).map(_genus) == merged["true_species"].astype(str).map(_genus)
    ).astype(int)

    ordered_groups = [
        ("overall", merged),
        ("normal", merged[merged["difficulty_stratum"] == "normal"].copy()),
        ("difficult", merged[merged["difficulty_stratum"] == "difficult"].copy()),
    ]

    rows: list[dict[str, float | int | str]] = []
    for group_name, group_df in ordered_groups:
        if group_df.empty:
            continue
        n_images = int(len(group_df))
        top1_correct = int(group_df["top1_correct"].sum())
        top3_correct = int(group_df["top3_correct"].sum())
        genus_top1_correct = int(group_df["genus_top1_correct"].sum())
        ci_low, ci_high = wilson_interval(top1_correct, n_images)

        rows.append(
            {
                "difficulty_stratum": group_name,
                "n_images": n_images,
                "top1_correct": top1_correct,
                "top1_acc": float(top1_correct) / float(n_images),
                "top1_ci_low": ci_low,
                "top1_ci_high": ci_high,
                "top3_correct": top3_correct,
                "top3_acc": float(top3_correct) / float(n_images),
                "genus_top1_correct": genus_top1_correct,
                "genus_top1_acc": float(genus_top1_correct) / float(n_images),
            }
        )

    return pd.DataFrame(rows)


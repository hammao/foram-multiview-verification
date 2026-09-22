from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import strftime

import pandas as pd
import torch
from PIL import Image

from src.inference.preprocess import _autocrop_otsu_bbox, _letterbox


def _species_to_genus(species: str) -> str:
    s = (species or "").strip()
    if not s:
        return ""
    return s.split()[0]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class LoadedExtractor:
    model: torch.nn.Module
    labels: list[str]
    input_size: int


def load_feature_extractor(checkpoints_dir: Path) -> LoadedExtractor:
    """
    Load a timm backbone as a feature extractor (classification head removed).

    This uses the classifier checkpoint weights but instantiates the model with num_classes=0,
    so it returns a pooled feature vector suitable for embedding similarity.
    """
    ckpt_path = checkpoints_dir / "classifier.pt"
    labels_path = checkpoints_dir / "label_map.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {ckpt_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing label map: {labels_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = str(ckpt["backbone"])
    input_size = int(ckpt.get("input_size", 384))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    import timm

    # Feature extractor: remove classifier head, keep pooled features.
    model = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
    # Load weights with head mismatches tolerated.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return LoadedExtractor(model=model, labels=labels, input_size=input_size)


def _embed_pil(
    im: Image.Image,
    *,
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Basic normalization compatible with timm defaults (ImageNet)
    import torchvision.transforms as T

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    im = _letterbox(im, input_size)
    x = tf(im).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    # Expect [B, D] -> [D]
    if feat.ndim == 2 and feat.shape[0] == 1:
        feat = feat[0]
    return feat.detach().float().cpu()


def embed_image_path(
    image_path: Path,
    *,
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    autocrop: str = "none",
) -> torch.Tensor:
    im = Image.open(image_path)
    if str(autocrop).lower().strip() == "otsu_bbox":
        im = _autocrop_otsu_bbox(im)
    return _embed_pil(im, model=model, input_size=input_size, device=device)


@dataclass(frozen=True)
class BuildIndexResult:
    index_path: Path
    meta_csv: Path
    n: int
    dim: int


def _l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=-1, keepdim=True), min=float(eps))


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [D] or [N,D], b: [M,D]
    a2 = _l2_normalize(a)
    b2 = _l2_normalize(b)
    return a2 @ b2.T


def build_embedding_index(
    *,
    manifest_csv: Path,
    checkpoints_dir: Path,
    out_dir: Path,
    image_col: str = "path_resized",
    autocrop_for_queries: str = "none",
    max_rows: int | None = None,
) -> BuildIndexResult:
    """
    Build an embedding index from the resized manifest (recommended).

    Writes:
    - embedding_index.pt: embeddings + centroids + OOD thresholds
    - embedding_meta.csv: per-row metadata used to build the index
    """
    df = pd.read_csv(manifest_csv)
    if df.empty:
        raise ValueError(f"Manifest is empty: {manifest_csv}")
    if "species" not in df.columns:
        raise ValueError("Manifest must contain 'species' column.")
    if image_col not in df.columns:
        raise ValueError(f"Manifest must contain '{image_col}' column.")
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "embedding_index.pt"
    meta_csv = out_dir / "embedding_meta.csv"

    loaded = load_feature_extractor(checkpoints_dir)
    ckpt_path = checkpoints_dir / "classifier.pt"
    ckpt_stat = None
    try:
        if ckpt_path.exists():
            st = ckpt_path.stat()
            ckpt_stat = {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        ckpt_stat = None
    device = _device()
    model = loaded.model.to(device)
    input_size = int(loaded.input_size)

    emb_list: list[torch.Tensor] = []
    species_list: list[str] = []
    genus_list: list[str] = []

    # Keep metadata columns if present
    keep_cols = [c for c in ["species", "specimen_id", "view_id", "source", "z_index", image_col] if c in df.columns]
    meta_out = df[keep_cols].copy()
    meta_out.rename(columns={image_col: "image_path"}, inplace=True)

    for _, r in df.iterrows():
        p = Path(str(r[image_col]))
        sp = str(r["species"])
        ge = _species_to_genus(sp)
        e = embed_image_path(p, model=model, input_size=input_size, device=device, autocrop="none")
        emb_list.append(e)
        species_list.append(sp)
        genus_list.append(ge)

    E = torch.stack(emb_list, dim=0)  # [N,D]
    E_n = _l2_normalize(E)
    N, D = int(E_n.shape[0]), int(E_n.shape[1])

    # Species centroids
    species_labels = sorted(set(species_list))
    sp_to_idx = {s: i for i, s in enumerate(species_labels)}
    sp_ids = torch.tensor([sp_to_idx[s] for s in species_list], dtype=torch.long)
    C = int(len(species_labels))
    sp_sum = torch.zeros((C, D), dtype=torch.float32)
    sp_cnt = torch.zeros((C,), dtype=torch.float32)
    sp_sum.index_add_(0, sp_ids, E_n)
    sp_cnt.index_add_(0, sp_ids, torch.ones((N,), dtype=torch.float32))
    sp_centroids = _l2_normalize(sp_sum / torch.clamp(sp_cnt.unsqueeze(1), min=1.0))

    # Genus centroids
    genus_labels = sorted(set(genus_list))
    g_to_idx = {g: i for i, g in enumerate(genus_labels)}
    g_ids = torch.tensor([g_to_idx[g] for g in genus_list], dtype=torch.long)
    G = int(len(genus_labels))
    g_sum = torch.zeros((G, D), dtype=torch.float32)
    g_cnt = torch.zeros((G,), dtype=torch.float32)
    g_sum.index_add_(0, g_ids, E_n)
    g_cnt.index_add_(0, g_ids, torch.ones((N,), dtype=torch.float32))
    g_centroids = _l2_normalize(g_sum / torch.clamp(g_cnt.unsqueeze(1), min=1.0))

    # OOD thresholds: distance to true species centroid for training items
    sims_true = (E_n * sp_centroids[sp_ids]).sum(dim=1)  # cosine similarity
    dists_true = (1.0 - sims_true).clamp(min=0.0, max=2.0)
    p95 = float(torch.quantile(dists_true, 0.95).item())
    p99 = float(torch.quantile(dists_true, 0.99).item())

    payload = {
        "created_at": strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_csv": str(manifest_csv),
        "checkpoints_dir": str(checkpoints_dir),
        "checkpoint_stat": ckpt_stat,
        "input_size": int(input_size),
        "embedding_dim": int(D),
        "autocrop_for_queries": str(autocrop_for_queries),
        "species_labels": species_labels,
        "genus_labels": genus_labels,
        "species_centroids": sp_centroids,
        "genus_centroids": g_centroids,
        "ood": {"p95_dist_to_true_species_centroid": p95, "p99_dist_to_true_species_centroid": p99},
    }
    torch.save(payload, index_path)
    meta_out.to_csv(meta_csv, index=False)

    return BuildIndexResult(index_path=index_path, meta_csv=meta_csv, n=N, dim=D)


def load_embedding_index(index_path: Path) -> dict:
    obj = torch.load(index_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid embedding index payload: {index_path}")
    return obj


def query_centroids(
    emb: torch.Tensor,
    *,
    index: dict,
    top_k: int = 10,
) -> dict:
    sp_centroids: torch.Tensor = index["species_centroids"]
    sp_labels: list[str] = index["species_labels"]
    g_centroids: torch.Tensor = index["genus_centroids"]
    g_labels: list[str] = index["genus_labels"]

    emb_n = _l2_normalize(emb.float().cpu())
    sp_sims = _cosine_sim(emb_n, sp_centroids).flatten()  # [C]
    g_sims = _cosine_sim(emb_n, g_centroids).flatten()  # [G]

    k = min(int(top_k), int(sp_sims.numel()))
    top_vals, top_idxs = torch.topk(sp_sims, k=k, largest=True, sorted=True)
    top_species = [{"species": sp_labels[int(i.item())], "cosine_similarity": float(v.item())} for v, i in zip(top_vals, top_idxs)]

    best_g = int(torch.argmax(g_sims).item()) if g_sims.numel() else -1
    best_g_label = g_labels[best_g] if best_g >= 0 else None
    best_g_sim = float(g_sims[best_g].item()) if best_g >= 0 else None

    # OOD: distance to nearest species centroid
    best_sp_sim = float(top_vals[0].item()) if k > 0 else float("nan")
    best_sp_dist = float(max(0.0, 1.0 - best_sp_sim))
    ood = index.get("ood", {}) if isinstance(index.get("ood", {}), dict) else {}

    return {
        "top_species": top_species,
        "nearest_species": top_species[0]["species"] if top_species else None,
        "nearest_species_cosine": best_sp_sim,
        "nearest_species_dist": best_sp_dist,
        "nearest_genus": best_g_label,
        "nearest_genus_cosine": best_g_sim,
        "ood_threshold_p95": float(ood.get("p95_dist_to_true_species_centroid", float("nan"))),
        "ood_threshold_p99": float(ood.get("p99_dist_to_true_species_centroid", float("nan"))),
    }


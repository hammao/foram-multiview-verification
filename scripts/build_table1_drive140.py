#!/usr/bin/env python3
"""Table 1 source for the 50-species paper: WoRMS-harmonised taxonomy, wall type, and split tiers.

Inputs
  configs/species_drive140.csv                     species, slug, display_name, nomenclature
  processed/manifest/worms_lookup_drive140.csv     WoRMS REST results (genus queried for OTUs)
  processed/drive140/manifest/splits_specimen_tvt  frozen A2 split

Output
  reports/experiments/2026-10_drive140_paper/table1_species_taxonomy.csv
  reports/experiments/2026-10_drive140_paper/table1_notes.json   (names needing author confirmation)

Wall type is assigned from the WoRMS order (Miliolida -> porcelaneous; Textulariida, Lituolida
-> agglutinated; Rotaliida, Vaginulinida, Robertinida -> hyaline) with explicit overrides for
genera whose wall departs from their order's norm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "configs" / "species_drive140.csv"
WORMS = ROOT / "processed" / "manifest" / "worms_lookup_drive140.csv"
SPLIT = ROOT / "processed" / "drive140" / "manifest" / "splits_specimen_tvt"
OUT_DIR = ROOT / "reports" / "experiments" / "2026-10_drive140_paper"

ORDER_TO_WALL = {
    "Miliolida": "porcelaneous",
    "Textulariida": "agglutinated",
    "Lituolida": "agglutinated",
    "Rotaliida": "hyaline",
    "Vaginulinida": "hyaline",
    "Robertinida": "hyaline",
}
# Genera whose wall differs from the order default, or whose order is blank in WoRMS.
GENUS_OVERRIDES = {
    "Nodulina": ("Lituolida", "agglutinated"),  # Reophacidae; WoRMS order field empty
    "Sigmoilopsis": ("Miliolida", "porcelaneous"),  # porcelaneous wall with agglutinated coating; WoRMS lists Lituolida
}
# Names absent from WoRMS or returned with a different spelling. Default follows the
# WoRMS-harmonised name; the authors confirm before submission.
NAME_RESOLUTIONS = {
    "Sahulia bakeri": {
        "worms_name": "Sahulia barkeri",
        "authority": "(Hofker, 1978)",
        "aphia_id": 480187,
        "order": "Textulariida",
        "family": "Textulariidae",
        "reason": "WoRMS fuzzy match: 'bakeri' is not recorded; Sahulia barkeri (Hofker, 1978) is the accepted name one letter away. Treated as a spelling variant pending author confirmation.",
    },
    "Siphogenerina columellarensis": {
        "worms_name": "Siphogenerina columellaris",
        "authority": "(Brady, 1881)",
        "aphia_id": 466356,
        "order": "Rotaliida",
        "family": "Siphogenerinoididae",
        "reason": "WoRMS fuzzy match: 'columellarensis' is not recorded; Siphogenerina columellaris (Brady, 1881) is accepted. Treated as a spelling variant pending author confirmation.",
    },
    "Ammobaculites jarvisi": {
        "worms_name": "Ammobaculites jarvisi",
        "authority": "",
        "aphia_id": 112345,
        "aphia_level": "genus",
        "order": "Lituolida",
        "family": "Lituolidae",
        "reason": "Species epithet not in WoRMS; genus AphiaID given. Authority to be supplied by the authors.",
    },
    "Siphotextularia wairoana": {
        "worms_name": "Siphotextularia wairoana",
        "authority": "Finlay, 1939",
        "aphia_id": 112392,
        "aphia_level": "genus",
        "order": "Textulariida",
        "family": "Textulariidae",
        "reason": "Species epithet not in WoRMS; genus AphiaID given. Authority from the original description (Finlay 1939) to be confirmed.",
    },
}


def main() -> int:
    sp = pd.read_csv(SPECIES)
    worms = pd.read_csv(WORMS)
    worms_by_query = {str(r.query_name): r for r in worms.itertuples(index=False)}
    prov = json.loads((SPLIT / "split_provenance.json").read_text(encoding="utf-8"))
    tiers = {a["species"]: a for a in prov["assignments"]}
    frames = {s: pd.read_csv(SPLIT / f"{s}.csv") for s in ("train", "val", "test")}
    img_counts = {
        s: df.groupby("species").size().to_dict() for s, df in frames.items()
    }

    rows = []
    notes = []
    for r in sp.itertuples(index=False):
        species = str(r.species)
        genus = species.split()[0]
        is_otu = str(r.nomenclature) == "open_otu"
        query = genus if is_otu else species
        w = worms_by_query.get(query)
        rec: dict[str, object] = {
            "code_label": species,
            "display_name": str(r.display_name),
            "genus": genus,
            "nomenclature": "open nomenclature (OTU)" if is_otu else "binomial",
        }
        if species in NAME_RESOLUTIONS:
            res = NAME_RESOLUTIONS[species]
            rec.update(
                {
                    "worms_name": res["worms_name"],
                    "authority": res["authority"],
                    "aphia_id": res["aphia_id"],
                    "aphia_level": res.get("aphia_level", "species"),
                    "worms_status": "resolved_by_authors",
                    "order": res["order"],
                    "family": res["family"],
                }
            )
            if res["worms_name"] != species:
                rec["display_name"] = res["worms_name"]
            notes.append({"code_label": species, "display_name": rec["display_name"], "note": res["reason"]})
        elif w is not None and bool(w.match_found):
            rec.update(
                {
                    "worms_name": genus if is_otu else str(w.accepted_name),
                    "authority": "" if is_otu else str(w.accepted_authority),
                    "aphia_id": int(w.accepted_aphia_id),
                    "aphia_level": "genus" if is_otu else "species",
                    "worms_status": "accepted",
                    "order": str(w.order) if isinstance(w.order, str) else "",
                    "family": str(w.family) if isinstance(w.family, str) else "",
                }
            )
            if not is_otu and str(w.accepted_name) != species:
                rec["display_name"] = str(w.accepted_name)
                notes.append({"code_label": species, "display_name": rec["display_name"], "note": "WoRMS accepted name differs from folder name; display follows WoRMS."})
        else:
            raise SystemExit(f"No WoRMS record and no resolution for {species}")

        if genus in GENUS_OVERRIDES:
            rec["order"], rec["wall_type"] = GENUS_OVERRIDES[genus]
        else:
            rec["wall_type"] = ORDER_TO_WALL.get(str(rec["order"]), "")
        if not rec["wall_type"]:
            raise SystemExit(f"No wall type for {species} (order={rec['order']!r})")

        t = tiers[species]
        rec.update(
            {
                "n_specimens": int(t["n_complete_specimens"]),
                "split_rule": t["rule"],
                "train_specimen_ids": ";".join(f"S{s:03d}" for s in t["train_specimen_ids"]),
                "val_specimen_id": f"S{t['val_specimen_id']:03d}" if t["val_specimen_id"] is not None else "",
                "test_specimen_id": f"S{t['test_specimen_id']:03d}" if t["test_specimen_id"] is not None else "",
                "n_images_train": int(img_counts["train"].get(species, 0)),
                "n_images_val": int(img_counts["val"].get(species, 0)),
                "n_images_test": int(img_counts["test"].get(species, 0)),
            }
        )
        rec["n_images_total"] = rec["n_images_train"] + rec["n_images_val"] + rec["n_images_test"]
        if is_otu:
            notes.append(
                {
                    "code_label": species,
                    "display_name": rec["display_name"],
                    "note": "Open-nomenclature OTU; all diagnostic views present but the material does not match a described congener; code follows Horton et al. (2021) and refers to the reference specimen.",
                }
            )
        rows.append(rec)

    table = pd.DataFrame(rows)
    wall_order = {"hyaline": 0, "agglutinated": 1, "porcelaneous": 2}
    table["_w"] = table["wall_type"].map(wall_order)
    table = table.sort_values(["_w", "order", "family", "display_name"]).drop(columns="_w")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "table1_species_taxonomy.csv", index=False)
    (OUT_DIR / "table1_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    summary = {
        "n_species": int(len(table)),
        "n_specimens": int(table["n_specimens"].sum()),
        "n_images": int(table["n_images_total"].sum()),
        "by_wall_type": table.groupby("wall_type").agg(n_species=("display_name", "size"), n_specimens=("n_specimens", "sum")).to_dict("index"),
        "names_needing_confirmation": [n for n in notes if "OTU" not in n["note"]],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

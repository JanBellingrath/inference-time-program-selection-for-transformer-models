"""Cross-benchmark anchor footpoint consistency for joint compositional catalogues.

If default routing anchors differ, primitives with identical (kind, args) can
touch different local states.  This module detects that via per-slot
``(position, anchor[position])`` tuples and optionally shrinks a joint bundle
(removing primitives / programs that are not 3-way consistent, remapping
incidence, dense tensors, and observed indices).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from data_prep.compositional.catalogue import (
    _save_incidence,
    _save_pair_incidence,
    build_incidence_tensor,
    build_pair_incidence_tensor,
)

logger = logging.getLogger("footpoint_filter")


def _slots_for_primitive(kind: str, args: Sequence[int]) -> List[int]:
    k = str(kind).lower()
    if k == "swap":
        return [int(args[0]), int(args[1])]
    if k in ("skip", "repeat"):
        return [int(args[0])]
    if k == "assign":
        return [int(args[0])]
    raise ValueError(f"unsupported primitive kind {kind!r}")


def footpoint(
    anchor: Sequence[int],
    kind: str,
    args: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    """Local routing state at all coordinates touched by the primitive."""
    slots = _slots_for_primitive(kind, args)
    a = list(anchor)
    return tuple((s, a[s]) for s in sorted(slots))


def anchors_list_identical(manifest: Dict[str, Any], *, bench_keys: Sequence[str]) -> bool:
    a0 = manifest["benchmarks"][bench_keys[0]]["anchor"]
    for b in bench_keys[1:]:
        if manifest["benchmarks"][b]["anchor"] != a0:
            return False
    return True


def inconsistent_primitive_indices(
    primitives: Sequence[Dict[str, Any]],
    anchors: Dict[str, Sequence[int]],
    *,
    bench_order: Sequence[str],
) -> Set[int]:
    bad: Set[int] = set()
    for p in primitives:
        kind = p["kind"]
        args = p["args"]
        fps = {b: footpoint(anchors[b], kind, args) for b in bench_order}
        uniq = {json.dumps(list(fps[b]), separators=(",", ":")) for b in bench_order}
        if len(uniq) > 1:
            bad.add(int(p["idx"]))
    return set(bad)


def filter_joint_bundle(
    bundle_dir: Path,
    out_dir: Path,
    *,
    bench_order: Optional[Sequence[str]] = None,
    force_filter: bool = False,
) -> Dict[str, Any]:
    """Write a footpoint-filtered copy of *bundle_dir* into *out_dir*.

    When all listed benchmark anchors are byte-identical and *force_filter* is
    false, copies the bundle unchanged and records *skipped_reason:* matching
    anchors.

    Otherwise drops any primitive whose footprint differs across benchmarks,
    drops legal programs that use a dropped primitive, re-indexes primitives
    and programs, rebuilds incidence / pair-incidence, slices dense tensors,
    and rewrites observed JSONL.
    """
    bundle_dir = Path(bundle_dir).resolve()
    out_dir = Path(out_dir).resolve()
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    benches = list(bench_order) if bench_order is not None else sorted(manifest["benchmarks"].keys())
    anchors = {b: tuple(manifest["benchmarks"][b]["anchor"]) for b in benches}

    meta: Dict[str, Any] = {"source_bundle": str(bundle_dir), "benchmarks": list(benches)}

    if not force_filter and anchors_list_identical(manifest, bench_keys=benches):
        logger.info("All benchmark anchors identical; copy bundle without footpoint filtering.")
        if out_dir != bundle_dir:
            shutil.copytree(bundle_dir, out_dir, dirs_exist_ok=True)
        meta["footpoint_filter_applied"] = False
        meta["skipped_reason"] = "anchors_identical"
        (out_dir / "footpoint_filter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    primitives = [json.loads(ln) for ln in (bundle_dir / manifest["primitives_path"]).read_text().splitlines() if ln.strip()]
    bad_prim = inconsistent_primitive_indices(primitives, anchors, bench_order=benches)

    joint_lp_rel = Path(manifest["joint"]["legal_programs_path"])
    lp_path = bundle_dir / joint_lp_rel
    rows = [json.loads(ln) for ln in lp_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: int(r["idx"]))

    kept_rows: List[Dict[str, Any]] = []
    for r in rows:
        pidx = [int(x) for x in r["primitive_indices"]]
        if set(pidx) & bad_prim:
            continue
        kept_rows.append(dict(r))

    if not kept_rows or not any(len(r.get("primitive_indices", [])) == 0 for r in kept_rows):
        raise RuntimeError("footpoint filter removed noop or emptied catalogue; check inputs.")

    old_prog_to_new = {int(r["idx"]): new_i for new_i, r in enumerate(kept_rows)}
    all_kept_prim = sorted({int(p) for r in kept_rows for p in r["primitive_indices"]})
    old_prim_to_new = {old: new_i for new_i, old in enumerate(all_kept_prim)}

    prim_by_old = {int(p["idx"]): p for p in primitives}
    new_primitives: List[Dict[str, Any]] = []
    for new_i, old_i in enumerate(all_kept_prim):
        rec = dict(prim_by_old[old_i])
        rec["idx"] = new_i
        new_primitives.append(rec)

    new_prim_lists: List[List[int]] = []
    new_keys: List[str] = []
    for r in kept_rows:
        sorted_idx = sorted(int(x) for x in r["primitive_indices"])
        new_prim_lists.append([old_prim_to_new[p] for p in sorted_idx])
        new_keys.append(str(r["key"]))

    a_idx, a_val, a_shape, lengths = build_incidence_tensor(new_prim_lists, len(all_kept_prim))
    pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(new_prim_lists)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_dir / "manifest.json", out_dir / "manifest.json.bak_pre_footpoint")
    prim_path = out_dir / "primitives.jsonl"
    with open(prim_path, "w") as f:
        for rec in new_primitives:
            f.write(json.dumps(rec, sort_keys=False) + "\n")

    lp_out = out_dir / joint_lp_rel
    lp_out.parent.mkdir(parents=True, exist_ok=True)
    with open(lp_out, "w") as f:
        for i, (key, plist) in enumerate(zip(new_keys, new_prim_lists)):
            f.write(
                json.dumps(
                    {"idx": i, "length": len(plist), "primitive_indices": plist, "key": key},
                )
                + "\n"
            )

    inc_rel = manifest["joint"]["incidence_path"]
    pair_rel = manifest["joint"]["pair_incidence_path"]
    (out_dir / Path(inc_rel).parent).mkdir(parents=True, exist_ok=True)
    _save_incidence(out_dir / inc_rel, a_idx, a_val, a_shape, lengths)
    (out_dir / Path(pair_rel).parent).mkdir(parents=True, exist_ok=True)
    _save_pair_incidence(out_dir / pair_rel, pair_index, b_idx, b_val, b_shape)

    old_cols = [int(r["idx"]) for r in kept_rows]
    for b in benches:
        binfo = manifest["benchmarks"][b]
        # observed
        obs_in = bundle_dir / binfo["observed_path"]
        obs_out = out_dir / binfo["observed_path"]
        obs_out.parent.mkdir(parents=True, exist_ok=True)
        n_out = 0
        with open(obs_in) as fi, open(obs_out, "w") as fo:
            for line in fi:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                oi = [int(x) for x in rec["obs_indices"]]
                od = [float(x) for x in rec["obs_deltas"]]
                ni, nd = [], []
                for i, d in zip(oi, od):
                    j = old_prog_to_new.get(i)
                    if j is not None:
                        ni.append(j)
                        nd.append(d)
                if not ni:
                    continue
                rec["obs_indices"] = ni
                rec["obs_deltas"] = nd
                rec["n_obs"] = len(ni)
                fo.write(json.dumps(rec) + "\n")
                n_out += 1
        # dense
        dpath = bundle_dir / binfo["dense_deltas_path"]
        payload = torch.load(dpath, map_location="cpu", weights_only=True)
        dm = payload["delta_matrix"].float()
        new_dm = dm[:, old_cols].clone()
        payload["delta_matrix"] = new_dm
        payload["n_programs"] = int(new_dm.shape[1])
        if "catalogue" in payload and isinstance(payload["catalogue"], dict):
            payload["catalogue"]["n_programs"] = int(new_dm.shape[1])
        dout = out_dir / binfo["dense_deltas_path"]
        dout.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, dout)
        mpath_rel = binfo.get("dense_keep_mask_path")
        if mpath_rel:
            mp = bundle_dir / mpath_rel
            if mp.is_file():
                mpayload = torch.load(mp, map_location="cpu", weights_only=True)
                km = mpayload["keep_mask"].float()
                mpayload["keep_mask"] = km[old_cols].clone()
                mo = out_dir / mpath_rel
                mo.parent.mkdir(parents=True, exist_ok=True)
                torch.save(mpayload, mo)

    # Patch manifest in memory
    new_manifest = json.loads(json.dumps(manifest))
    new_manifest["M"] = len(all_kept_prim)
    new_manifest["primitives_path"] = str(Path("primitives.jsonl"))
    new_manifest["catalogue_kind"] = str(new_manifest.get("catalogue_kind", "")) + "_footpoint_filtered"
    if "joint" in new_manifest:
        new_manifest["joint"]["n_programs"] = len(kept_rows)
        new_manifest["joint"]["n_pairs"] = int(pair_index.shape[0])
    per_bench_obs: Dict[str, int] = {}
    for b in benches:
        obs_p = out_dir / new_manifest["benchmarks"][b]["observed_path"]
        per_bench_obs[b] = sum(1 for _ in open(obs_p))

    for b in benches:
        bi = new_manifest["benchmarks"][b]
        bi["n_legal_programs"] = len(kept_rows)
        bi["n_legal_pairs"] = int(pair_index.shape[0])
        bi["n_questions_kept"] = per_bench_obs[b]
        bi["n_observed_pairs"] = 0

    new_manifest["footpoint_filter"] = {
        "inconsistent_primitive_indices_source": sorted(bad_prim),
        "n_primitives_before": len(primitives),
        "n_primitives_after": len(all_kept_prim),
        "n_programs_before": len(rows),
        "n_programs_after": len(kept_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")

    meta.update(
        {
            "footpoint_filter_applied": True,
            "n_bad_primitives": len(bad_prim),
            "programs_before": len(rows),
            "programs_after": len(kept_rows),
        },
    )
    (out_dir / "footpoint_filter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        "Footpoint filter: dropped %d primitives, %d programs remain (was %d).",
        len(bad_prim),
        len(kept_rows),
        len(rows),
    )
    return meta

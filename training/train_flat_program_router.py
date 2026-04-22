"""Train the :class:`FlatProgramRouter` baseline.

This baseline shares the question encoder (compressor + projection) with
the compositional router so that any difference in held-out compositional
generalisation can be attributed to the *output* representation
(``A u_q + B v_q − λ ℓ`` vs. an atomic per-program linear head) rather
than to the encoder capacity.

The loss is the same softmax cross-entropy over the observed candidate
support (or, optionally, the full legal set when supplied with dense
deltas) used by the compositional router.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from routers.compositional_router import (
    CompositionalDataset,
    LegalCatalogue,
    collate_compositional,
    load_artifacts,
    softmax_ce_on_observed,
)
from routers.flat_program_router import (
    FlatProgramRouter,
    program_scores_per_benchmark_flat,
)
from training.train_compositional_router import (
    _build_compressor,
    _infer_d_model,
    _maybe_swap_to_dense,
    _split_indices,
)

logger = logging.getLogger("train_flat_program_router")


# ---------------------------------------------------------------------------
# Loss + epoch
# ---------------------------------------------------------------------------


def _flat_loss_on_batch(
    router: FlatProgramRouter,
    batch: Dict[str, Any],
    bench_id_to_n_programs: Dict[int, int],
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    enc = batch["encoder_input"].to(device)
    attn = batch["attention_mask"]
    if attn is not None:
        attn = attn.to(device)
    g_q = router.encode(enc, attn)
    obs_indices = batch["obs_indices"].to(device)
    obs_deltas = batch["obs_deltas"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    program_scores = program_scores_per_benchmark_flat(
        router, g_q, batch["benchmark_id"].to(device),
        bench_id_to_n_programs=bench_id_to_n_programs,
        obs_indices=obs_indices,
    )
    return softmax_ce_on_observed(program_scores, obs_indices, obs_deltas, obs_mask, tau, student_temp=1.0)


def _epoch(
    router: FlatProgramRouter,
    loader: DataLoader,
    bench_id_to_n_programs: Dict[int, int],
    tau: float,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_dense_supervision: bool = False,
) -> float:
    is_train = optimizer is not None
    router.train(mode=is_train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = _maybe_swap_to_dense(batch, enabled=use_dense_supervision)
        loss = _flat_loss_on_batch(router, batch, bench_id_to_n_programs, tau, device)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = int(batch["benchmark_id"].numel())
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(total_count, 1)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_flat_router(
    catalogue_dir: _Path,
    output_path: _Path,
    *,
    benchmarks: Optional[Sequence[str]] = None,
    compressor_type: str = "last_token",
    compressor_d_compress: int = 256,
    compressor_n_heads: int = 4,
    compressor_n_latent: int = 1,
    encoder_hidden_dims: Sequence[int] = (),
    encoder_dropout: float = 0.1,
    freeze_compressor: bool = False,
    d_latent: int = 128,
    tau: float = 1.0,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    val_fraction: float = 0.15,
    seed: int = 42,
    use_full_sequence: bool = False,
    device: Optional[torch.device] = None,
    dense_delta_paths: Optional[Dict[str, _Path]] = None,
    use_dense_supervision: bool = False,
    observed_path_overrides: Optional[Dict[str, _Path]] = None,
    dense_keep_mask_paths: Optional[Dict[str, _Path]] = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(catalogue_dir, benchmarks=benchmarks)
    if not artifacts.catalogues:
        raise SystemExit("no benchmarks selected from artifacts")
    bench_list = list(artifacts.benchmarks)
    bench_to_id = {b: i for i, b in enumerate(sorted(bench_list))}

    dataset = CompositionalDataset(
        artifacts,
        benchmarks=bench_list,
        use_full_sequence=use_full_sequence,
        bench_to_id=bench_to_id,
        dense_delta_paths=dense_delta_paths if use_dense_supervision or dense_delta_paths else None,
        observed_path_overrides=observed_path_overrides,
        dense_keep_mask_paths=dense_keep_mask_paths,
    )
    if len(dataset) == 0:
        raise SystemExit("flat router: no samples after dataset construction.")

    train_idx, val_idx = _split_indices(len(dataset), val_fraction, seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        collate_fn=collate_compositional,
    )
    val_loader = (
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                   collate_fn=collate_compositional)
        if val_idx else None
    )

    d_model = _infer_d_model(dataset)
    compressor = _build_compressor(
        compressor_type, d_model,
        d_compress=compressor_d_compress,
        n_heads=compressor_n_heads,
        n_latent=compressor_n_latent,
        dropout=encoder_dropout,
    )
    router = FlatProgramRouter.from_artifacts(
        compressor=compressor,
        catalogues=artifacts.catalogues,
        bench_to_id=bench_to_id,
        d=d_latent,
        encoder_hidden_dims=encoder_hidden_dims,
        dropout=encoder_dropout,
        freeze_compressor=freeze_compressor,
    ).to(device)

    optimizer = torch.optim.AdamW(
        router.parameters(), lr=lr, weight_decay=weight_decay,
    )

    bench_id_to_n_programs = {
        int(bench_to_id[b]): int(artifacts.catalogues[b].n_programs) for b in bench_list
    }

    best_val = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        train_loss = _epoch(
            router, train_loader, bench_id_to_n_programs, tau, device,
            optimizer=optimizer, use_dense_supervision=use_dense_supervision,
        )
        val_loss = float("nan")
        if val_loader is not None:
            with torch.no_grad():
                val_loss = _epoch(
                    router, val_loader, bench_id_to_n_programs, tau, device,
                    use_dense_supervision=use_dense_supervision,
                )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
        elif val_loader is None and (best_epoch < 0 or train_loss < best_val):
            best_val = train_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
        logger.info("epoch=%d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

    if best_state is not None:
        router.load_state_dict(best_state)

    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": "flat_program_router",
        "state_dict": router.state_dict(),
        "config": {
            "compressor_type": compressor_type,
            "compressor_d_compress": compressor_d_compress,
            "compressor_n_heads": compressor_n_heads,
            "compressor_n_latent": compressor_n_latent,
            "encoder_hidden_dims": list(encoder_hidden_dims),
            "encoder_dropout": encoder_dropout,
            "freeze_compressor": freeze_compressor,
            "d_latent": d_latent,
            "tau": tau,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_fraction": val_fraction,
            "seed": seed,
            "use_full_sequence": use_full_sequence,
            "use_dense_supervision": bool(use_dense_supervision),
            "benchmarks": bench_list,
            "bench_to_id": bench_to_id,
            "bench_n_programs": {b: int(artifacts.catalogues[b].n_programs) for b in bench_list},
        },
        "best_epoch": best_epoch,
        "best_val": best_val,
        "history": history,
        "catalogue_dir": str(catalogue_dir),
    }
    torch.save(payload, output_path)
    logger.info("flat router saved -> %s (best epoch %d, val_loss %.4f)",
                output_path, best_epoch, best_val)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=_Path)
    p.add_argument("--output_path", required=True, type=_Path)
    p.add_argument("--benchmarks", nargs="*", default=None)

    p.add_argument("--compressor_type", choices=["last_token", "top_down_attention"],
                   default="last_token")
    p.add_argument("--compressor_d_compress", type=int, default=256)
    p.add_argument("--compressor_n_heads", type=int, default=4)
    p.add_argument("--compressor_n_latent", type=int, default=1)
    p.add_argument("--encoder_hidden_dims", nargs="*", type=int, default=[])
    p.add_argument("--encoder_dropout", type=float, default=0.1)
    p.add_argument("--freeze_compressor", action="store_true")
    p.add_argument("--d_latent", type=int, default=128)

    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_full_sequence", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--log_level", default="INFO")

    p.add_argument("--dense_deltas", nargs="*", default=None,
                   help="bench=path entries for dense delta matrices.")
    p.add_argument("--use_dense_supervision", action="store_true")
    p.add_argument("--observed_dir", default=None,
                   help="Override observed/{bench}.jsonl files.")
    p.add_argument("--dense_keep_mask_dir", default=None,
                   help="Directory of {bench}.pt keep_mask files.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if args.compressor_type != "last_token" and not args.use_full_sequence:
        raise SystemExit("--compressor_type top_down_attention requires --use_full_sequence")
    device = torch.device(args.device) if args.device else None

    dense_paths: Dict[str, _Path] = {}
    for entry in (args.dense_deltas or []):
        if "=" not in entry:
            raise SystemExit(f"--dense_deltas expects bench=path entries, got {entry!r}")
        bench, path = entry.split("=", 1)
        dense_paths[bench] = _Path(path)

    observed_overrides: Dict[str, _Path] = {}
    if args.observed_dir is not None:
        obs_dir = _Path(args.observed_dir)
        if not obs_dir.is_dir():
            raise SystemExit(f"--observed_dir not a directory: {obs_dir}")
        # Defer resolution to dataset; we just glob what is there.
        for path in sorted(obs_dir.glob("*.jsonl")):
            observed_overrides[path.stem] = path

    dense_keep_paths: Dict[str, _Path] = {}
    if args.dense_keep_mask_dir is not None:
        mdir = _Path(args.dense_keep_mask_dir)
        if not mdir.is_dir():
            raise SystemExit(f"--dense_keep_mask_dir not a directory: {mdir}")
        for path in sorted(mdir.glob("*.pt")):
            dense_keep_paths[path.stem] = path

    train_flat_router(
        catalogue_dir=args.catalogue_dir,
        output_path=args.output_path,
        benchmarks=args.benchmarks,
        compressor_type=args.compressor_type,
        compressor_d_compress=args.compressor_d_compress,
        compressor_n_heads=args.compressor_n_heads,
        compressor_n_latent=args.compressor_n_latent,
        encoder_hidden_dims=args.encoder_hidden_dims,
        encoder_dropout=args.encoder_dropout,
        freeze_compressor=args.freeze_compressor,
        d_latent=args.d_latent,
        tau=args.tau,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        use_full_sequence=args.use_full_sequence,
        device=device,
        dense_delta_paths=dense_paths,
        use_dense_supervision=bool(args.use_dense_supervision),
        observed_path_overrides=observed_overrides,
        dense_keep_mask_paths=dense_keep_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

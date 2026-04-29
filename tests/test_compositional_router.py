"""Acceptance tests for the first-order compositional router (Step 2)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import torch
import torch.nn.functional as F

from core.edit_dsl import (
    canonical_key_str,
    enumerate_admissible_programs,
)
from data_prep.build_compositional_catalogues import (
    build_incidence_tensor,
    build_legal_programs,
    build_pair_incidence_tensor,
)
from routers.compositional_router import (
    CompositionalRouter,
    LegalCatalogue,
    PairwiseScorer,
    PrimitiveSpec,
    compute_pair_relation_features,
    local_moebius_loss,
    N_RELATION_FEATURES,
    program_scores_from_primitive_scores,
    program_scores_with_pairs,
    softmax_ce_on_observed,
)
from routers.residual_compressors import LastTokenCompressor


# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------


TOY_ANCHOR = [0, 1, 2, 3, 4, 5]
TOY_GEOMETRY = {
    "K": 2,
    "swap_radius": 2,
    "editable_start": 0,
    "include_assign": False,
    "dedupe_assign_with_struct": False,
}


def _toy_artifacts():
    """Build O_train (all enumerated 1-program primitives) and the per-anchor catalogue."""
    one_step = [
        prog[0]
        for prog in enumerate_admissible_programs(
            TOY_ANCHOR,
            K=1,
            editable_indices=tuple(range(len(TOY_ANCHOR))),
            swap_radius=TOY_GEOMETRY["swap_radius"],
            include_assign=TOY_GEOMETRY["include_assign"],
        )
        if len(prog) == 1
    ]
    primitives = []
    key_to_idx = {}
    for j, p in enumerate(sorted(set(one_step), key=lambda q: (q.kind, q.args))):
        key = canonical_key_str((p,))
        primitives.append(PrimitiveSpec(idx=j, kind=p.kind, args=tuple(p.args), key=key))
        key_to_idx[key] = j

    programs, primitive_indices, n_dropped = build_legal_programs(
        TOY_ANCHOR, geometry=TOY_GEOMETRY, primitive_key_to_idx=key_to_idx,
    )
    assert n_dropped == 0
    M = len(primitives)
    a_idx, a_val, a_shape, lengths = build_incidence_tensor(primitive_indices, M)
    A = torch.sparse_coo_tensor(a_idx, a_val, size=a_shape).coalesce()
    pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(primitive_indices)
    if pair_index.numel() > 0:
        B = torch.sparse_coo_tensor(b_idx, b_val, size=b_shape).coalesce()
    else:
        B = None
    catalogue = LegalCatalogue(
        benchmark="toy", anchor=list(TOY_ANCHOR), A=A, lengths=lengths,
        B=B, pair_index=pair_index if pair_index.numel() > 0 else None,
    )
    return primitives, programs, primitive_indices, catalogue


# ---------------------------------------------------------------------------
# B. Program score equation
# ---------------------------------------------------------------------------


def test_program_score_equation_matches_per_row_sum():
    primitives, _programs, primitive_indices, catalogue = _toy_artifacts()
    M = len(primitives)
    N = catalogue.n_programs
    torch.manual_seed(0)
    s_q = torch.randn(3, M)
    lam = 0.25

    S_matrix = program_scores_from_primitive_scores(s_q, catalogue.A, catalogue.lengths, lam)
    assert S_matrix.shape == (3, N)

    expected = torch.empty(3, N)
    for r, prims in enumerate(primitive_indices):
        col_sum = s_q[:, prims].sum(dim=1) if prims else torch.zeros(3)
        expected[:, r] = col_sum - lam * len(prims)
    torch.testing.assert_close(S_matrix, expected, rtol=1e-6, atol=1e-6)
    # Empty program is the no-op row 0 with score 0.
    torch.testing.assert_close(S_matrix[:, 0], torch.zeros(3), rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# C. Loss normalises only over observed indices
# ---------------------------------------------------------------------------


def test_loss_normalises_only_over_observed_support():
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    M = len(primitives)
    N = catalogue.n_programs
    torch.manual_seed(1)
    s_q = torch.randn(2, M)
    lam = 0.1
    tau = 0.5
    S_q = program_scores_from_primitive_scores(s_q, catalogue.A, catalogue.lengths, lam)

    obs_a = torch.tensor([0, 3, 5])  # arbitrary subset
    obs_b = torch.tensor([1, 2])
    deltas_a = torch.tensor([0.0, 0.7, -0.2])
    deltas_b = torch.tensor([0.1, 0.4])

    K = max(len(obs_a), len(obs_b))
    obs_indices = torch.zeros(2, K, dtype=torch.long)
    obs_deltas = torch.zeros(2, K)
    obs_mask = torch.zeros(2, K)
    obs_indices[0, : len(obs_a)] = obs_a
    obs_indices[1, : len(obs_b)] = obs_b
    obs_deltas[0, : len(obs_a)] = deltas_a
    obs_deltas[1, : len(obs_b)] = deltas_b
    obs_mask[0, : len(obs_a)] = 1.0
    obs_mask[1, : len(obs_b)] = 1.0

    actual_loss = softmax_ce_on_observed(S_q, obs_indices, obs_deltas, obs_mask, tau)

    losses = []
    for row, obs, deltas in [(0, obs_a, deltas_a), (1, obs_b, deltas_b)]:
        scores_obs = S_q[row, obs]
        log_probs = F.log_softmax(scores_obs, dim=0)
        targets = F.softmax(deltas / tau, dim=0)
        losses.append(-(targets * log_probs).sum())
    expected_loss = torch.stack(losses).mean()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# E. Unseen legal program receives a finite score
# ---------------------------------------------------------------------------


def test_unseen_legal_program_is_scoreable_after_partial_training():
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    d_model = 8
    compressor = LastTokenCompressor(d_model=d_model)
    router = CompositionalRouter(
        primitives=primitives,
        compressor=compressor,
        d=4,
        num_positions=len(TOY_ANCHOR),
        encoder_hidden_dims=[],
        dropout=0.0,
        use_id_embedding=True,
        edit_hidden_dims=(8,),
        edit_dropout=0.0,
        unary_hidden_dims=(8,),
        unary_dropout=0.0,
    )

    torch.manual_seed(2)
    x = torch.randn(2, d_model)

    # Train a few micro steps using only a tiny observed support; the unseen
    # row 4 must still produce a finite score from the forward pass.
    obs_indices = torch.tensor([[0, 1], [0, 2]])
    obs_deltas = torch.tensor([[0.0, 0.5], [0.0, -0.3]])
    obs_mask = torch.ones_like(obs_deltas)
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-2)
    for _ in range(3):
        s_q = router.primitive_scores(x)
        S_q = router.program_scores(s_q, catalogue, lam=0.05)
        loss = softmax_ce_on_observed(S_q, obs_indices, obs_deltas, obs_mask, tau=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        s_q = router.primitive_scores(x)
        S_q = router.program_scores(s_q, catalogue, lam=0.05)
    unseen_row = 4
    assert unseen_row not in {0, 1, 2}
    assert torch.isfinite(S_q[:, unseen_row]).all()
    assert S_q.shape == (2, catalogue.n_programs)


# ---------------------------------------------------------------------------
# Pairwise extension
# ---------------------------------------------------------------------------


def test_pair_incidence_matches_program_pairs():
    """Each row of B contains exactly C(|e|,2) ones; every pair_index entry has i<j."""
    _primitives, _programs, primitive_indices, catalogue = _toy_artifacts()
    assert catalogue.B is not None and catalogue.pair_index is not None
    B_dense = catalogue.B.to_dense()
    expected_per_row = torch.tensor(
        [len(p) * (len(p) - 1) // 2 for p in primitive_indices], dtype=torch.float32,
    )
    assert torch.equal(B_dense.sum(dim=1), expected_per_row)
    pi = catalogue.pair_index
    assert (pi[:, 0] < pi[:, 1]).all()
    # Pair universe is unique.
    pairs_as_tuples = {(int(a), int(b)) for a, b in pi.tolist()}
    assert len(pairs_as_tuples) == pi.shape[0]


def test_pair_scorer_is_symmetric():
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    assert catalogue.pair_index is not None and catalogue.pair_index.shape[0] >= 2
    M = len(primitives)
    d_phi = 6
    d_z = 5
    torch.manual_seed(7)
    Phi = torch.randn(M, d_phi)
    g_q = torch.randn(2, d_z)
    rel = compute_pair_relation_features(
        primitives, catalogue.pair_index, num_positions=len(TOY_ANCHOR),
    )
    assert rel.shape == (catalogue.pair_index.shape[0], N_RELATION_FEATURES)

    scorer = PairwiseScorer(d_z=d_z, d_phi=d_phi, d_r=N_RELATION_FEATURES,
                            hidden_dims=(8, 8), dropout=0.0, zero_init_last=False)
    scorer.eval()
    v_orig = scorer(g_q, Phi, catalogue.pair_index, rel)

    swapped = catalogue.pair_index.flip(dims=[1])
    v_swap = scorer(g_q, Phi, swapped, rel)
    torch.testing.assert_close(v_orig, v_swap, rtol=1e-6, atol=1e-6)


def test_program_score_with_pairs_matches_per_row_sum():
    """``S_q = A u + B v − λℓ`` equals the row-wise sum-of-primitives-plus-pairs."""
    _primitives, _programs, primitive_indices, catalogue = _toy_artifacts()
    assert catalogue.B is not None
    M = len(_primitives)
    P = catalogue.pair_index.shape[0]
    torch.manual_seed(3)
    u_q = torch.randn(2, M)
    v_q = torch.randn(2, P)
    lam = 0.2

    S = program_scores_with_pairs(u_q, v_q, catalogue.A, catalogue.B, catalogue.lengths, lam)

    # Map each unordered pair → pair index.
    pair_to_id = {tuple(sorted((int(a), int(b)))): r
                  for r, (a, b) in enumerate(catalogue.pair_index.tolist())}
    expected = torch.empty_like(S)
    for r, prims in enumerate(primitive_indices):
        unary = u_q[:, prims].sum(dim=1) if prims else torch.zeros(2)
        if len(prims) >= 2:
            pair_ids = []
            for ai in range(len(prims)):
                for bi in range(ai + 1, len(prims)):
                    pair_ids.append(pair_to_id[tuple(sorted((int(prims[ai]), int(prims[bi]))))])
            pair_term = v_q[:, pair_ids].sum(dim=1)
        else:
            pair_term = torch.zeros(2)
        expected[:, r] = unary + pair_term - lam * len(prims)
    torch.testing.assert_close(S, expected, rtol=1e-6, atol=1e-6)


def test_pair_router_pair_off_collapses_to_unary():
    """A router with use_pairs=False matches the unary baseline exactly."""
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    d_model = 8
    compressor = LastTokenCompressor(d_model=d_model)
    router = CompositionalRouter(
        primitives=primitives, compressor=compressor, d=4,
        num_positions=len(TOY_ANCHOR), encoder_hidden_dims=[], dropout=0.0,
        use_pairs=False,
    )
    torch.manual_seed(4)
    x = torch.randn(2, d_model)
    s_q = router.primitive_scores(x)
    S_unary = program_scores_from_primitive_scores(s_q, catalogue.A, catalogue.lengths, lam=0.1)
    S_via_router = router.program_scores(s_q, catalogue, lam=0.1)
    torch.testing.assert_close(S_via_router, S_unary, rtol=1e-6, atol=1e-6)


def test_pair_router_zero_init_starts_equal_to_unary():
    """Zero-init pair MLP produces the same scores as the unary baseline at step 0."""
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    d_model = 8
    compressor = LastTokenCompressor(d_model=d_model)
    router = CompositionalRouter(
        primitives=primitives, compressor=compressor, d=4,
        num_positions=len(TOY_ANCHOR), encoder_hidden_dims=[], dropout=0.0,
        use_pairs=True, pair_hidden_dims=(8,), pair_dropout=0.0, pair_zero_init=True,
    )
    router.attach_pair_features([catalogue])
    router.eval()

    torch.manual_seed(5)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    v_q = router.pair_scores_from_g(g_q, catalogue)
    assert v_q is not None
    torch.testing.assert_close(v_q, torch.zeros_like(v_q), rtol=0, atol=1e-7)
    S_pair = router.program_scores(u_q, catalogue, lam=0.1, v_q=v_q)
    S_unary = router.program_scores(u_q, catalogue, lam=0.1, v_q=None)
    torch.testing.assert_close(S_pair, S_unary, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Architecture refactor: edit MLP + unary MLP scorer
# ---------------------------------------------------------------------------


def _toy_router(**kwargs):
    primitives, _programs, _prim_indices, catalogue = _toy_artifacts()
    d_model = 8
    compressor = LastTokenCompressor(d_model=d_model)
    defaults = dict(
        primitives=primitives, compressor=compressor, d=4,
        num_positions=len(TOY_ANCHOR), encoder_hidden_dims=[], dropout=0.0,
        edit_hidden_dims=(8,), edit_dropout=0.0,
        unary_hidden_dims=(8,), unary_dropout=0.0,
    )
    defaults.update(kwargs)
    router = CompositionalRouter(**defaults)
    return router, primitives, catalogue, d_model


def test_unary_scorer_is_mlp_not_dot_product():
    """Unary scores are not equal to g_q @ Phi.T (dot product); they come
    from the MLP scorer over [z_q, phi_j]."""
    router, _primitives, _catalogue, d_model = _toy_router()
    router.eval()
    torch.manual_seed(11)
    x = torch.randn(3, d_model)
    g_q = router.encode(x)
    Phi = router.phi()
    u_mlp = router.primitive_scores_from_g(g_q)        # [B, M]
    u_dot = g_q @ Phi.t()                              # [B, M]
    assert u_mlp.shape == u_dot.shape
    diff = (u_mlp - u_dot).abs().max().item()
    assert diff > 1e-3, "unary scorer collapsed to a dot product"


def test_dot_unary_scorer_matches_closed_form():
    """Legacy dot scorer path computes ``g @ Phi.T (+ b)`` exactly."""
    router, _primitives, _catalogue, d_model = _toy_router(
        unary_scorer_type="dot",
        primitive_bias=True,
    )
    router.eval()
    torch.manual_seed(21)
    x = torch.randn(3, d_model)
    g_q = router.encode(x)
    Phi = router.phi()
    u_dot = router.primitive_scores_from_g(g_q)
    expected = g_q @ Phi.t() + router.unary_scorer.bias
    torch.testing.assert_close(u_dot, expected, rtol=1e-6, atol=1e-6)


def test_pair_scorer_uses_post_mlp_phi():
    """Gradient from a pair-only loss flows back into the edit MLP."""
    router, _primitives, catalogue, d_model = _toy_router(
        use_pairs=True, pair_hidden_dims=(8,), pair_dropout=0.0,
        pair_zero_init=False,
    )
    router.attach_pair_features([catalogue])
    router.train()
    torch.manual_seed(12)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    v_q = router.pair_scores_from_g(g_q, catalogue)
    assert v_q is not None and v_q.numel() > 0
    loss = (v_q ** 2).mean()
    loss.backward()
    edit_grads = [p.grad for p in router.phi_enc.parameters() if p.grad is not None]
    assert edit_grads, "edit MLP has no parameters with gradients"
    norms = [g.abs().sum().item() for g in edit_grads]
    assert max(norms) > 0.0, "edit MLP received zero gradient from pair loss"


def test_topk_pair_pruning_equivalence_at_full_k():
    """With k >= M the top-K masked v_q equals the full v_q exactly."""
    router, _primitives, catalogue, d_model = _toy_router(
        use_pairs=True, pair_hidden_dims=(8,), pair_dropout=0.0,
        pair_zero_init=False,
    )
    router.attach_pair_features([catalogue])
    router.eval()
    torch.manual_seed(13)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    v_full = router.pair_scores_from_g(g_q, catalogue)
    v_topk = router.pair_scores_from_g_topk(g_q, u_q, catalogue, k=router.M)
    torch.testing.assert_close(v_full, v_topk, rtol=0, atol=0)


def test_topk_pair_pruning_zeros_out_of_shortlist_pairs():
    """With small k pairs whose endpoints aren't in top-K are zeroed."""
    router, _primitives, catalogue, d_model = _toy_router(
        use_pairs=True, pair_hidden_dims=(8,), pair_dropout=0.0,
        pair_zero_init=False,
    )
    router.attach_pair_features([catalogue])
    router.eval()
    torch.manual_seed(14)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    v_full = router.pair_scores_from_g(g_q, catalogue)
    k = 3
    v_topk = router.pair_scores_from_g_topk(g_q, u_q, catalogue, k=k)
    assert v_topk is not None and v_full is not None
    topk_idx = u_q.topk(k, dim=-1).indices
    in_shortlist = torch.zeros(u_q.shape, dtype=torch.bool)
    in_shortlist.scatter_(1, topk_idx, True)
    pi = catalogue.pair_index
    keep = in_shortlist.index_select(1, pi[:, 0]) & in_shortlist.index_select(1, pi[:, 1])
    expected = v_full * keep.to(v_full.dtype)
    torch.testing.assert_close(v_topk, expected, rtol=0, atol=0)
    assert (v_topk[~keep].abs() == 0).all()


# ---------------------------------------------------------------------------
# Local Möbius supervision
# ---------------------------------------------------------------------------


def test_local_moebius_loss_returns_none_when_targets_absent():
    """A batch without local_unary/local_pair keys yields total=None."""
    router, _primitives, _catalogue, d_model = _toy_router()
    router.eval()
    torch.manual_seed(15)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    out = local_moebius_loss(
        router=router, g_q=g_q, u_q=u_q, batch={},
        use_unary=True, use_pair=True,
    )
    assert out["unary"] is None
    assert out["pair"] is None
    assert out["total"] is None


def test_local_moebius_loss_unary_is_finite_and_zero_at_perfect_targets():
    """With targets equal to current unary scores the loss is exactly 0."""
    router, _primitives, _catalogue, d_model = _toy_router()
    router.eval()
    torch.manual_seed(16)
    x = torch.randn(3, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    M = router.M
    rows = torch.tensor([0, 1, 2, 0])
    js = torch.tensor([1, 2, 3 % M, 0])
    targets = u_q[rows, js].detach().clone()
    batch = {
        "local_unary": {"row": rows, "j": js, "target": targets},
    }
    out = local_moebius_loss(
        router=router, g_q=g_q, u_q=u_q, batch=batch,
        use_unary=True, use_pair=False,
    )
    assert out["unary"] is not None
    assert torch.isfinite(out["unary"])
    torch.testing.assert_close(out["unary"], torch.zeros_like(out["unary"]),
                               rtol=0, atol=1e-6)


def test_local_moebius_loss_pair_supervision_runs_and_grads_flow():
    """Pair local loss runs end-to-end and produces gradients for the pair scorer."""
    router, _primitives, catalogue, d_model = _toy_router(
        use_pairs=True, pair_hidden_dims=(8,), pair_dropout=0.0,
        pair_zero_init=False,
    )
    router.attach_pair_features([catalogue])
    router.train()
    torch.manual_seed(17)
    x = torch.randn(2, d_model)
    g_q = router.encode(x)
    u_q = router.primitive_scores_from_g(g_q)
    pi = catalogue.pair_index
    assert pi is not None and pi.shape[0] >= 2
    pair_rows = torch.tensor([0, 1, 0])
    pair_i = pi[:3, 0].clone()
    pair_j = pi[:3, 1].clone()
    pair_target = torch.tensor([0.5, -0.3, 0.1])
    batch = {
        "local_pair": {
            "row": pair_rows, "i": pair_i, "j": pair_j, "target": pair_target,
        }
    }
    out = local_moebius_loss(
        router=router, g_q=g_q, u_q=u_q, batch=batch,
        use_unary=False, use_pair=True, pair_weight=1.0,
    )
    assert out["pair"] is not None
    assert out["unary"] is None
    assert torch.isfinite(out["pair"])
    out["pair"].backward()
    pair_grads = [p.grad for p in router.pair_scorer.parameters() if p.grad is not None]
    assert pair_grads, "pair scorer has no gradients"
    assert max(g.abs().sum().item() for g in pair_grads) > 0.0

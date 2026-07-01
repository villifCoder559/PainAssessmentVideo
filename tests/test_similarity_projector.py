import unittest
import sys
import os

import numpy as np
import torch

# Add root to path to import cross_space_projection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cross_space_projection import (
    SimilarityProjector,
    _build_similarity_projector,
    _fit_procrustes_solution,
)


def _semi_ortho_err(W):
    """Max abs deviation from semi-orthogonality of a (m, n) matrix W (checks the smaller side)."""
    m, n = W.shape
    I = (W @ W.t()) if m <= n else (W.t() @ W)
    k = min(m, n)
    return float((I - torch.eye(k, dtype=W.dtype)).abs().max())


class TestSimilarityProjector(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def _make_fit(self, d_old, d_new, K=80):
        """Closed-form procrustes fit on random anchors; returns (A, B, sol, procrustes_params)."""
        A = np.random.randn(K, d_old).astype(np.float32)
        B = np.random.randn(K, d_new).astype(np.float32)
        sol = _fit_procrustes_solution(A, B)
        pp = {'mu_old': sol['mu_A'], 'mu_new': sol['mu_B'],
              'scale': sol['scale'], 'R': sol['R']}
        return A, B, sol, pp

    def test_init_matches_closed_form(self):
        """At init the structured forward must reproduce s*(x - mu_A) @ R + mu_B (both orientations)."""
        for d_old, d_new in [(8, 4), (4, 8), (6, 6)]:
            A, _, sol, pp = self._make_fit(d_old, d_new)
            proj = _build_similarity_projector(pp, 'cpu')  # also exercises the internal assert
            with torch.no_grad():
                y_mod = proj(torch.from_numpy(A)).numpy().astype(np.float64)
            y_ref = sol['scale'] * ((A.astype(np.float64) - sol['mu_A']) @ sol['R']) + sol['mu_B']
            rel = np.abs(y_mod - y_ref).max() / max(np.abs(y_ref).max(), 1e-12)
            self.assertLess(rel, 1e-4, f"init mismatch for {d_old}->{d_new}: rel={rel:.2e}")

    def test_to_linear_equivalence(self):
        """to_linear() must be an exact affine equivalent of the structured forward."""
        for d_old, d_new in [(8, 4), (4, 8)]:
            A, _, _, pp = self._make_fit(d_old, d_new)
            proj = _build_similarity_projector(pp, 'cpu')
            lin = proj.to_linear()
            x = torch.from_numpy(A)
            with torch.no_grad():
                err = (lin(x) - proj(x)).abs().max().item()
            self.assertLess(err, 1e-4, f"to_linear mismatch for {d_old}->{d_new}: {err:.2e}")
            # Persisted form must be a plain nn.Linear state_dict (weight/bias) the loaders expect.
            self.assertEqual(set(lin.state_dict().keys()), {'weight', 'bias'})

    def test_orthogonality_preserved_under_optimization(self):
        """After gradient steps R stays (semi-)orthogonal and the scale stays a single scalar."""
        for d_old, d_new in [(8, 4), (4, 8), (6, 6)]:
            A, B, _, pp = self._make_fit(d_old, d_new)
            proj = _build_similarity_projector(pp, 'cpu')
            x, tgt = torch.from_numpy(A), torch.from_numpy(B)
            opt = torch.optim.AdamW(proj.parameters(), lr=1e-2)
            for _ in range(60):
                opt.zero_grad()
                ((proj(x) - tgt) ** 2).mean().backward()
                opt.step()
            err = _semi_ortho_err(proj.rot.weight.detach())
            self.assertLess(err, 1e-3, f"orthogonality lost for {d_old}->{d_new}: {err:.2e}")
            # Scale is one scalar parameter, so every singular value of (s*R) equals exp(log_scale).
            sR = (torch.exp(proj.log_scale.detach()) * proj.rot.weight.detach().t())  # (d_old, d_new)
            sv = torch.linalg.svdvals(sR)
            self.assertLess(float((sv - sv.mean()).abs().max()), 1e-3,
                            f"scale not isotropic for {d_old}->{d_new}")

    def test_plain_linear_baseline_breaks_structure(self):
        """A plain nn.Linear warm-started identically does NOT preserve orthogonality (contrast)."""
        d_old, d_new = 8, 4
        A, B, _, pp = self._make_fit(d_old, d_new)
        lin = _build_similarity_projector(pp, 'cpu').to_linear()  # same start point, unconstrained
        x, tgt = torch.from_numpy(A), torch.from_numpy(B)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
        for _ in range(60):
            opt.zero_grad()
            ((lin(x) - tgt) ** 2).mean().backward()
            opt.step()
        # Normalize out the (unknown) scale, then measure orthogonality of the rows.
        W = lin.weight.detach()
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.assertGreater(_semi_ortho_err(Wn), 1e-2,
                           "plain Linear unexpectedly stayed orthogonal")

    def test_deepcopy_and_state_dict_roundtrip(self):
        """deepcopy and state_dict round-trips (used for best-epoch tracking) must be exact."""
        import copy
        A, _, _, pp = self._make_fit(8, 4)
        proj = _build_similarity_projector(pp, 'cpu')
        x = torch.from_numpy(A)
        proj2 = copy.deepcopy(proj)
        sd = {k: v.clone() for k, v in proj.state_dict().items()}
        proj.load_state_dict(sd)
        with torch.no_grad():
            self.assertLess((proj2(x) - proj(x)).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()

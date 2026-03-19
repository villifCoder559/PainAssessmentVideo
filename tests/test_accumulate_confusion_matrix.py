"""
Tests for _accumulate_confusion_matrices in new_plot_res_from_server.py.

Each confusion matrix has shape (N+1)×(N+1) where:
  - rows/cols 0..N-1 are valid class labels
  - the last row/col (index N) is the out-of-range (OOR) bin

The merge must:
  1. Sum valid-class blocks aligned by class index (zero-pad missing classes)
  2. Sum OOR columns correctly (OOR col of smaller matrix must NOT alias
     into a valid-class column of the larger matrix)
  3. Keep the OOR row as all-zeros in the output
"""
import sys
import os
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── local copy of the function under test ─────────────────────────────────────
# Mirrors _accumulate_confusion_matrices from new_plot_res_from_server.py so
# the tests have no plotting / heavy-import dependencies.

def _accumulate_confusion_matrices(accum, t):
  """
  Merge two raw-count confusion matrices that may have different sizes.

  Each matrix is assumed to have the format (N+1)×(N+1) where the last row and
  column represent an out-of-range (OOR) bin.  When the matrices differ in size,
  the valid-class block ([:-1, :-1]) and OOR column ([:-1, -1]) of the smaller
  matrix are zero-padded to match the larger one before summing.  The OOR row is
  always set to zero in the output (true labels are never OOR).

  Args:
    accum: Running-sum tensor of dtype long, or None on the first call.
    t:     New fold's raw-count confusion matrix (long tensor, (N+1)×(N+1)).

  Returns:
    Merged (max_N+1) × (max_N+1) long tensor.
  """
  if accum is None:
    return t
  if accum.shape == t.shape:
    return accum + t

  a_valid, a_oor = accum[:-1, :-1], accum[:-1, -1]
  t_valid, t_oor = t[:-1, :-1],     t[:-1, -1]

  s_a, s_t = a_valid.shape[0], t_valid.shape[0]
  max_s = max(s_a, s_t)

  if s_a < max_s:
    pad = max_s - s_a
    a_valid = F.pad(a_valid, (0, pad, 0, pad))
    a_oor   = F.pad(a_oor,   (0, pad))
  if s_t < max_s:
    pad = max_s - s_t
    t_valid = F.pad(t_valid, (0, pad, 0, pad))
    t_oor   = F.pad(t_oor,   (0, pad))

  merged_oor = (a_oor + t_oor).unsqueeze(1)
  top    = torch.cat([a_valid + t_valid, merged_oor], dim=1)
  bottom = torch.zeros(1, max_s + 1, dtype=accum.dtype)
  return torch.cat([top, bottom], dim=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_matrix(n_valid, seed=42):
  """
  Create a (n_valid+1)×(n_valid+1) confusion matrix with random counts.
  The OOR row (last row) is forced to zero to simulate real data.

  Args:
    n_valid: Number of valid classes.
    seed:    Random seed for reproducibility.

  Returns:
    Long tensor of shape (n_valid+1, n_valid+1).
  """
  torch.manual_seed(seed)
  m = torch.randint(0, 10, (n_valid + 1, n_valid + 1), dtype=torch.long)
  m[-1, :] = 0
  return m


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAccumulateConfusionMatrices(unittest.TestCase):

  def test_same_size(self):
    """Same-size matrices: result equals element-wise sum."""
    a = _make_matrix(9, seed=0)
    b = _make_matrix(9, seed=1)
    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (10, 10))
    self.assertTrue(torch.equal(result, a + b))

  def test_small_plus_big(self):
    """10×10 + 12×12 → 12×12 with correct class alignment and OOR placement."""
    # 10×10: 9 valid classes (0-8), OOR bin at index 9
    a = torch.zeros(10, 10, dtype=torch.long)
    a[0, 0] = 5   # class 0 true → class 0 pred
    a[0, 9] = 2   # class 0 true → OOR pred
    a[8, 8] = 3   # class 8 true → class 8 pred

    # 12×12: 11 valid classes (0-10), OOR bin at index 11
    b = torch.zeros(12, 12, dtype=torch.long)
    b[0, 0]  = 7   # class 0 true → class 0 pred
    b[0, 11] = 1   # class 0 true → OOR pred
    b[9, 9]  = 4   # class 9 true → class 9 pred (not in a)
    b[10, 10]= 6   # class 10 true → class 10 pred (not in a)

    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (12, 12))
    self.assertEqual(result[0, 0].item(),  12)   # 5+7
    self.assertEqual(result[0, 11].item(),  3)   # OOR: 2+1 at col 11
    self.assertEqual(result[8, 8].item(),   3)   # only from a
    self.assertEqual(result[9, 9].item(),   4)   # only from b
    self.assertEqual(result[10, 10].item(), 6)   # only from b
    self.assertTrue(torch.all(result[-1, :] == 0), "OOR row must be zero")

  def test_big_plus_small(self):
    """12×12 + 10×10 → same result as small + big (commutativity)."""
    a = torch.zeros(10, 10, dtype=torch.long)
    a[0, 0] = 5; a[0, 9] = 2; a[8, 8] = 3

    b = torch.zeros(12, 12, dtype=torch.long)
    b[0, 0] = 7; b[0, 11] = 1; b[9, 9] = 4; b[10, 10] = 6

    self.assertTrue(torch.equal(
      _accumulate_confusion_matrices(a, b),
      _accumulate_confusion_matrices(b, a),
    ))

  def test_oor_not_aliased_to_valid_class(self):
    """
    Critical: OOR column of the smaller matrix must land at the last column
    of the result, NOT at the same column index as a valid class in the larger.

    a (10×10): a[0, 9] = 99  → class 0 predicted as OOR
    b (12×12): b[0, 9] = 1   → class 0 predicted as class 9 (valid)

    In the result: result[0, 9] should be 1 (class 9 pred from b only),
                   result[0, 11] should be 99 (OOR pred from a only).
    """
    a = torch.zeros(10, 10, dtype=torch.long)
    a[0, 9] = 99

    b = torch.zeros(12, 12, dtype=torch.long)
    b[0, 9] = 1

    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result[0, 9].item(),  1,  "col 9 should be class-9 pred from b only")
    self.assertEqual(result[0, 11].item(), 99, "col 11 (OOR) should hold a's OOR predictions")

  def test_first_call_returns_t_unchanged(self):
    """_accumulate_confusion_matrices(None, t) returns t unmodified."""
    t = _make_matrix(9)
    result = _accumulate_confusion_matrices(None, t)
    self.assertTrue(torch.equal(result, t))

  def test_three_way_accumulation(self):
    """Three-way accumulation is associative and produces correct output shape."""
    # a: 8 valid → 9×9; b: 10 valid → 11×11; c: 9 valid → 10×10
    # max valid = 10 (from b) → result is 11×11
    a = _make_matrix(8,  seed=0)
    b = _make_matrix(10, seed=1)
    c = _make_matrix(9,  seed=2)

    result_abc = _accumulate_confusion_matrices(_accumulate_confusion_matrices(a, b), c)
    result_acb = _accumulate_confusion_matrices(_accumulate_confusion_matrices(a, c), b)

    self.assertEqual(result_abc.shape, (11, 11))
    self.assertTrue(torch.equal(result_abc, result_acb), "three-way accumulation should be order-independent")

  def test_oor_row_always_zero(self):
    """OOR row of the result is all-zeros regardless of input sizes."""
    pairs = [
      (_make_matrix(7 + i, seed=i), _make_matrix(9 + i, seed=i + 10))
      for i in range(5)
    ]
    for a, b in pairs:
      result = _accumulate_confusion_matrices(a, b)
      self.assertTrue(torch.all(result[-1, :] == 0),
                      f"OOR row not zero for shapes {a.shape} + {b.shape}")

  def test_all_zeros_small_plus_nonzero_big(self):
    """Adding a zero matrix (small) to a nonzero one (big): result = padded big."""
    # a: 9 valid classes → 10×10; b: 10 valid classes → 11×11
    # max valid = 10 → result is 11×11
    a = torch.zeros(10, 10, dtype=torch.long)

    b = _make_matrix(10, seed=7)   # 11×11

    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (11, 11))
    # valid block should match b's valid block
    self.assertTrue(torch.equal(result[:-1, :-1], b[:-1, :-1]))
    # OOR column should match b's OOR column
    self.assertTrue(torch.equal(result[:-1, -1], b[:-1, -1]))
    self.assertTrue(torch.all(result[-1, :] == 0))

  def test_minimum_size(self):
    """2×2 matrices (1 valid class + OOR): no index errors."""
    a = torch.tensor([[3, 1], [0, 0]], dtype=torch.long)
    b = torch.tensor([[5, 2], [0, 0]], dtype=torch.long)
    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (2, 2))
    self.assertEqual(result[0, 0].item(), 8)
    self.assertEqual(result[0, 1].item(), 3)
    self.assertEqual(result[1, 0].item(), 0)

  def test_minimum_size_mismatched(self):
    """2×2 (1 valid) + 3×3 (2 valid): OOR column aligned correctly."""
    # a: 1 valid class, OOR at col 1
    a = torch.tensor([[4, 2], [0, 0]], dtype=torch.long)
    # b: 2 valid classes, OOR at col 2
    b = torch.tensor([[1, 0, 3], [0, 7, 0], [0, 0, 0]], dtype=torch.long)

    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (3, 3))
    # class 0 → class 0: 4 + 1 = 5
    self.assertEqual(result[0, 0].item(), 5)
    # class 0 → OOR (col 2): 2 (from a, was col 1) + 3 (from b, col 2) = 5
    self.assertEqual(result[0, 2].item(), 5)
    # class 1 → class 1: 0 + 7 = 7 (only b has class 1)
    self.assertEqual(result[1, 1].item(), 7)
    # OOR row is zero
    self.assertTrue(torch.all(result[-1, :] == 0))

  def test_only_oor_predictions(self):
    """Matrix where all predictions are OOR (full last column) merges correctly."""
    # a (10×10): every sample predicted as OOR
    a = torch.zeros(10, 10, dtype=torch.long)
    for i in range(9):
      a[i, 9] = i + 1   # class i → OOR

    b = _make_matrix(9, seed=3)   # normal 10×10
    result = _accumulate_confusion_matrices(a, b)
    self.assertEqual(result.shape, (10, 10))
    # OOR column should sum a's OOR with b's OOR
    for i in range(9):
      self.assertEqual(result[i, 9].item(), a[i, 9].item() + b[i, 9].item())


if __name__ == '__main__':
  unittest.main()

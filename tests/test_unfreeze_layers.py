import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.backbone import VideoBackbone
from custom.helper import MODEL_TYPE


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_blocks_frozen(blocks, indices):
  """Assert that every parameter of each block at the given indices is frozen.

  Args:
    blocks:  nn.ModuleList of encoder blocks.
    indices: Iterable of integer block indices to check.
  """
  for i in indices:
    for name, param in blocks[i].named_parameters():
      assert not param.requires_grad, (
        f"block[{i}].{name} should be frozen (requires_grad=False) but is trainable."
      )


def _assert_blocks_trainable(blocks, indices):
  """Assert that every parameter of each block at the given indices is trainable.

  Args:
    blocks:  nn.ModuleList of encoder blocks.
    indices: Iterable of integer block indices to check.
  """
  for i in indices:
    for name, param in blocks[i].named_parameters():
      assert param.requires_grad, (
        f"block[{i}].{name} should be trainable (requires_grad=True) but is frozen."
      )


# ---------------------------------------------------------------------------
# VideoMAEv2 S — 12 encoder blocks
# ---------------------------------------------------------------------------

class TestVideoMAEv2SUnfreeze(unittest.TestCase):
  """Tests for selective unfreezing of VideoMAEv2-Small backbone (12 blocks)."""

  EXPECTED_N_BLOCKS = 12

  @classmethod
  def setUpClass(cls):
    """Load VideoMAEv2-S backbone once for all tests in this class."""
    cls.backbone = VideoBackbone(MODEL_TYPE.VIDEOMAE_v2_S, freeze_backbone=True)

  def test_encoder_blocks_wired(self):
    """encoder_blocks must point to model.blocks and have 12 entries."""
    self.assertIs(
      self.backbone.encoder_blocks,
      self.backbone.model.blocks,
      "encoder_blocks should be the same object as model.blocks",
    )
    self.assertEqual(
      len(self.backbone.encoder_blocks),
      self.EXPECTED_N_BLOCKS,
      f"Expected {self.EXPECTED_N_BLOCKS} blocks, got {len(self.backbone.encoder_blocks)}",
    )

  def test_freeze_all(self):
    """freeze_backbone=True must freeze all encoder blocks."""
    self.backbone._apply_freeze_logic(freeze_backbone=True, unfreeze_layers=0)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS))

  def test_unfreeze_all(self):
    """freeze_backbone=False, unfreeze_layers=0 must make all blocks trainable."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=0)
    _assert_blocks_trainable(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS))

  def test_unfreeze_last_1(self):
    """unfreeze_layers=1 must unfreeze only the last block."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=1)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 1))
    _assert_blocks_trainable(self.backbone.encoder_blocks, [self.EXPECTED_N_BLOCKS - 1])

  def test_unfreeze_last_2(self):
    """unfreeze_layers=2 must unfreeze only the last 2 blocks."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=2)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 2))
    _assert_blocks_trainable(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 2, self.EXPECTED_N_BLOCKS))

  def test_unfreeze_exceeds_raises(self):
    """unfreeze_layers > total blocks must raise ValueError."""
    with self.assertRaises(ValueError):
      self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=self.EXPECTED_N_BLOCKS + 1)


# ---------------------------------------------------------------------------
# DFER — 16 encoder blocks
# ---------------------------------------------------------------------------

class TestDFERUnfreeze(unittest.TestCase):
  """Tests for selective unfreezing of DFER backbone (16 encoder blocks)."""

  EXPECTED_N_BLOCKS = 16

  @classmethod
  def setUpClass(cls):
    """Load DFER backbone once for all tests in this class."""
    cls.backbone = VideoBackbone(MODEL_TYPE.DFER, freeze_backbone=True)

  def test_encoder_blocks_wired(self):
    """encoder_blocks must point to model.blocks and have 16 entries."""
    self.assertIs(
      self.backbone.encoder_blocks,
      self.backbone.model.blocks,
      "encoder_blocks should be the same object as model.blocks",
    )
    self.assertEqual(
      len(self.backbone.encoder_blocks),
      self.EXPECTED_N_BLOCKS,
      f"Expected {self.EXPECTED_N_BLOCKS} blocks, got {len(self.backbone.encoder_blocks)}",
    )

  def test_freeze_all(self):
    """freeze_backbone=True must freeze all encoder blocks."""
    self.backbone._apply_freeze_logic(freeze_backbone=True, unfreeze_layers=0)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS))

  def test_unfreeze_all(self):
    """freeze_backbone=False, unfreeze_layers=0 must make all blocks trainable."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=0)
    _assert_blocks_trainable(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS))

  def test_unfreeze_last_1(self):
    """unfreeze_layers=1 must unfreeze only the last block."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=1)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 1))
    _assert_blocks_trainable(self.backbone.encoder_blocks, [self.EXPECTED_N_BLOCKS - 1])

  def test_unfreeze_last_2(self):
    """unfreeze_layers=2 must unfreeze only the last 2 blocks."""
    self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=2)
    _assert_blocks_frozen(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 2))
    _assert_blocks_trainable(self.backbone.encoder_blocks, range(self.EXPECTED_N_BLOCKS - 2, self.EXPECTED_N_BLOCKS))

  def test_unfreeze_exceeds_raises(self):
    """unfreeze_layers > total blocks must raise ValueError."""
    with self.assertRaises(ValueError):
      self.backbone._apply_freeze_logic(freeze_backbone=False, unfreeze_layers=self.EXPECTED_N_BLOCKS + 1)


if __name__ == '__main__':
  unittest.main()

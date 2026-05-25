# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from vjepa2.evals.hub.preprocessor import vjepa2_preprocessor
from vjepa2.src.hub.backbones import (
    vjepa2_ac_vit_giant,
    vjepa2_vit_giant,
    vjepa2_vit_giant_384,
    vjepa2_vit_huge,
    vjepa2_vit_large,
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_large_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
)

dependencies = ["torch", "timm", "einops"]

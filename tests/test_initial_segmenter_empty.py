from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from clutt3rseg.deg_postprocess import build_instance_masks_from_leaves
from clutt3rseg.initial_segmenter import (
    _empty_initial_data,
    _has_initial_masks,
    initial_segmentation_consistency,
)


class EmptyInitialSegmentationTest(unittest.TestCase):
    def test_has_initial_masks_only_checks_requested_frames(self) -> None:
        with TemporaryDirectory() as root:
            mask_dir = Path(root) / "instance_masks"
            mask_dir.mkdir()
            (mask_dir / "mask_000003_00.png").touch()

            self.assertFalse(_has_initial_masks(mask_dir, [0, 1, 2]))
            self.assertTrue(_has_initial_masks(mask_dir, [2, 3]))

    def test_empty_initial_data_matches_normal_result_contract(self) -> None:
        data = _empty_initial_data()

        self.assertEqual(data["instance_embeddings"], {})
        self.assertIsNone(data["mean_ground_embedding"])
        self.assertEqual(data["inst_colors_u8"].shape, (0, 3))
        self.assertEqual(data["inst_colors_u8"].dtype, np.uint8)
        self.assertEqual(data["node2inst"], {})
        self.assertEqual(data["instance_pcds"], {})
        self.assertEqual(data["original_data"]["inst2all_points"], {})

    def test_initial_segmentation_returns_valid_empty_deg_masks(self) -> None:
        with TemporaryDirectory() as root:
            root = Path(root)
            mask_dir = root / "data" / "instance_masks"
            mask_dir.mkdir(parents=True)
            args = SimpleNamespace(experiment_data_dir=root, initial_idx=[0, 1, 2])

            data = initial_segmentation_consistency(args, clip=None, device="cpu")

            self.assertEqual(data["node2inst"], {})
            self.assertEqual(data["original_data"]["inst2all_points"], {})

            frame_ids, masks = build_instance_masks_from_leaves(
                data["node2inst"],
                set(),
                mask_dir,
                [(0, 10, 4, 5), (1, 11, 3, 2)],
            )
            self.assertEqual(frame_ids, [10, 11])
            self.assertEqual([mask.shape for mask in masks], [(4, 5), (3, 2)])
            self.assertTrue(all(mask.dtype == np.int32 and not mask.any() for mask in masks))


if __name__ == "__main__":
    unittest.main()

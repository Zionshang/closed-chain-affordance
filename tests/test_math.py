from __future__ import annotations

import unittest

import torch

import cca_planner.math as cm


class TorchMathTests(unittest.TestCase):
    def test_float32_small_angle_log_preserves_rotation(self):
        vector = torch.tensor([3e-5, -4e-5, 2e-5], dtype=torch.float32)
        rotation = cm.matrix_exp3(cm.vec_to_so3(vector))
        recovered = cm.so3_log_map(rotation)
        torch.testing.assert_close(recovered, vector, atol=2e-7, rtol=2e-4)

    def test_transform_inverse_round_trip(self):
        twist = torch.tensor([0.2, -0.1, 0.3, 0.4, 0.2, -0.2], dtype=torch.float64)
        transform = cm.matrix_exp6(cm.vec_to_se3(twist))
        identity = transform @ cm.trans_inv(transform)
        torch.testing.assert_close(identity, torch.eye(4, dtype=torch.float64), atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)

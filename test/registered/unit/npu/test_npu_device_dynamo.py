"""Verify device capability helpers do not introduce TorchDynamo graph breaks."""

import unittest

import torch

from sglang.srt.utils.device import (
    is_cuda,
    is_cuda_alike,
    is_hip,
    is_hpu,
    is_npu,
    is_xpu,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="extra-a-test-npu")


class TestDeviceChecksDynamoConstant(unittest.TestCase):
    def test_device_predicates_compile_fullgraph_without_breaks(self):
        """All device checks must fold as Dynamo constants without graph breaks."""
        torch._dynamo.reset()

        def fn(x: torch.Tensor) -> torch.Tensor:
            if is_cuda() or is_hip() or is_cuda_alike():
                return x + 1.0
            if is_hpu() or is_xpu():
                return x + 2.0
            if is_npu():
                return x + 3.0
            return x + 4.0

        x = torch.ones(4)
        expected = fn(x)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="eager")
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

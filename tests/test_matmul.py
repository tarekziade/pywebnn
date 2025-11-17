import unittest

import torch

from tests.wpt_utils import UnsupportedCase, load_wpt_cases, run_wpt_case

MATMUL_CASES = load_wpt_cases("matmul_tests.json")


class MatmulConformanceTests(unittest.TestCase):
    def test_all_cases(self):
        executed = 0
        for case in MATMUL_CASES:
            with self.subTest(case=case["name"]):
                try:
                    results, expected, tolerances = run_wpt_case(case)
                except UnsupportedCase:
                    continue
                executed += 1
                for name, exp in expected.items():
                    actual = results[name]
                    rtol, atol = tolerances[name]
                    torch.testing.assert_close(
                        actual,
                        exp.to(actual.device),
                        rtol=rtol,
                        atol=atol,
                    )
        self.assertNotEqual(executed, 0, "No runnable matmul cases were executed")

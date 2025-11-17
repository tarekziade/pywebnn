import unittest

import torch

from tests.wpt_utils import BACKENDS, UnsupportedCase, load_wpt_cases, run_wpt_case


class WPTConformanceBase(unittest.TestCase):
    data_file: str = ""

    def _run_cases_from_file(self):
        cases = load_wpt_cases(self.data_file)
        for backend in BACKENDS:
            executed = 0
            for case in cases:
                with self.subTest(case=case["name"], backend=backend):
                    try:
                        results, expected, tolerances = run_wpt_case(
                            case, backend=backend
                        )
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
            self.assertNotEqual(
                executed,
                0,
                f"No runnable WPT cases in '{self.data_file}' for backend {backend}",
            )


class AddConformanceTests(WPTConformanceBase):
    data_file = "add_tests.json"

    def test_add(self):
        self._run_cases_from_file()


class ClampConformanceTests(WPTConformanceBase):
    data_file = "clamp_tests.json"

    def test_clamp(self):
        self._run_cases_from_file()


class SoftmaxConformanceTests(WPTConformanceBase):
    data_file = "softmax_tests.json"

    def test_softmax(self):
        self._run_cases_from_file()


class Conv2dConformanceTests(WPTConformanceBase):
    data_file = "conv2d_tests.json"

    def test_conv2d(self):
        self._run_cases_from_file()


class ReluConformanceTests(WPTConformanceBase):
    data_file = "relu_tests.json"

    def test_relu(self):
        self._run_cases_from_file()


class MaxPool2dConformanceTests(WPTConformanceBase):
    data_file = "maxpool2d_tests.json"

    def test_maxpool2d(self):
        self._run_cases_from_file()

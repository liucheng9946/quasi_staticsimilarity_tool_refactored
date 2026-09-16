from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import app
from similarity_engine.api import SimilarityAPI
from similarity_engine.ui.component_manager import UserComponentManager


PROTO = {
    "E": 210000.0,
    "fy": 270.0,
    "bs": 0.02,
    "l": 3600.0,
    "section": {"B": 450.0, "tw": 14.0},
}
MODEL = {
    "E": 210000.0,
    "fy": 270.0,
    "bs": 0.02,
    "l": 1800.0,
    "section": {"B": 220.0, "tw": 8.0},
}

EXPECTED = {
    2: {
        "SH": 0.5,
        "SE": 1.0,
        "Ssigma": 1.0,
        "Sbs": 1.0,
        "Sy": 0.4888888888888889,
        "SA": 0.2778505897771953,
        "SI": 0.06571742030506804,
        "SK": 0.5257393624405443,
        "Sd": 0.5,
        "SF": 0.25,
        "Ap": 24416.0,
        "Am": 6784.0,
        "Ip": 774361578.6666666,
        "Im": 50889045.333333336,
    },
    3: {
        "SH": 0.5,
        "SE": 1.0,
        "Ssigma": 1.0,
        "Sbs": 1.0,
        "Sy": 0.4888888888888889,
        "SA": 0.2778505897771953,
        "SI": 0.06571742030506804,
        "SK": 0.5257393624405443,
        "Sd": 0.5,
        "SF": 0.26286968122027216,
        "Ap": 24416.0,
        "Am": 6784.0,
        "Ip": 774361578.6666666,
        "Im": 50889045.333333336,
    },
    4: {
        "SH": 0.5,
        "SE": 1.0,
        "Ssigma": 1.0,
        "Sbs": 1.0,
        "Sy": 0.4888888888888889,
        "SA": 0.2778505897771953,
        "SI": 0.06571742030506804,
        "SK": 0.5257393624405443,
        "Sd": 0.5113636363636364,
        "SF": 0.26884399215709653,
        "Ap": 24416.0,
        "Am": 6784.0,
        "Ip": 774361578.6666666,
        "Im": 50889045.333333336,
    },
}


class BuiltinRegressionTest(unittest.TestCase):
    def test_default_hollow_square_all_methods(self) -> None:
        for method_type, expected in EXPECTED.items():
            with self.subTest(method_type=method_type):
                result = app.calculate_similarity(
                    section_type=4,
                    method_type=method_type,
                    scb=0.5,
                    proto=PROTO,
                    model=MODEL,
                    user_inputs={},
                    show_detail=1,
                )
                for key, expected_value in expected.items():
                    self.assertLessEqual(
                        abs(float(result[key]) - expected_value),
                        1.0e-12,
                        msg=f"method={method_type}, key={key}",
                    )


class UserComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="quasi_components_"))
        self.api = SimilarityAPI(self.root)
        self.manager = UserComponentManager(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_install_calculate_delete_section(self) -> None:
        definition, _ = self.manager.install(
            self.manager.section_template(), "rectangular.yaml", "section"
        )
        self.api.reload_components()
        self.assertEqual(definition.id, "rectangular_section")

        proto = {
            "E": 210000.0,
            "fy": 270.0,
            "bs": 0.02,
            "l": 3600.0,
            "section": {"B": 300.0, "H": 450.0},
        }
        model = {
            "E": 210000.0,
            "fy": 270.0,
            "bs": 0.02,
            "l": 1800.0,
            "section": {"B": 150.0, "H": 225.0},
        }
        result = self.api.calculate(
            section="rectangular_section",
            method="builtin_csl_n",
            SCB=0.5,
            proto=proto,
            model=model,
            user_inputs={},
        ).to_legacy_dict()
        self.assertAlmostEqual(result["SA"], 0.25)
        self.assertAlmostEqual(result["SI"], 0.0625)
        self.assertAlmostEqual(result["Sy"], 0.5)
        self.assertAlmostEqual(result["Sd"], 0.5)
        self.assertAlmostEqual(result["SF"], 0.25)

        self.manager.delete("section", "rectangular_section")
        self.api.reload_components()
        ids = {item.definition.id for item in self.api.list_sections()}
        self.assertNotIn("rectangular_section", ids)

    def test_install_and_calculate_method(self) -> None:
        definition, _ = self.manager.install(
            self.manager.method_template(), "custom_method.yaml", "method"
        )
        self.api.reload_components()
        self.assertEqual(definition.id, "custom_quasi_static_method")
        result = self.api.calculate(
            section="builtin_hollow_square",
            method="custom_quasi_static_method",
            SCB=0.5,
            proto=PROTO,
            model=MODEL,
            user_inputs={"alpha": 1.2},
        ).to_legacy_dict()
        self.assertAlmostEqual(result["Sd"], 0.5)
        self.assertAlmostEqual(result["SF"], 1.2 * result["SK"] * result["Sd"])
        self.assertAlmostEqual(result["Ssigma"], 1.0)

    def test_reject_python_and_unsafe_expression(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.install(b"print('x')", "component.py", "section")

        unsafe = b"""id: unsafe_section
name: Unsafe
parameters:
  - name: B
    label: B
    default: 1
formulas:
  area: __import__('os')
  inertia: B
  characteristic: B
"""
        with self.assertRaises(ValueError):
            self.manager.install(unsafe, "unsafe.yaml", "section")


if __name__ == "__main__":
    unittest.main(verbosity=2)

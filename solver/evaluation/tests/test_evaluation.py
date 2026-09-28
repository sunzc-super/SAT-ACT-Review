from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from solver.evaluation.run_evaluation import solver_command
from solver.evaluation.summarize import calculate


def record(instance: str, propagations: int, conflicts: int, decisions: int):
    return {
        "instance": instance,
        "result": {
            "status": "SAT",
            "propagations": propagations,
            "conflicts": conflicts,
            "decisions": decisions,
            "wall_time": 1.0,
            "inference_time": 0.1,
            "request_time": 0.2,
            "model_calls": 3,
        },
    }


class EvaluationTest(unittest.TestCase):
    def test_public_baseline_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = solver_command(
                root / "solver",
                root / "input.cnf",
                root / "result.txt",
                "CADICAL-BASELINE",
                0,
                100,
                0,
                "127.0.0.1:41070",
            )
        self.assertEqual(command[command.index("--mode") + 1], "CADICAL-BASELINE")

    def test_model_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = solver_command(
                root / "solver",
                root / "input.cnf",
                root / "result.txt",
                "SATACT",
                2,
                100,
                5,
                "127.0.0.1:41070",
            )
        self.assertEqual(command[command.index("--mode") + 1], "SATACT")
        self.assertEqual(command[command.index("--neuro_calls") + 1], "5")

    def test_summary_metrics(self) -> None:
        baseline = [record("a", 100, 10, 20), record("b", 200, 20, 40)]
        current = [record("a", 80, 8, 16), record("b", 160, 16, 32)]
        metrics = calculate(current, baseline)
        self.assertAlmostEqual(metrics["delta_propagations"], -20.0)
        self.assertAlmostEqual(metrics["median_propagation_ratio"], 0.8)
        self.assertEqual(metrics["wins"], 2.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import unittest

from gen_data.action_rollouts.cli import (
    _select_pipeline_task,
    build_parser,
    main,
)


class CliTests(unittest.TestCase):
    def test_manifest_workers_default_and_positive_values(self) -> None:
        commands = (
            [
                "manifest-satbench",
                "--split",
                "train=/tmp/train",
                "--difficulty",
                "hard",
                "--family",
                "ps",
            ],
            [
                "manifest-satcomp",
                "--collection",
                "2024=/tmp/sc-2024",
            ],
        )
        for command in commands:
            common = [*command, "--dataset-release-id", "test"]
            self.assertEqual(1, build_parser().parse_args(common).workers)
            self.assertEqual(
                12, build_parser().parse_args([*common, "--workers", "12"]).workers
            )
            for value in ("0", "-1"):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        build_parser().parse_args([*common, "--workers", value])

    def test_prepare_exposes_only_readable_target_names(self) -> None:
        common = [
            "pipeline-prepare",
            "--preset",
            "satbench",
            "--binary",
            "/tmp/worker",
            "--build-root",
            "/tmp/build",
            "--manifest",
            "/tmp/build/manifests/instances.jsonl",
        ]
        args = build_parser().parse_args(
            [
                *common,
                "--target-initial-keep-count",
                "2",
                "--target-early-window-end",
                "8",
                "--target-early-sample-count",
                "2",
                "--target-post-restart-keep-count",
                "2",
                "--target-late-fallback-window-end",
                "20",
            ]
        )
        self.assertEqual(2, args.target_initial_keep_count)
        self.assertEqual(20, args.target_late_fallback_window_end)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args([*common, "--target-k1", "2"])


    def test_eval_timeout_parser_accepts_finite_nonnegative_values(self) -> None:
        common = [
            "pipeline-prepare", "--preset", "satbench",
            "--binary", "/tmp/worker", "--build-root", "/tmp/build",
            "--manifest", "/tmp/build/manifests/instances.jsonl",
        ]
        for text, expected in (("0", 0.0), ("0.5", 0.5), ("1.5", 1.5), ("1200", 1200.0)):
            args = build_parser().parse_args([*common, "--eval-max-timeout", text])
            self.assertEqual(expected, args.eval_max_timeout)
        for text in ("-1", "nan", "inf", "-inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args([*common, "--eval-max-timeout", text])

    def test_max_actions_mode_parser(self) -> None:
        common = [
            "pipeline-prepare", "--preset", "satbench",
            "--binary", "/tmp/worker", "--build-root", "/tmp/build",
            "--manifest", "/tmp/build/manifests/instances.jsonl",
        ]
        self.assertEqual(
            "default", build_parser().parse_args(common).max_actions_mode
        )
        self.assertEqual(
            "variable",
            build_parser().parse_args(
                [*common, "--max-actions-mode", "variable"]
            ).max_actions_mode,
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([*common, "--max-actions-mode", "invalid"])


class PipelineTaskSelectionTests(unittest.TestCase):
    @staticmethod
    def _parts(count: int) -> list[dict[str, object]]:
        return [
            {"part_stem": f"part-{ordinal:05d}", "ordinal": ordinal}
            for ordinal in range(count)
        ]

    def test_default_cli_selection_covers_the_full_plan(self) -> None:
        args = build_parser().parse_args(
            [
                "pipeline-run-all",
                "--policy",
                "/tmp/policy.json",
                "--plan",
                "/tmp/plan.jsonl",
            ]
        )
        self.assertEqual(1, args.task_count)
        self.assertEqual(0, args.task_index)
        selected = list(
            _select_pipeline_task(
                self._parts(10),
                task_count=args.task_count,
                task_index=args.task_index,
            )
        )
        self.assertEqual(list(range(10)), [part["ordinal"] for part in selected])

    def test_round_robin_tasks_are_disjoint_and_complete(self) -> None:
        parts = self._parts(10)
        selections = [
            list(
                _select_pipeline_task(
                    parts,
                    task_count=3,
                    task_index=task_index,
                )
            )
            for task_index in range(3)
        ]
        ordinals = [
            [part["ordinal"] for part in selection]
            for selection in selections
        ]
        self.assertEqual([[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]], ordinals)
        flattened = [ordinal for task in ordinals for ordinal in task]
        self.assertEqual(list(range(10)), sorted(flattened))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_task_count_larger_than_plan_allows_empty_tasks(self) -> None:
        selected = list(
            _select_pipeline_task(
                self._parts(2),
                task_count=4,
                task_index=3,
            )
        )
        self.assertEqual([], selected)

    def test_task_index_must_be_in_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_index"):
            list(
                _select_pipeline_task(
                    self._parts(3),
                    task_count=3,
                    task_index=3,
                )
            )

    def test_cli_rejects_nonpositive_count_and_negative_index(self) -> None:
        common = [
            "pipeline-run-all",
            "--policy",
            "/tmp/policy.json",
            "--plan",
            "/tmp/plan.jsonl",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args([*common, "--task-count", "0"])
            with self.assertRaises(SystemExit):
                build_parser().parse_args([*common, "--task-index", "-1"])

    def test_cli_rejects_task_index_equal_to_count_before_loading_policy(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(
                    [
                        "pipeline-run-all",
                        "--policy",
                        "/tmp/not-used-policy.json",
                        "--plan",
                        "/tmp/not-used-plan.jsonl",
                        "--task-count",
                        "3",
                        "--task-index",
                        "3",
                    ]
                )


if __name__ == "__main__":
    unittest.main()

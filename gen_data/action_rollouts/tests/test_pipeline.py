from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from gen_data.action_rollouts.cli import _completed_part_matches
from gen_data.action_rollouts.pipeline import (
    JsonlDemultiplexer,
    PipelinePartError,
    PipelinePolicy,
    build_worker_argv,
    execute_pipeline_part,
    forced_evaluation_literals,
    load_pipeline_policy,
    pipeline_policy_from_record,
    read_pipeline_plan,
    select_target_traces,
    write_pipeline_policy,
    write_pipeline_plan,
    write_worker_raw_schema,
)
from gen_data.action_rollouts.schema import MANIFEST_SCHEMA


def trace(instance: str, eligible: int, *, restarts: int = 0) -> list:
    return [
        "tr",
        instance,
        eligible,
        eligible,
        eligible,
        0,
        0,
        0,
        eligible - 1,
        0,
        restarts,
        2,
        5,
        f"{eligible:016x}",
        f"{eligible + 100:016x}",
        1,
        1,
    ]


class CountingSink:
    def __init__(self) -> None:
        self.bytes = 0
        self.records = 0

    def write(self, payload) -> int:
        self.bytes += len(payload)
        return len(payload)

    def finish_record(self) -> None:
        self.records += 1


class PipelineTests(unittest.TestCase):
    def make_policy(self, root: Path, binary: Path, manifest: Path, **changes) -> PipelinePolicy:
        values = dict(
            policy_id="0" * 32,
            output_tag="test",
            binary=binary,
            build_root=root / "build",
            manifest=manifest,
            target_initial_keep_count=1,
            target_early_window_end=2,
            target_early_sample_count=1,
            target_post_restart_keep_count=1,
            target_late_fallback_window_end=3,
            timeout_seconds=10,
            raw_codec="gzip",
            solver_settings=(("walk", 0),),
        )
        values.update(changes)
        return PipelinePolicy(**values)

    def test_eval_timeout_policy_validation_and_old_policy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            binary = root / "worker"
            binary.touch()
            binary.chmod(0o755)
            manifest = root / "manifest"
            manifest.touch()
            policy = self.make_policy(
                root,
                binary,
                manifest,
                eval_max_timeout=0.5,
                plain=True,
                max_actions_mode="variable",
            )
            config_path = write_pipeline_policy(root / "policy.json", policy)
            record = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(0.5, record["eval_max_timeout"])
            self.assertTrue(record["plain"])
            self.assertEqual("variable", record["max_actions_mode"])
            self.assertEqual(policy, load_pipeline_policy(config_path))

            del record["eval_max_timeout"]
            del record["plain"]
            del record["max_actions_mode"]
            old_policy = pipeline_policy_from_record(record)
            self.assertEqual(0.0, old_policy.eval_max_timeout)
            self.assertFalse(old_policy.plain)
            self.assertEqual("default", old_policy.max_actions_mode)

            for invalid in (-1, float("nan"), float("inf"), -float("inf"), True):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "eval_max_timeout"):
                        self.make_policy(
                            root, binary, manifest, eval_max_timeout=invalid
                        )

    def test_selector_uses_restart_and_no_restart_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            binary = root / "worker"
            binary.touch()
            binary.chmod(0o755)
            manifest = root / "manifest"
            manifest.touch()
            policy = self.make_policy(root, binary, manifest)

            no_restart = select_target_traces([trace("i", n) for n in (1, 2, 3)], policy)
            self.assertEqual([1, 2, 3], [item[4] for item in no_restart])

            with_restart = select_target_traces(
                [trace("i", 1), trace("i", 2), trace("i", 3, restarts=1)], policy
            )
            self.assertEqual([1, 2, 3], [item[4] for item in with_restart])

            # A post-restart callback already retained by the early selector
            # does not consume the distinct post-restart quota.
            overlap_policy = self.make_policy(
                root,
                binary,
                manifest,
                target_initial_keep_count=1,
                target_early_window_end=2,
                target_early_sample_count=1,
                target_post_restart_keep_count=2,
                target_late_fallback_window_end=5,
            )
            overlap = select_target_traces(
                [
                    trace("i", 1),
                    trace("i", 2, restarts=1),
                    trace("i", 3, restarts=1),
                    trace("i", 4, restarts=1),
                ],
                overlap_policy,
            )
            self.assertEqual([1, 2, 3, 4], [item[4] for item in overlap])

            config_path = write_pipeline_policy(root / "policy.json", policy)
            self.assertEqual(policy, load_pipeline_policy(config_path))
            policy_record = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(1, policy_record["target_initial_keep_count"])
            self.assertNotIn("k1", policy_record)
            self.assertNotIn("late_max", policy_record)
            raw_schema_path = write_worker_raw_schema(root / "worker.raw.schema.json")
            raw_schema = json.loads(raw_schema_path.read_text(encoding="utf-8"))
            self.assertIn("sl", raw_schema["stage_tags"]["states"])

    def test_part_error_is_process_pool_pickle_safe(self) -> None:
        original = PipelinePartError(
            "synthetic failure",
            log_path=Path("attempt.jsonl"),
            statistics_path=Path("failed.json"),
        )
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual("synthetic failure", str(restored))
        self.assertEqual(Path("attempt.jsonl"), restored.log_path)
        self.assertEqual(Path("failed.json"), restored.statistics_path)

    def test_demultiplexer_never_parses_large_state(self) -> None:
        state_sink = CountingSink()
        small_sink = CountingSink()
        events = []

        def guarded_loader(payload: bytes):
            self.assertFalse(payload.startswith(b'["st",'))
            return json.loads(payload)

        demux = JsonlDemultiplexer(
            {"st": state_sink, "out": small_sink},
            parse_tags={"out"},
            on_record=lambda tag, value: events.append((tag, value)),
            json_loader=guarded_loader,
        )
        large_state = b'["st","instance",1,"' + (b"x" * (5 * 1024 * 1024)) + b'"]\n'
        output = b'["out","instance","capture",0,0,0,0,0,1,1,1,0]\n'
        payload = large_state + output
        for start in range(0, len(payload), 997):
            demux.feed(payload[start : start + 997])
        demux.finish()
        self.assertEqual(1, state_sink.records)
        self.assertGreater(state_sink.bytes, 5 * 1024 * 1024)
        self.assertEqual(1, small_sink.records)
        self.assertLess(demux.peak_small_buffer_bytes, 1024)
        self.assertEqual("st", events[0][0])
        self.assertIsNone(events[0][1])

    def test_eval_argv_contains_complete_locator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            binary = root / "worker"
            binary.touch()
            binary.chmod(0o755)
            manifest = root / "manifest"
            manifest.touch()
            policy = self.make_policy(
                root, binary, manifest, eval_max_timeout=1.5
            )
            locator = [
                "sl",
                "i",
                3,
                7,
                8,
                2,
                9,
                10,
                11,
                12,
                13,
                "0000000000000014",
                -4,
                "15",
                "16",
                "17",
                "18",
                "19",
                20,
                21,
                22,
                23,
            ]
            argv = build_worker_argv(
                policy,
                mode="eval",
                instance_id="i",
                source_path=root / "input.cnf",
                locator=locator,
                force_literal=-4,
            )
            self.assertNotIn("--plain", argv)
            required = {
                "--target=3",
                "--force-literal=-4",
                "--expected-callback=7",
                "--expected-raw-decision=8",
                "--expected-prefix-hash=0000000000000014",
                "--expected-assignment-hash=15",
                "--expected-trail-hash=16",
                "--expected-clause-canonical-hash=17",
                "--expected-clause-order-hash=18",
                "--expected-eligibility-hash=19",
                "--expected-conflicts=10",
                "--expected-decisions=11",
                "--expected-propagations=12",
                "--expected-restarts=13",
                "--expected-trail=9",
                "--expected-native-literal=-4",
                "--eval-max-timeout=1.5",
            }
            self.assertTrue(required.issubset(set(argv)))

            zero_native = list(locator)
            zero_native[12] = 0
            proposal = ["pr", "i", 3, [[-4, 1, 0.0, 0.0, "1"]]]
            self.assertEqual([-4], forced_evaluation_literals(zero_native, proposal))
            proposal[3][0][0] = -(1 << 31)
            with self.assertRaises(ValueError):
                forced_evaluation_literals(zero_native, proposal)

    def test_discover_argv_uses_readable_target_option_names(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            binary = root / "worker"
            binary.touch()
            binary.chmod(0o755)
            manifest = root / "manifest"
            manifest.touch()
            policy = self.make_policy(
                root,
                binary,
                manifest,
                eval_max_timeout=0.5,
                plain=True,
                max_actions_mode="variable",
            )
            argv = build_worker_argv(
                policy,
                mode="discover",
                instance_id="i",
                source_path=root / "input.cnf",
            )
            self.assertIn("--discover-eligible-window-end=3", argv)
            self.assertIn("--discover-post-restart-keep-count=1", argv)
            self.assertIn("--plain", argv)
            self.assertFalse(
                any(item.startswith("--eval-max-timeout=") for item in argv)
            )
            capture_argv = build_worker_argv(
                policy,
                mode="capture",
                instance_id="i",
                source_path=root / "input.cnf",
                targets=(1,),
            )
            self.assertIn("--plain", capture_argv)
            self.assertIn("--max-actions-mode=variable", capture_argv)
            self.assertFalse(
                any(item.startswith("--eval-max-timeout=") for item in capture_argv)
            )
            self.assertFalse(any("early-limit" in item for item in argv))
            self.assertFalse(
                any(
                    item.startswith("--discover-post-restart-keep=")
                    for item in argv
                )
            )

    def test_fake_worker_full_raw_part(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            source = root / "tiny.cnf"
            source.write_text("p cnf 2 1\n1 -2 0\n", encoding="ascii")
            worker = root / "fake_worker.py"
            worker.write_text(
                "#!" + sys.executable + "\n" + FAKE_WORKER,
                encoding="utf-8",
            )
            worker.chmod(0o755)
            manifest = root / "instances.jsonl"
            manifest_record = {
                "schema": MANIFEST_SCHEMA,
                "instance_id": "fake-instance",
                "dataset_group": "satbench/medium/sr/test",
                "part_id": 0,
                "group_index": 0,
                "source_path": str(source.resolve()),
                "source_relpath": "sat/tiny.cnf",
                "accepted": True,
                "size": {
                    "variables": 2,
                    "clauses": 1,
                    "literal_occurrences": 2,
                    "nodes_2v_plus_c": 5,
                },
            }
            manifest.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")
            policy = self.make_policy(root, worker, manifest)

            plan_result = write_pipeline_plan(policy)
            self.assertEqual(1, plan_result.part_count)
            parts = list(read_pipeline_plan(plan_result.path))
            self.assertEqual(1, len(parts))
            result = execute_pipeline_part(policy, parts[0])
            self.assertEqual(1, result.instances_completed)
            self.assertEqual(3, result.states_materialized)
            self.assertEqual(9, result.evaluations)
            statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))
            self.assertEqual("complete", statistics["status"])
            self.assertEqual(9, statistics["counts"]["valid_evaluations"])
            self.assertTrue(_completed_part_matches(policy, parts[0]))

            original_statistics = result.statistics_path.read_bytes()
            statistics["schema"] = "wrong-schema"
            result.statistics_path.write_text(
                json.dumps(statistics) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "policy/plan"):
                _completed_part_matches(policy, parts[0])
            result.statistics_path.write_bytes(original_statistics)

            states_path = Path(result.output_files["states"]["path"])
            original_states = states_path.read_bytes()
            states_path.write_bytes(original_states + b"x")
            with self.assertRaisesRegex(ValueError, "wrong size"):
                _completed_part_matches(policy, parts[0])
            states_path.write_bytes(original_states)

    def test_duplicate_plan_stem_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            path = Path(temporary) / "pipeline.parts.jsonl"
            record = {
                "schema": "decisiontrace-actioneval-raw-part-plan-v2",
                "policy_id": "0" * 32,
                "dataset_group": "test",
                "part_id": 0,
                "part_stem": "test.part-00000",
                "instances": [{"instance_id": "i"}],
            }
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate.*part_stem"):
                list(read_pipeline_plan(path))

    def test_failed_worker_publishes_only_attempt_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-pipeline-") as temporary:
            root = Path(temporary)
            source = root / "tiny.cnf"
            source.write_text("p cnf 1 0\n", encoding="ascii")
            worker = root / "fail_worker.py"
            worker.write_text(
                "#!" + sys.executable + "\nimport sys\nsys.stderr.write('intentional\\n')\nsys.exit(7)\n",
                encoding="utf-8",
            )
            worker.chmod(0o755)
            manifest = root / "instances.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": MANIFEST_SCHEMA,
                        "instance_id": "failed-instance",
                        "dataset_group": "satcomp/2022/main",
                        "part_id": 0,
                        "group_index": 0,
                        "source_path": str(source.resolve()),
                        "source_relpath": "cnf/tiny.cnf",
                        "accepted": True,
                        "size": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            policy = self.make_policy(root, worker, manifest)
            part = next(read_pipeline_plan(write_pipeline_plan(policy).path))
            with self.assertRaises(PipelinePartError) as caught:
                execute_pipeline_part(policy, part)
            self.assertTrue(caught.exception.log_path.is_file())
            self.assertTrue(caught.exception.statistics_path.is_file())
            failed = json.loads(caught.exception.statistics_path.read_text(encoding="utf-8"))
            self.assertEqual("failed", failed["status"])
            self.assertEqual(policy.policy_id, failed["policy_id"])
            suffix = ".jsonl.gz"
            for stage in ("traces", "states", "proposals", "evals"):
                self.assertFalse(
                    (policy.build_root / stage / f"{part['part_stem']}{suffix}").exists()
                )


FAKE_WORKER = r'''
import json
import sys
import time

options = {}
for arg in sys.argv[1:-1]:
    if arg == "--no-schema":
        continue
    if arg.startswith("--") and "=" in arg:
        key, value = arg[2:].split("=", 1)
        options.setdefault(key, []).append(value)

mode = options["mode"][-1]
instance = options["instance-id"][-1]

def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)

def hashes(target):
    return (
        f"{target:016x}",
        f"{target + 10:016x}",
        f"{target + 20:016x}",
        f"{target + 30:016x}",
        f"{target + 40:016x}",
        f"{target + 100:016x}",
    )

if mode == "discover":
    for target in (1, 2, 3):
        prefix, _, _, _, _, eligibility = hashes(target)
        emit(["tr", instance, target, target, target, 0, 0, 0, target - 1, 0, 0,
              2, 5, prefix, eligibility, 1, 1])
    emit(["out", instance, "discover", 0, 0, 3, 0, 0, 3, 3, 0, 0])
elif mode == "capture":
    targets = [int(item) for item in options["targets"][-1].split(",")]
    for target in targets:
        prefix, assignment, trail, canonical, order, eligibility = hashes(target)
        emit(["sl", instance, target, target, target, 0, 0, 0, target - 1, 0, 0,
              prefix, 1, assignment, trail, canonical, order, eligibility, 5, 2, 2, 64])
        emit(["st", instance, target, target, target, 0, 0, 0, target - 1, 0, 0,
              prefix, 1, 2, 2, 1, 2, assignment, trail, canonical, order, eligibility,
              [1, 2], "03", [0, 0], [-1, -1], [0, 0], [-1, -1],
              [0.5, 0.25], [1, -1], [0, 2], [1, -2]])
        emit(["pr", instance, target,
              [[1, 1, 0.5, 0.25, "0000000000000001"],
               [-2, 16, 0.25, 0.5, "0000000000000002"]]])
    emit(["out", instance, "capture", 0, 0, 3, 0, 0, 3, 3, len(targets), 0])
elif mode == "eval":
    target = int(options["target"][-1])
    requested = int(options["force-literal"][-1])
    prefix = options["expected-prefix-hash"][-1]
    native = int(options["expected-native-literal"][-1])
    applied = native if requested == 0 else requested
    callback = int(options["expected-callback"][-1])
    raw = int(options["expected-raw-decision"][-1])
    if requested == -2 and "timeouttest=1" in options.get("set", []):
        time.sleep(2)
    emit(["ev", instance, target, callback, raw, requested, True, True, True, True,
          prefix, native, applied, 10, False, False, 1, 1, 2, 0])
    emit(["out", instance, "eval", 10, 1, 1, 2, 0, target, target, 0, 0])
else:
    raise SystemExit(2)
'''


if __name__ == "__main__":
    unittest.main()

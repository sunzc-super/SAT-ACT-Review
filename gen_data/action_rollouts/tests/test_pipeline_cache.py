from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gen_data.action_rollouts.cli import build_parser
from gen_data.action_rollouts.pipeline import (
    PipelinePolicy,
    PipelinePartError,
    PIPELINE_PLAN_SCHEMA,
    WorkerError,
    _WorkerResult,
    _remove_staged_cnf,
    _stage_cnf_in_cache,
    canonical_json_bytes,
    execute_pipeline_part,
    load_pipeline_policy,
    pipeline_policy_from_record,
    write_pipeline_policy,
)


class CnfCacheTests(unittest.TestCase):
    def make_policy(self, root: Path, **changes: object) -> PipelinePolicy:
        binary = root / "worker"
        binary.touch()
        binary.chmod(0o755)
        manifest = root / "instances.jsonl"
        manifest.touch()
        values: dict[str, object] = {
            "policy_id": "0" * 32,
            "output_tag": "test",
            "binary": binary,
            "build_root": root / "build",
            "manifest": manifest,
        }
        values.update(changes)
        return PipelinePolicy(**values)

    def test_cli_accepts_cnf_cache_root(self) -> None:
        args = build_parser().parse_args(
            [
                "pipeline-prepare",
                "--preset",
                "satbench",
                "--binary",
                "/tmp/worker",
                "--build-root",
                "/tmp/build",
                "--manifest",
                "/tmp/instances.jsonl",
                "--cnf-cache-root",
                "/dev/shm/decisiontrace-test",
            ]
        )
        self.assertEqual("/dev/shm/decisiontrace-test", args.cnf_cache_root)

    def test_policy_round_trip_and_legacy_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-policy-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root, cnf_cache_root=root / "cache")
            path = write_pipeline_policy(root / "policy.json", policy)
            self.assertEqual(policy, load_pipeline_policy(path))

            record = json.loads(path.read_text(encoding="utf-8"))
            del record["cnf_cache_root"]
            legacy = pipeline_policy_from_record(record)
            self.assertIsNone(legacy.cnf_cache_root)

    def test_stage_preserves_supported_suffix_and_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-cache-") as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            for suffix in (".cnf", ".cnf.gz", ".cnf.xz", ".cnf.bz2"):
                source = root / f"input{suffix}"
                payload = f"payload-{suffix}".encode("ascii")
                source.write_bytes(payload)
                staged = _stage_cnf_in_cache(source, cache_root)
                assert staged is not None
                self.assertTrue(staged.name.endswith(suffix))
                self.assertEqual(payload, staged.read_bytes())
                _remove_staged_cnf(staged)
                self.assertFalse(staged.exists())

    def test_disabled_cache_does_not_create_a_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-cache-") as temporary:
            source = Path(temporary) / "input.cnf"
            source.write_text("p cnf 0 0\n", encoding="ascii")
            self.assertIsNone(_stage_cnf_in_cache(source, None))

    def test_pipeline_uses_and_removes_staged_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-cache-") as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            source = root / "input.cnf"
            source.write_text("p cnf 0 0\n", encoding="ascii")
            policy = self.make_policy(root, cnf_cache_root=cache_root)
            worker_sources: list[Path] = []

            def run_worker(argv, demultiplexer, *, timeout_seconds):
                del timeout_seconds
                worker_source = Path(argv[-1])
                worker_sources.append(worker_source)
                self.assertNotEqual(source, worker_source)
                self.assertEqual(source.read_bytes(), worker_source.read_bytes())
                demultiplexer.feed(
                    canonical_json_bytes(
                        ["out", "instance", "discover", "unknown", 0, 0, 0, 0, 0, 0, 0, 0]
                    )
                    + b"\n"
                )
                demultiplexer.finish()
                return _WorkerResult(0.0, "", {"out": 1})

            part = {
                "schema": PIPELINE_PLAN_SCHEMA,
                "policy_id": policy.policy_id,
                "part_stem": "cache-test.part-00000",
                "instances": [
                    {"instance_id": "instance", "source_path": str(source)}
                ],
            }
            with mock.patch(
                "gen_data.action_rollouts.pipeline._run_worker",
                side_effect=run_worker,
            ):
                execute_pipeline_part(policy, part)

            self.assertEqual(1, len(worker_sources))
            self.assertFalse(worker_sources[0].exists())
            self.assertEqual([], list(cache_root.iterdir()))

    def test_pipeline_removes_staged_source_after_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-cache-") as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            source = root / "input.cnf"
            source.write_text("p cnf 0 0\n", encoding="ascii")
            policy = self.make_policy(root, cnf_cache_root=cache_root)
            staged_paths: list[Path] = []

            def fail_worker(argv, demultiplexer, *, timeout_seconds):
                del demultiplexer, timeout_seconds
                staged_paths.append(Path(argv[-1]))
                raise WorkerError("synthetic failure")

            part = {
                "schema": PIPELINE_PLAN_SCHEMA,
                "policy_id": policy.policy_id,
                "part_stem": "cache-failure.part-00000",
                "instances": [
                    {"instance_id": "instance", "source_path": str(source)}
                ],
            }
            with mock.patch(
                "gen_data.action_rollouts.pipeline._run_worker",
                side_effect=fail_worker,
            ):
                with self.assertRaises(PipelinePartError):
                    execute_pipeline_part(policy, part)

            self.assertEqual(1, len(staged_paths))
            self.assertFalse(staged_paths[0].exists())
            self.assertEqual([], list(cache_root.iterdir()))


if __name__ == "__main__":
    unittest.main()

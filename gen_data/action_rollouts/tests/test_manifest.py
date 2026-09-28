from __future__ import annotations

import gzip
import json
import lzma
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gen_data.action_rollouts.manifest import (
    ManifestLimits,
    build_manifest,
    enumerate_satbench,
    enumerate_satcomp,
    read_manifest,
)
from gen_data.action_rollouts.satcomp_split import (
    SATCOMPSplitAssignment,
    satcomp_source_expectations,
    select_satcomp_sources,
)


SMALL_CNF = "c small\np cnf 2 1\n1 -2 0\n"


class ManifestTests(unittest.TestCase):
    def test_satbench_is_nonrecursive_read_only_and_applies_separate_limits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            split_root = Path(temporary) / "medium" / "sr" / "train"
            sat = split_root / "sat"
            unsat = split_root / "unsat"
            sat.mkdir(parents=True)
            unsat.mkdir()
            (sat / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            with gzip.open(sat / "same-content.cnf.gz", "wb") as stream:
                stream.write(SMALL_CNF.encode("ascii"))
            (sat / "nested").mkdir()
            (sat / "nested" / "ignored.cnf").write_text(SMALL_CNF, encoding="ascii")
            (unsat / "too-many-nodes.cnf").write_text(
                "p cnf 6 1\n1 0\n", encoding="ascii"
            )
            (unsat / "notes.txt").write_text("not CNF", encoding="utf-8")
            leaf_files_before = {
                path.relative_to(split_root): path.read_bytes()
                for path in split_root.rglob("*")
                if path.is_file()
            }

            sources = enumerate_satbench(
                {"train": split_root}, difficulty="medium", family="sr"
            )
            self.assertEqual(3, len(sources))
            output = split_root / "ae-test" / "DecisionTrace-ActionEval-v1" / "manifest"
            result = build_manifest(
                sources,
                output_root=output,
                output_tag="ae-test",
                dataset_release_id="g4satbench-80K",
                limits=ManifestLimits(max_input_nodes=10, max_literal_occurrences=10),
                part_size=2,
            )

            self.assertEqual((3, 2, 1), (
                result.discovered_count,
                result.accepted_count,
                result.rejected_count,
            ))
            records = list(read_manifest(result.manifest_path))
            self.assertTrue(records[0]["instance_id"].startswith("sb.g4satbench-80k.medium.sr"))
            self.assertEqual("medium/sr/train/sat/a.cnf", records[0]["identity_relpath"])
            self.assertEqual(5, records[0]["size"]["nodes_2v_plus_c"])
            rejected = [record for record in records if not record["accepted"]]
            self.assertEqual(["input_nodes_exceeded"], rejected[0]["skip_reasons"])
            accepted = [record for record in records if record["accepted"]]
            self.assertEqual(
                accepted[0]["source_instance_group_id"],
                accepted[1]["source_instance_group_id"],
            )
            leaf_files_after = {
                path.relative_to(split_root): path.read_bytes()
                for leaf in (sat, unsat)
                for path in leaf.rglob("*")
                if path.is_file()
            }
            expected_leaf_files = {
                key: value
                for key, value in leaf_files_before.items()
                if key.parts[0] in {"sat", "unsat"}
            }
            self.assertEqual(expected_leaf_files, leaf_files_after)
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(10, metadata["limits"]["max_input_nodes"])

    def test_satcomp_scans_only_explicit_collection_cnf_leaf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            collection = Path(temporary) / "sc-2024"
            cnf_root = collection / "cnf"
            cnf_root.mkdir(parents=True)
            (cnf_root / "direct.cnf").write_text(SMALL_CNF, encoding="ascii")
            (cnf_root / "nested").mkdir()
            (cnf_root / "nested" / "ignored.cnf").write_text(SMALL_CNF, encoding="ascii")

            sources = enumerate_satcomp({"2024": collection}, track="main")
            self.assertEqual(1, len(sources))
            output = collection / "ae-test" / "DecisionTrace-ActionEval-v1" / "manifest"
            result = build_manifest(
                sources,
                output_root=output,
                output_tag="ae-test",
                dataset_release_id="satcomp",
            )
            record = next(read_manifest(result.manifest_path))
            self.assertEqual("2024", record["competition_year"])
            self.assertEqual("main", record["track"])
            self.assertEqual("direct.cnf", record["identity_relpath"])
            self.assertEqual("cnf/direct.cnf", record["source_relpath"])
            self.assertTrue(record["instance_id"].startswith("sc.satcomp.2024.main.direct."))

    def test_output_may_be_sibling_but_not_inside_cnf_leaf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            split_root = Path(temporary) / "train"
            for label in ("sat", "unsat"):
                (split_root / label).mkdir(parents=True)
            (split_root / "sat" / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            sources = enumerate_satbench(
                {"train": split_root}, difficulty="medium", family="sr"
            )
            with self.assertRaisesRegex(ValueError, "non-nested"):
                build_manifest(
                    sources,
                    output_root=split_root / "sat" / "pollution",
                    output_tag="ae-test",
                    dataset_release_id="release",
                )

    def test_truncated_xz_becomes_invalid_record(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            collection = Path(temporary) / "sc-2022"
            cnf_root = collection / "cnf"
            cnf_root.mkdir(parents=True)
            compressed = lzma.compress(SMALL_CNF.encode("ascii"), format=lzma.FORMAT_XZ)
            (cnf_root / "broken.cnf.xz").write_bytes(compressed[:-8])
            sources = enumerate_satcomp({"2022": collection})
            result = build_manifest(
                sources,
                output_root=collection / "DecisionTrace-ActionEval-v1-test" / "manifests",
                output_tag="test",
                dataset_release_id="satcomp",
            )
            record = next(read_manifest(result.manifest_path))
            self.assertFalse(record["accepted"])
            self.assertEqual(["invalid_dimacs"], record["skip_reasons"])

    def test_same_named_content_in_two_competition_years_has_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            root = Path(temporary)
            collections = {}
            for year in ("2022", "2023"):
                collection = root / f"sc-{year}"
                (collection / "cnf").mkdir(parents=True)
                (collection / "cnf" / "same.cnf").write_text(SMALL_CNF, encoding="ascii")
                collections[year] = collection
            result = build_manifest(
                enumerate_satcomp(collections),
                output_root=root / "DecisionTrace-ActionEval-v1-test" / "manifests",
                output_tag="test",
                dataset_release_id="satcomp",
            )
            records = list(read_manifest(result.manifest_path, accepted_only=True))
            self.assertEqual(2, len(records))
            self.assertEqual(2, len({record["instance_id"] for record in records}))

    def test_parallel_manifest_matches_serial_order_and_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            root = Path(temporary)
            split_root = root / "train"
            sat = split_root / "sat"
            unsat = split_root / "unsat"
            sat.mkdir(parents=True)
            unsat.mkdir()
            (sat / "b.cnf").write_text("p cnf 2 1\n1 0\n", encoding="ascii")
            (sat / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            broken = lzma.compress(SMALL_CNF.encode("ascii"), format=lzma.FORMAT_XZ)
            (unsat / "broken.cnf.xz").write_bytes(broken[:-8])
            (unsat / "filtered.cnf").write_text("p cnf 20 1\n1 0\n", encoding="ascii")
            sources = enumerate_satbench(
                {"train": split_root}, difficulty="hard", family="ps"
            )
            common = {
                "output_tag": "parallel-test",
                "dataset_release_id": "test-release",
                "limits": ManifestLimits(
                    max_input_nodes=10, max_literal_occurrences=10
                ),
                "part_size": 1,
            }
            serial = build_manifest(
                sources, output_root=root / "serial", workers=1, **common
            )
            parallel = build_manifest(
                sources, output_root=root / "parallel", workers=2, **common
            )
            self.assertEqual(
                serial.manifest_path.read_bytes(), parallel.manifest_path.read_bytes()
            )

    def test_expected_content_uses_the_single_manifest_inspection_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            root = Path(temporary)
            collection = root / "sc-2024"
            cnf_root = collection / "cnf"
            cnf_root.mkdir(parents=True)
            (cnf_root / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            sources = enumerate_satcomp({"2024": collection})
            baseline = build_manifest(
                sources,
                output_root=root / "baseline",
                output_tag="test",
                dataset_release_id="satcomp",
            )
            record = next(read_manifest(baseline.manifest_path))
            expected = {sources[0].path: record["cnf_sha256"]}
            from gen_data.action_rollouts import manifest

            with mock.patch.object(
                manifest, "inspect_dimacs", wraps=manifest.inspect_dimacs
            ) as inspect:
                build_manifest(
                    sources,
                    output_root=root / "checked",
                    output_tag="test",
                    dataset_release_id="satcomp",
                    expected_content=expected,
                )
            self.assertEqual(1, inspect.call_count)

    def test_expected_content_rejects_changed_cnf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            root = Path(temporary)
            collection = root / "sc-2024"
            cnf_root = collection / "cnf"
            cnf_root.mkdir(parents=True)
            (cnf_root / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            sources = enumerate_satcomp({"2024": collection})
            with self.assertRaisesRegex(ValueError, "content changed since split"):
                build_manifest(
                    sources,
                    output_root=root / "checked",
                    output_tag="test",
                    dataset_release_id="satcomp",
                    expected_content={sources[0].path: "0" * 64},
                    workers=2,
                )

    def test_satcomp_selection_defers_content_check_to_manifest_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            root = Path(temporary)
            collection = root / "sc-2024"
            cnf_root = collection / "cnf"
            cnf_root.mkdir(parents=True)
            (cnf_root / "a.cnf").write_text(SMALL_CNF, encoding="ascii")
            sources = enumerate_satcomp({"2024": collection})
            baseline = build_manifest(
                sources,
                output_root=root / "baseline",
                output_tag="test",
                dataset_release_id="satcomp",
            )
            record = next(read_manifest(baseline.manifest_path))
            key = ("2024", "main", "cnf/a.cnf")
            assignment = SATCOMPSplitAssignment(
                path=root / "assignment.jsonl",
                sha256="test",
                header={},
                sources={
                    key: {
                        "split": "train",
                        "selected_for_generation": True,
                        "valid_dimacs": True,
                        "source_instance_group_id": record["cnf_sha256"],
                    }
                },
            )
            selected = select_satcomp_sources(
                sources, assignment, "train", validate_content=False
            )
            expected = satcomp_source_expectations(selected, assignment)
            from gen_data.action_rollouts import manifest

            with mock.patch.object(
                manifest, "inspect_dimacs", wraps=manifest.inspect_dimacs
            ) as inspect:
                build_manifest(
                    selected,
                    output_root=root / "selected",
                    output_tag="test",
                    dataset_release_id="satcomp",
                    expected_content=expected,
                )
            self.assertEqual(1, inspect.call_count)


if __name__ == "__main__":
    unittest.main()

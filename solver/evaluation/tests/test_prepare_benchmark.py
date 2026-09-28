from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from solver.evaluation.common import load_manifest
from solver.evaluation.prepare_benchmark import build_manifest, write_manifest


class PrepareBenchmarkTest(unittest.TestCase):
    def test_builds_balanced_relative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for family in ("ca", "sr", "ps"):
                family_root = root / "sources" / family
                sources[family] = family_root
                for split in ("valid", "test"):
                    for label in ("sat", "unsat"):
                        directory = family_root / split / label
                        directory.mkdir(parents=True)
                        for index in range(2):
                            (directory / f"{index}.cnf").write_text(
                                "p cnf 1 1\n1 0\n", encoding="utf-8"
                            )
            output = root / "prepared"
            output.mkdir()
            rows = build_manifest(sources, output, seed=1, groups=1, instances_per_label=1)
            write_manifest(output, rows)
            loaded = load_manifest(output)
            self.assertEqual(len(loaded), 12)
            self.assertTrue(all(not Path(row["relative_path"]).is_absolute() for row in loaded))
            self.assertTrue(all((output / row["relative_path"]).is_symlink() for row in loaded))


if __name__ == "__main__":
    unittest.main()

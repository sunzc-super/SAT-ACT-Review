from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import gen_data.action_rollouts.dimacs as dimacs


class DimacsStreamingTests(unittest.TestCase):
    def test_tokens_and_clause_can_cross_read_boundaries(self) -> None:
        content = b"c test\np cnf 12 1\n12 -12 1 -2 0\n"
        with tempfile.TemporaryDirectory(prefix="decisiontrace-test-") as temporary:
            path = Path(temporary) / "cross-boundary.cnf"
            path.write_bytes(content)
            original_read_bytes = dimacs._READ_BYTES
            try:
                dimacs._READ_BYTES = 5
                info = dimacs.inspect_dimacs(path)
            finally:
                dimacs._READ_BYTES = original_read_bytes
            self.assertEqual(4, info.literal_occurrences)
            self.assertEqual(4, info.max_clause_length)
            self.assertEqual(len(content), info.cnf_bytes)
            self.assertEqual(hashlib.sha256(content).hexdigest(), info.cnf_sha256)


if __name__ == "__main__":
    unittest.main()

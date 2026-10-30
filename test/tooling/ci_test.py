"""Check matrix boundaries and evidence-based nightly skipping."""

import datetime
import unittest
from unittest.mock import patch

from harness_test import load


class CiTest(unittest.TestCase):
    def test_full_matrix_covers_seven_distinct_native_targets(self):
        ci = load("ci")
        targets = {
            row["target"] for row in ci.matrix("full")["include"] + ci.matrix("alpine")["include"]
        }
        self.assertEqual(len(targets), 7)
        self.assertLess(len(ci.matrix("pr")["include"]), len(ci.matrix("full")["include"]))

    def marker(self, age=0):
        created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=age)
        return {
            "expired": False,
            "created_at": created.isoformat(),
            "workflow_run": {"id": 42, "head_branch": "main", "head_sha": "a" * 40},
        }

    def test_recent_success_on_same_sha_skips(self):
        ci = load("ci")
        with patch.object(
            ci,
            "api",
            side_effect=[
                {"artifacts": [self.marker()]},
                {"conclusion": "success", "event": "push"},
            ],
        ):
            self.assertFalse(ci.should_run("a" * 40))

    def test_failed_old_or_different_sha_cannot_skip(self):
        ci = load("ci")
        with patch.object(
            ci,
            "api",
            side_effect=[
                {"artifacts": [self.marker()]},
                {"conclusion": "failure", "event": "push"},
            ],
        ):
            self.assertTrue(ci.should_run("a" * 40))
        with patch.object(ci, "api", return_value={"artifacts": [self.marker(age=8)]}):
            self.assertTrue(ci.should_run("a" * 40))
        with patch.object(ci, "api", return_value={"artifacts": [self.marker()]}):
            self.assertTrue(ci.should_run("b" * 40))


if __name__ == "__main__":
    unittest.main()

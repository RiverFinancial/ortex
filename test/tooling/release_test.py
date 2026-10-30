"""Test release classification and fail-closed artifact checks."""

import hashlib
import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_test import load


class ReleaseTest(unittest.TestCase):
    def test_conventional_commits_keep_breaking_and_unknown_subjects(self):
        release = load("release")
        self.assertEqual(
            release.group_commit("feat(io): add names", ""), ("Features", "io: add names")
        )
        self.assertEqual(
            release.group_commit("fix!: change shape", ""), ("Breaking changes", "change shape")
        )
        self.assertEqual(
            release.group_commit("chore: update runtime", "BREAKING CHANGE: new floor"),
            ("Breaking changes", "update runtime"),
        )
        self.assertEqual(
            release.group_commit("An inherited subject", ""), ("Other", "An inherited subject")
        )

    def test_binary_architecture_must_match_target(self):
        package = load("package")
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 62)
        package.inspect_binary(header, "x86_64-unknown-linux-gnu")
        with self.assertRaisesRegex(ValueError, "architecture"):
            package.inspect_binary(header, "aarch64-unknown-linux-gnu")

    def test_complete_assembly_rejects_missing_targets(self):
        package = load("package")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ci").mkdir()
            (root / "dist").mkdir()
            (root / "mix.exs").write_text('version: "1.2.3"')
            (root / "ci/compatibility.json").write_text(
                json.dumps(
                    {"pr": [{"target": "x86_64-unknown-linux-gnu"}], "platforms": [], "alpine": []}
                )
            )
            with (
                patch.object(package, "ROOT", root),
                self.assertRaisesRegex(ValueError, "missing targets"),
            ):
                package.assemble(True)

    def test_archive_rejects_extra_members(self):
        package = load("package")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ci").mkdir()
            (root / "dist").mkdir()
            (root / "mix.exs").write_text('version: "1.2.3"')
            (root / "ci/compatibility.json").write_text(
                json.dumps(
                    {"pr": [{"target": "x86_64-unknown-linux-gnu"}], "platforms": [], "alpine": []}
                )
            )
            with tarfile.open(
                root / "dist/ortex-1.2.3-x86_64-unknown-linux-gnu.tar.gz", "w:gz"
            ) as archive:
                for name in ["libortex.so", "surprise"]:
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))
            with (
                patch.object(package, "ROOT", root),
                self.assertRaisesRegex(ValueError, "contents"),
            ):
                package.assemble(True)

    def test_changed_artifact_and_partial_bundle_cannot_be_verified(self):
        release = load("release")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "package.tar"
            artifact.write_bytes(b"approved")
            manifest = {
                "version": release.version(),
                "complete": False,
                "sha256": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
            }
            (directory / "artifacts.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "every target"):
                release.verify_artifacts(directory)
            manifest["complete"] = True
            (directory / "artifacts.json").write_text(json.dumps(manifest))
            artifact.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                release.verify_artifacts(directory)


if __name__ == "__main__":
    unittest.main()

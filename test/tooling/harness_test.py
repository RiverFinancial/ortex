"""Exercise staging isolation, scenario timeouts, and package extraction boundaries."""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HarnessTest(unittest.TestCase):
    def test_staged_check_sees_index_and_preserves_unstaged_contents(self):
        checker = load("check")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            source = root / "space $(echo not-executed).py"
            source.write_text("import os\n")  # invalid staged content
            subprocess.run(["git", "-C", str(root), "add", "--", source.name], check=True)
            source.write_text('print("valid working tree")\n')
            with (
                patch.object(checker, "ROOT", root),
                patch.object(sys, "argv", ["check", "--staged"]),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(checker.main(), 1)
            self.assertIn("F401", output.getvalue())
            self.assertEqual(source.read_text(), 'print("valid working tree")\n')
            staged = subprocess.check_output(
                ["git", "-C", str(root), "show", f":{source.name}"], text=True
            )
            self.assertEqual(staged, "import os\n")

    def test_scenario_success_and_failure_are_preserved(self):
        runner = load("scenario")
        status, output = runner.execute(
            [sys.executable, "-c", 'print("done"); raise SystemExit(7)'], 10
        )
        self.assertEqual(status, 7)
        self.assertIn("done", output)

    def test_scenario_timeout_terminates_the_process(self):
        runner = load("scenario")
        before = time.monotonic()
        status, output = runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], 0.2)
        self.assertEqual(status, 124)
        self.assertLess(time.monotonic() - before, 5)
        self.assertIn("exceeded", output)

    def test_hex_contents_cannot_escape_the_unpack_directory(self):
        package = load("package")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as inner:
                entry = tarfile.TarInfo("../escaped")
                entry.size = 4
                inner.addfile(entry, io.BytesIO(b"oops"))
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as outer:
                entry = tarfile.TarInfo("contents.tar.gz")
                entry.size = len(payload.getvalue())
                outer.addfile(entry, io.BytesIO(payload.getvalue()))
            with self.assertRaises(tarfile.FilterError):
                package.unpack(archive, root / "unpacked")
            self.assertFalse((root / "escaped").exists())


if __name__ == "__main__":
    unittest.main()

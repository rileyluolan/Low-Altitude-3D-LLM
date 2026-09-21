"""CPU-only checks for paired recipes, source data preservation and run isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import run


class ExperimentContract(unittest.TestCase):
    def setUp(self):
        temporary_root = run.ROOT / "runtime/test"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "dataset"
        info = self.data / "train/meta/info.json"
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps({
            "total_episodes": 5175, "total_frames": 1720096, "fps": 5,
            "features": {k: {"shape": s} for k, s in {
                "state": [4], "actions": [4], "image": [256, 256, 3], "first_image": [256, 256, 3]
            }.items()},
        }))

    def args(self, *extra):
        return run.arguments(["prepare", "--data-root", str(self.data),
                              "--work-dir", str(self.root / "output"), *extra])

    def test_pair_changes_only_treatment_and_output_identity(self):
        for epochs in (1, 5):
            a = run.plan(self.args("--variant", "no_state", "--epochs", str(epochs)))
            b = run.plan(self.args("--variant", "state", "--epochs", str(epochs)))
            changed = {key for key in a if a[key] != b[key]}
            self.assertEqual(changed, {"discrete_state_input", "config_name", "checkpoint_root"})
            self.assertFalse(a["discrete_state_input"])
            self.assertTrue(b["discrete_state_input"])

    def test_prepare_preserves_source_and_rejects_wrong_link(self):
        source_before = {p.relative_to(self.data): p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        arms = [self.args("--variant", variant) for variant in ("state", "no_state")]
        for args in arms:
            run.prepare(args, run.plan(args))
        stats = list((self.root / "output/assets").rglob("norm_stats.json"))
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0].read_bytes(), stats[1].read_bytes())
        source_after = {p.relative_to(self.data): p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        self.assertEqual(source_before, source_after)
        link = self.root / "output/data/task_overall/train"
        self.assertTrue(link.is_symlink())
        link.unlink()
        link.symlink_to(self.root / "wrong-dataset")
        with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
            run.prepare(arms[0], run.plan(arms[0]))

    def test_training_refuses_overwrite_and_mismatched_resume(self):
        args = self.args("--run-name", "run1")
        recipe = run.plan(args)
        run.record_training(args, recipe)
        Path(recipe["checkpoint_root"]).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            run.record_training(args, recipe)
        args.resume = True
        run.record_training(args, recipe)
        args.seed += 1
        with self.assertRaisesRegex(RuntimeError, "identical recorded recipe"):
            run.record_training(args, run.plan(args))

    def test_output_names_cannot_escape_run_directory(self):
        for flag in ("--run-name", "--eval-tag"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(flag, "../other-experiment")


if __name__ == "__main__":
    unittest.main()

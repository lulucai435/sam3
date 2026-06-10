"""Unit tests for scripts/build_oxe_robot_label_registry.py (handoff §10)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))

import build_oxe_robot_label_registry as M  # noqa: E402


# ------------------------------------------------------------------ normalize


class TestNormalize(unittest.TestCase):
    def test_lowercase_and_strip(self):
        self.assertEqual(M.normalize_label("  Robotic Arm  "), "robotic arm")

    def test_underscore_to_space(self):
        self.assertEqual(M.normalize_label("black_robotic_arm"), "black robotic arm")

    def test_hyphen_to_space(self):
        self.assertEqual(M.normalize_label("black-robotic-arm"), "black robotic arm")

    def test_punctuation_to_space(self):
        self.assertEqual(M.normalize_label("robot, arm."), "robot arm")

    def test_collapse_repeated_whitespace(self):
        self.assertEqual(M.normalize_label("robotic    arm\t\tbase"), "robotic arm base")

    def test_nfkc(self):
        # Full-width characters → ascii equivalents
        self.assertEqual(M.normalize_label("Ｒｏｂｏｔｉｃ Ａｒｍ"), "robotic arm")

    def test_non_string(self):
        self.assertEqual(M.normalize_label(None), "")
        self.assertEqual(M.normalize_label(42), "")
        self.assertEqual(M.normalize_label([]), "")

    def test_only_punctuation_returns_empty(self):
        self.assertEqual(M.normalize_label(",.-_"), "")


# ------------------------------------------------------------------ signature


class TestSignature(unittest.TestCase):
    def test_same_signature_for_same_input(self):
        s1 = M.make_signature("bridge", "pick up cup", ["robotic arm", "cup", "table"])
        s2 = M.make_signature("bridge", "pick up cup", ["table", "cup", "robotic arm"])
        self.assertEqual(s1, s2, "order should not matter")

    def test_different_signature_for_different_dataset(self):
        self.assertNotEqual(
            M.make_signature("bridge", "x", ["a", "b"]),
            M.make_signature("droid", "x", ["a", "b"]),
        )

    def test_different_signature_for_different_instruction(self):
        self.assertNotEqual(
            M.make_signature("bridge", "x", ["a"]),
            M.make_signature("bridge", "y", ["a"]),
        )

    def test_different_signature_for_different_objects(self):
        self.assertNotEqual(
            M.make_signature("bridge", "x", ["a", "b"]),
            M.make_signature("bridge", "x", ["a", "c"]),
        )


# ------------------------------------------------------------------ qwen parse


class TestQwenParse(unittest.TestCase):
    def _make_items(self):
        return [
            {"id": 0, "dataset": "x", "instruction": "i",
             "objects": ["robotic arm", "table"]},
            {"id": 1, "dataset": "x", "instruction": "i",
             "objects": ["yellow robot", "couch"]},
        ]

    def test_normal_response(self):
        parsed = {
            "results": [
                {"id": 0, "robot_labels": [
                    {"label": "robotic arm", "category": "robotic_arm",
                     "confidence": 1.0, "reason": "canonical"}
                ]},
                {"id": 1, "robot_labels": []},
            ]
        }
        out = M.validate_qwen_batch(parsed, self._make_items())
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0]["label"], "robotic arm")
        self.assertEqual(out[1], [])

    def test_missing_id_rejected(self):
        parsed = {"results": [
            {"id": 0, "robot_labels": []},
            # missing id=1
        ]}
        with self.assertRaisesRegex(ValueError, "missing ids"):
            M.validate_qwen_batch(parsed, self._make_items())

    def test_duplicate_id_rejected(self):
        parsed = {"results": [
            {"id": 0, "robot_labels": []},
            {"id": 0, "robot_labels": []},
        ]}
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            M.validate_qwen_batch(parsed, self._make_items())

    def test_changed_label_rejected(self):
        parsed = {"results": [
            {"id": 0, "robot_labels": [
                {"label": "robot arm", "category": "robotic_arm",  # not in input
                 "confidence": 1.0}
            ]},
            {"id": 1, "robot_labels": []},
        ]}
        with self.assertRaisesRegex(ValueError, "not in input objects"):
            M.validate_qwen_batch(parsed, self._make_items())

    def test_bad_category_rejected(self):
        parsed = {"results": [
            {"id": 0, "robot_labels": [
                {"label": "robotic arm", "category": "manipulator", "confidence": 1.0}
            ]},
            {"id": 1, "robot_labels": []},
        ]}
        with self.assertRaisesRegex(ValueError, "bad category"):
            M.validate_qwen_batch(parsed, self._make_items())

    def test_unexpected_id_rejected(self):
        parsed = {"results": [
            {"id": 99, "robot_labels": []},
        ]}
        with self.assertRaisesRegex(ValueError, "unexpected id"):
            M.validate_qwen_batch(parsed, [{"id": 0, "dataset": "x", "instruction": "",
                                            "objects": ["a"]}])


# ------------------------------------------------------------------ bucket


class TestBucket(unittest.TestCase):
    def test_bucket_zero(self):
        self.assertEqual(M.bucket_labels([])[0], "bucket_zero")
        # only non-robot entries → also zero
        self.assertEqual(M.bucket_labels([
            {"label": "table", "category": "non_robot", "confidence": 1.0}
        ])[0], "bucket_zero")

    def test_bucket_single(self):
        bucket, key = M.bucket_labels([
            {"label": "robotic arm", "category": "robotic_arm", "confidence": 1.0}
        ])
        self.assertEqual(bucket, "bucket_single")
        self.assertEqual(key, ("robotic arm",))

    def test_bucket_multi(self):
        bucket, key = M.bucket_labels([
            {"label": "robotic arm", "category": "robotic_arm", "confidence": 1.0},
            {"label": "sara robot base", "category": "robotic_arm", "confidence": 1.0},
        ])
        self.assertEqual(bucket, "bucket_multi")
        self.assertEqual(key, ("robotic arm", "sara robot base"))

    def test_dedup_labels_within_metadata(self):
        bucket, key = M.bucket_labels([
            {"label": "robotic arm", "category": "robotic_arm", "confidence": 1.0},
            {"label": "robotic arm", "category": "robotic_arm", "confidence": 0.9},
        ])
        self.assertEqual(bucket, "bucket_single")
        self.assertEqual(key, ("robotic arm",))


# ------------------------------------------------------------------ scan + resume


class TestScanAndResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.out = Path(self.tmp.name) / "out"
        self.root.mkdir()
        self.out.mkdir()
        # Create 3 fake metadata.json files in nested dirs
        self._make_meta("ds1/v1/shard1/ep1", {
            "language_instruction": "pick", "objects": ["Robotic Arm", "Table"],
            "primary_camera": "cam", "gemini_frame_indices": [0, 10],
            "gemini_model": "test"
        })
        self._make_meta("ds1/v1/shard1/ep2", {
            "language_instruction": "pick", "objects": ["robotic arm", "table"],
            "primary_camera": "cam", "gemini_frame_indices": [0, 10],
            "gemini_model": "test"
        })
        self._make_meta("ds2/v1/shard1/ep1", {
            "language_instruction": "push", "objects": ["robotic arm", "yellow robot"],
            "primary_camera": "cam", "gemini_frame_indices": [0, 10],
            "gemini_model": "test"
        })

    def tearDown(self):
        self.tmp.cleanup()

    def _make_meta(self, relpath: str, contents: dict):
        p = self.root / relpath / "metadata.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(contents))

    def test_scan_indexes_all(self):
        M.stage_scan(self.root, self.out, checkpoint_every=1, log_every=1)
        lines = [json.loads(l) for l in (self.out / "metadata_index.jsonl").read_text().splitlines() if l]
        self.assertEqual(len(lines), 3)
        # ds1 ep1/ep2 should have same signature (same dataset, instruction, normalized objects)
        sigs_ds1 = [l["signature"] for l in lines if l["dataset"] == "ds1"]
        self.assertEqual(len(sigs_ds1), 2)
        self.assertEqual(sigs_ds1[0], sigs_ds1[1])

    def test_scan_resume_skips_processed(self):
        M.stage_scan(self.root, self.out, checkpoint_every=1, log_every=1)
        first_lines = (self.out / "metadata_index.jsonl").read_text().splitlines()
        # Second run shouldn't add any new lines
        M.stage_scan(self.root, self.out, checkpoint_every=1, log_every=1)
        second_lines = (self.out / "metadata_index.jsonl").read_text().splitlines()
        self.assertEqual(len(first_lines), len(second_lines))


# ------------------------------------------------------------------ classify resume


class TestClassifyResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        # Make a tiny index with 2 signatures
        rows = [
            {"path": "/a/meta.json", "dataset": "ds", "model": "t",
             "instruction": "pick", "objects_raw": ["robotic arm", "table"],
             "objects_norm": ["robotic arm", "table"],
             "signature": M.make_signature("ds", "pick", ["robotic arm", "table"]),
             "primary_camera": "c", "gemini_frame_indices": []},
            {"path": "/b/meta.json", "dataset": "ds", "model": "t",
             "instruction": "push", "objects_raw": ["robotic arm", "cup"],
             "objects_norm": ["robotic arm", "cup"],
             "signature": M.make_signature("ds", "push", ["robotic arm", "cup"]),
             "primary_camera": "c", "gemini_frame_indices": []},
        ]
        (self.out / "metadata_index.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        # Pre-populate checkpoint with first signature already done
        (self.out / "qwen_classification_checkpoint.jsonl").write_text(
            json.dumps({"signature": rows[0]["signature"],
                        "robot_labels": [
                            {"label": "robotic arm", "category": "robotic_arm",
                             "confidence": 1.0, "reason": "canonical"}
                        ],
                        "model": "test", "prompt_version": "x", "ts": 0}) + "\n")
        # Make a fake response for the remaining sig
        self.expected_remaining_sig = rows[1]["signature"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_classify_skips_done_and_only_calls_remaining(self):
        fake_response = {
            "results": [
                {"id": 0, "robot_labels": [
                    {"label": "robotic arm", "category": "robotic_arm",
                     "confidence": 1.0, "reason": "x"}
                ]}
            ]
        }
        called_items: list = []

        def fake_call(items, *a, **kw):
            called_items.append(list(items))
            return fake_response

        with patch.object(M, "call_qwen_text", side_effect=fake_call), \
             patch.dict("os.environ", {"DASHSCOPE_API_KEY": "fake-key"}):
            M.stage_classify(self.out, model="m", batch_size=10,
                             timeout=5.0, max_retries=1)
        # Only 1 sig was todo → 1 batch with 1 item
        self.assertEqual(len(called_items), 1)
        self.assertEqual(len(called_items[0]), 1)
        # checkpoint has both sigs now
        ckp = (self.out / "qwen_classification_checkpoint.jsonl").read_text().splitlines()
        sigs_in_ckp = {json.loads(l)["signature"] for l in ckp}
        self.assertIn(self.expected_remaining_sig, sigs_in_ckp)


# ------------------------------------------------------------------ overrides


class TestManualOverrides(unittest.TestCase):
    def test_override_loaded_and_normalized(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("Mechanical Arm: robotic_arm\nclaw: gripper\nhuman hand: non_robot\n")
            p = Path(f.name)
        try:
            overrides = M.load_manual_overrides(p)
            self.assertEqual(overrides["mechanical arm"], "robotic_arm")
            self.assertEqual(overrides["claw"], "gripper")
            self.assertEqual(overrides["human hand"], "non_robot")
        finally:
            p.unlink()

    def test_override_rejects_invalid_category(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("foo: bogus_category\n")
            p = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                M.load_manual_overrides(p)
        finally:
            p.unlink()


# ------------------------------------------------------------------ finalize: yaml stability + uncertain exclusion


class TestFinalizeOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        # Minimal index: 3 metadata, one with a uncertain label
        rows = [
            {"path": "/a/meta.json", "dataset": "ds1", "model": "t",
             "instruction": "pick", "objects_raw": ["robotic arm", "table"],
             "objects_norm": ["robotic arm", "table"],
             "signature": "sig_a",
             "primary_camera": "c", "gemini_frame_indices": []},
            {"path": "/b/meta.json", "dataset": "ds1", "model": "t",
             "instruction": "pick", "objects_raw": ["robotic arm", "table"],
             "objects_norm": ["robotic arm", "table"],
             "signature": "sig_a",
             "primary_camera": "c", "gemini_frame_indices": []},
            {"path": "/c/meta.json", "dataset": "ds2", "model": "t",
             "instruction": "x", "objects_raw": ["yellow robot", "couch"],
             "objects_norm": ["yellow robot", "couch"],
             "signature": "sig_b",
             "primary_camera": "c", "gemini_frame_indices": []},
        ]
        (self.out / "metadata_index.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        # Classification: robotic arm → robotic_arm; yellow robot → uncertain
        (self.out / "qwen_classification_checkpoint.jsonl").write_text(
            json.dumps({"signature": "sig_a",
                        "robot_labels": [
                            {"label": "robotic arm", "category": "robotic_arm",
                             "confidence": 1.0, "reason": "x"}
                        ],
                        "model": "m", "prompt_version": "v", "ts": 0}) + "\n" +
            json.dumps({"signature": "sig_b",
                        "robot_labels": [
                            {"label": "yellow robot", "category": "uncertain",
                             "confidence": 0.5, "reason": "x"}
                        ],
                        "model": "m", "prompt_version": "v", "ts": 0}) + "\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_finalize_excludes_uncertain_from_registry(self):
        M.stage_finalize(self.out, None, 0.85, "m")
        import yaml as _yaml
        registry = _yaml.safe_load((self.out / "robot_label_registry.yaml").read_text())
        self.assertIn("robotic arm", registry["labels"])
        self.assertNotIn("yellow robot", registry["labels"])
        # uncertain CSV should contain yellow robot
        csv_txt = (self.out / "uncertain_labels.csv").read_text()
        self.assertIn("yellow robot", csv_txt)

    def test_yaml_output_stable(self):
        M.stage_finalize(self.out, None, 0.85, "m")
        a = (self.out / "robot_label_registry.yaml").read_bytes()
        M.stage_finalize(self.out, None, 0.85, "m")
        b = (self.out / "robot_label_registry.yaml").read_bytes()
        self.assertEqual(a, b)

    def test_override_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("yellow robot: robotic_arm\n")
            p = Path(f.name)
        try:
            M.stage_finalize(self.out, p, 0.85, "m")
            classifications = json.loads((self.out / "all_label_classifications.json").read_text())
            self.assertEqual(classifications["yellow robot"]["category"], "robotic_arm")
            self.assertEqual(classifications["yellow robot"]["source"], "manual_override")
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()

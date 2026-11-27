import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_pilot import summarize_pilot


FIXTURES = Path(__file__).parent / "fixtures/s3"


class SummarizePilotTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.metrics = Path(directory.name) / "synthetic.jsonl"
        self.output = Path(directory.name) / "summary-test.json"

    def test_first_successful_updates_and_step_distance(self):
        result = summarize_pilot(FIXTURES / "metrics_repeated.jsonl", self.output)
        self.assertEqual(result, {"s_per_step": 6, "peak_vram_gb": 2,
                                  "max_length": 2048, "n_dropped_overlength": 2})
        self.assertEqual(json.loads(self.output.read_bytes()), result)

    def test_missing_vram_serializes_as_null(self):
        summarize_pilot(FIXTURES / "metrics_missing_vram.jsonl", self.output)
        self.assertIsNone(json.loads(self.output.read_bytes())["peak_vram_gb"])

    def test_partial_vram_is_unknown_and_step_zero_is_excluded(self):
        lines = (FIXTURES / "metrics_missing_vram.jsonl").read_bytes().split(b"\n")
        lines.insert(1, b'{"step":0,"optimizer_updated":true,"time_unix":0}')
        last = dict(json.loads(lines[-2]), peak_vram_bytes=1073741824)
        lines[-2] = json.dumps(last).encode()
        self.metrics.write_bytes(b"\n".join(lines))
        result = summarize_pilot(self.metrics)
        self.assertIsNone(result["peak_vram_gb"])
        self.assertEqual(result["s_per_step"], 10)

    def test_maximum_includes_intermediate_success_and_fractional_gb(self):
        rows = [json.loads(line) for line in
                (FIXTURES / "metrics_repeated.jsonl").read_bytes().split(b"\n") if line]
        for row in rows:
            if row.get("step") == 2 and row["optimizer_updated"]:
                row["peak_vram_bytes"] = 5 * 2**29
        self.metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(summarize_pilot(self.metrics)["peak_vram_gb"], 2.5)

    def test_resumed_logs_rejected_before_writing(self):
        for name, reason in (("metrics_two_configs", "Multiple config"),
                             ("metrics_decreasing", "Decreasing steps")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, reason):
                summarize_pilot(FIXTURES / (name + ".jsonl"), self.output)
            self.assertFalse(self.output.exists())

    def test_insufficient_distinct_successes(self):
        lines = (FIXTURES / "metrics_missing_vram.jsonl").read_bytes().split(b"\n")
        self.metrics.write_bytes(b"\n".join([lines[0], lines[1], lines[1]]))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            summarize_pilot(self.metrics)

    def test_missing_config_rejected(self):
        self.metrics.write_bytes(b'{"step":1,"optimizer_updated":true,"time_unix":10}\n')
        with self.assertRaisesRegex(ValueError, "Missing config"):
            summarize_pilot(self.metrics)

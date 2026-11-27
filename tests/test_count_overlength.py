import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.count_overlength import count_overlength
from src.training.train_sft_generator import CompletionOnlyDataCollator


FIXTURE = Path(__file__).parent / "fixtures/s3/sft_examples.jsonl"


class CharacterTokenizer:
    """Test-only tokenizer with one ID per Unicode character and distinct BOS/EOS."""
    eos_token_id = 2

    def encode(self, text, add_special_tokens):
        return ([1] if add_special_tokens else []) + list(map(ord, text))


class CountOverlengthTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "example.jsonl"
        self.tokenizer = CharacterTokenizer()
        self.lines = FIXTURE.read_bytes().split(b"\n")[:-1]

    def test_full_schema_evidence_and_overlength(self):
        self.assertIn("\u2028".encode(), self.lines[0])
        result = count_overlength(FIXTURE, self.tokenizer, 400)
        self.assertEqual(result, dict(n=3, n_kept=2, n_dropped_overlength=1,
                                      max_length=400, epoch_steps=1))

    def test_v2_prompt_masks_evidence_and_supervises_only_completion(self):
        collator = CompletionOnlyDataCollator(self.tokenizer)
        for line in self.lines[:2]:
            row = json.loads(line)
            with self.subTest(evidence=row["evidence"]):
                encoded = collator.encode_example(row)
                end = encoded["pred_position"] + 1
                prompt = "".join(map(chr, encoded["input_ids"][1:end]))  # skip BOS
                self.assertTrue(prompt.endswith(
                    f'<schema>{row["schema"]}</schema>\n<evidence>{row["evidence"]}</evidence>'
                    f'\n<question>{row["question"]}</question>\n[PRED]'
                ))
                self.assertEqual(encoded["labels"][:end], [-100] * end)
                target = f'<think>\n\n</think>\n<answer>\n{row["answer"]}\n</answer>'
                self.assertEqual(encoded["labels"][end:], list(map(ord, target)) + [2])

    def test_default_2048_drops_a_wide_full_schema(self):
        row = json.loads(self.lines[1])
        columns = ", ".join(f"c{i} TEXT" for i in range(256))
        row.update(db_id="toy_wide", schema=f"CREATE TABLE items (n INTEGER, {columns});")
        self.path.write_bytes(self.lines[1] + b"\n" + json.dumps(row).encode())
        self.assertEqual(count_overlength(self.path, self.tokenizer), dict(
            n=2, n_kept=1, n_dropped_overlength=1, max_length=2048, epoch_steps=1
        ))

    def test_exact_sequence_limit_includes_eos(self):
        row = json.loads(self.lines[1])
        self.path.write_bytes(self.lines[1])  # A final LF is optional.
        encoded = CompletionOnlyDataCollator(self.tokenizer).encode_example(row)
        self.assertEqual(encoded["input_ids"][-1], self.tokenizer.eos_token_id)
        width = len(encoded["input_ids"])
        self.assertEqual(count_overlength(self.path, self.tokenizer, width)["n_kept"], 1)
        self.assertEqual(count_overlength(self.path, self.tokenizer, width - 1)["n_kept"], 0)

    def test_filtered_ids_reach_collator_with_original_indices(self):
        rows = [dict(json.loads(self.lines[i]), question_id=f"sft_train:{i}") for i in (0, 2)]
        self.path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
        original = CompletionOnlyDataCollator.encode_example
        with patch.object(CompletionOnlyDataCollator, "encode_example", autospec=True,
                          side_effect=original) as encode:
            self.assertEqual(count_overlength(self.path, self.tokenizer)["n_kept"], 2)
        self.assertEqual([call.args[1] for call in encode.call_args_list], rows)

    def test_blank_row_reports_zero_based_line(self):
        self.path.write_bytes(self.lines[0] + b"\n \t\n" + self.lines[1])
        with self.assertRaisesRegex(ValueError, "0-based line 1"):
            count_overlength(self.path, self.tokenizer)

    def test_standalone_cr_is_not_a_record_separator(self):
        self.path.write_bytes(b"\r".join(self.lines[:2]))
        with self.assertRaises(json.JSONDecodeError):
            count_overlength(self.path, self.tokenizer)

    def test_missing_sql_propagates(self):
        row = json.loads(self.lines[1])
        del row["answer"]
        self.path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "requires answer"):
            count_overlength(self.path, self.tokenizer)

    def test_other_max_length_message_propagates(self):
        self.path.write_text('{"input_ids":[1,2],"labels":[1,2]}')
        with self.assertRaisesRegex(ValueError, "Encoded example exceeds max_length$"):
            count_overlength(self.path, self.tokenizer, 1)

    def test_tokenizer_error_propagates(self):
        with patch.object(self.tokenizer, "encode", side_effect=RuntimeError("tokenizer unavailable")):
            with self.assertRaisesRegex(RuntimeError, "tokenizer unavailable"):
                count_overlength(FIXTURE, self.tokenizer)

    def test_epoch_steps_boundary_on_kept_rows(self):
        for kept, expected in ((7424, 928), (7425, 929)):
            with self.subTest(kept=kept):
                self.path.write_bytes((self.lines[1] + b"\n") * kept + self.lines[2])
                result = count_overlength(self.path, self.tokenizer, 400)
                self.assertEqual((result["n_kept"], result["epoch_steps"]), (kept, expected))

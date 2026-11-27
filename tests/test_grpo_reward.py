"""W1 contract tests on the repository mini DB; never model benchmark results."""
import inspect
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch

from src.training.grpo_reward import (
    GoldExecutionError, GoldExecutionMonitor, GoldErrorRateExceeded,
    load_gold_rows, clear_gold_cache, compute_arctic_reward,
    dr_grpo_advantages, dr_grpo_token_objective,
)
from src.training.train_grpo_generator import (
    DenseGRPOTrainer, compute_dense_rewards, compute_exec_reward,
    grpo_token_objective, normalize_group_advantages, parse_args,
)


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/mini/financial_mini.sqlite"
GOLD = "SELECT client_id, name FROM clients WHERE district_id = 1"


def response(sql, think=""):
    return f"<think>{think}</think><answer>{sql}</answer>"


class TestArcticExecutionReward(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DB.is_file(), f"Required repository mini DB missing: {DB}")
        clear_gold_cache()
        self.gold_rows = load_gold_rows(GOLD, DB)

    def test_exact_result_and_empty_think(self):
        self.assertEqual(compute_arctic_reward(response(GOLD), self.gold_rows, DB), 1.0)

    def test_row_order_and_duplicates_use_set_comparison(self):
        sql = GOLD + " UNION ALL " + GOLD + " ORDER BY client_id DESC"
        self.assertEqual(compute_arctic_reward(response(sql), self.gold_rows, DB), 1.0)

    def test_executable_wrong_result(self):
        sql = "SELECT client_id, name FROM clients WHERE district_id = 999"
        self.assertEqual(compute_arctic_reward(response(sql), self.gold_rows, DB), 0.1)

    def test_syntax_error_and_missing_table(self):
        for sql in ("SELEC client_id FROM clients", "SELECT * FROM missing_w1_table"):
            with self.subTest(sql=sql):
                self.assertEqual(compute_arctic_reward(response(sql), self.gold_rows, DB), 0.0)

    def test_missing_or_empty_answer(self):
        for text in (GOLD, "<think>SQL</think>", "<answer> </answer>", "<answer>"):
            with self.subTest(text=text):
                self.assertEqual(compute_arctic_reward(text, self.gold_rows, DB), 0.0)

    def test_candidate_timeout_really_executes_for_five_seconds(self):
        sql = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL "
               "SELECT x + 1 FROM n WHERE x < 1000000000000) SELECT SUM(x) FROM n")
        started = time.monotonic()
        value = compute_arctic_reward(response(sql), self.gold_rows, DB, timeout=5.0)
        elapsed = time.monotonic() - started
        self.assertEqual(value, 0.0)
        self.assertGreaterEqual(elapsed, 4.8)
        self.assertLess(elapsed, 20.0, "SQLite VM timeout did not interrupt the recursive query")

    def test_destructive_query_does_not_change_db(self):
        before = DB.read_bytes()
        value = compute_arctic_reward(response("DELETE FROM clients"), self.gold_rows, DB)
        self.assertEqual(value, 0.0)
        self.assertEqual(DB.read_bytes(), before)

    def test_gold_is_cached_once_across_a_group(self):
        import src.training.train_grpo_generator as trainer
        clear_gold_cache()
        with patch.object(trainer, "execute_sqlite_query", wraps=trainer.execute_sqlite_query) as execute:
            for _ in range(4):
                compute_dense_rewards(response(GOLD), GOLD, "unused evidence", str(DB))
        gold_calls = [call for call in execute.call_args_list
                      if call.kwargs.get("timeout") == 30.0]
        candidate_calls = [call for call in execute.call_args_list
                           if call.kwargs.get("timeout") == 5.0]
        self.assertEqual(len(gold_calls), 1)
        self.assertEqual(len(candidate_calls), 4)

    def test_invalid_gold_raises_and_failure_is_cached(self):
        import src.training.train_grpo_generator as trainer
        clear_gold_cache()
        with patch.object(trainer, "execute_sqlite_query", wraps=trainer.execute_sqlite_query) as execute:
            for _ in range(2):
                with self.assertRaises(GoldExecutionError):
                    load_gold_rows("SELECT * FROM missing_gold_w1", DB)
        self.assertEqual(execute.call_count, 1)

    def test_format_and_atomic_are_diagnostics_only(self):
        result = compute_dense_rewards(response(GOLD), GOLD, "ignored", str(DB))
        self.assertEqual(result["r_format"], 1.0)
        self.assertEqual(result["r_atomic"], 1.0)
        self.assertEqual(result["total_reward"], 1.0)
        self.assertNotIn("r_mem", result)
        self.assertEqual(compute_exec_reward(GOLD, GOLD, str(DB)), 1.0)

    def test_equal_bad_rollouts_are_zero_signal(self):
        result = DenseGRPOTrainer().evaluate_group(
            [response("SELECT * FROM missing_candidate")] * 4, GOLD, "", str(DB))
        self.assertEqual(result["raw_rewards"], [0.0] * 4)
        self.assertEqual(result["advantages"], [0.0] * 4)
        self.assertFalse(result["advantage_collapse_prevented"])


class TestDrGRPOContract(unittest.TestCase):
    def test_two_reward_values_center_without_std_scaling(self):
        self.assertEqual(dr_grpo_advantages([1.0, 0.0]), [0.5, -0.5])
        self.assertEqual(normalize_group_advantages([1.0, 0.0]), [0.5, -0.5])

    def test_equal_empty_and_nonfinite_rewards(self):
        self.assertEqual(dr_grpo_advantages([0.1] * 4), [0.0] * 4)
        self.assertEqual(dr_grpo_advantages([]), [])
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                dr_grpo_advantages([0.0, value])

    def test_good_token_gradient_is_independent_of_completion_length(self):
        gradients = []
        for length in (1, 7):
            new = torch.full((length,), -1.0, requires_grad=True)
            objective = dr_grpo_token_objective(new, new.detach().clone(),
                new.detach().clone(), 0.8, norm_constant=256, beta_kl=0.0)
            objective.backward()
            gradients.append(new.grad[0].item())
            # Every supplied completion token, including the last/EOS, participates.
            self.assertTrue(torch.all(new.grad != 0))
        self.assertAlmostEqual(gradients[0], -0.8 / 256, places=8)
        self.assertEqual(gradients[0], gradients[1])

    def test_kl_uses_same_constant_and_detaches_old_and_reference(self):
        gradients = []
        for length in (1, 7):
            new = torch.full((length,), -1.0, requires_grad=True)
            old = torch.full((length,), -1.0, requires_grad=True)
            reference = torch.full((length,), -0.5, requires_grad=True)
            dr_grpo_token_objective(new, old, reference, 0.0,
                norm_constant=256, beta_kl=0.001).backward()
            gradients.append(new.grad[0].item())
            self.assertIsNone(old.grad)
            self.assertIsNone(reference.grad)
        self.assertEqual(gradients[0], gradients[1])
        self.assertAlmostEqual(gradients[0], 0.001 * (1 - math.exp(0.5)) / 256, places=9)

    def test_old_policy_ratio_and_explicit_normalizer(self):
        new = torch.tensor([-0.4], requires_grad=True)
        loss = grpo_token_objective(new, torch.tensor([-1.5]), new.detach(), 1.0,
                                   beta_kl=0.0, norm_constant=256)
        loss.backward()
        self.assertEqual(new.grad.item(), 0.0)
        with self.assertRaises(ValueError):
            dr_grpo_token_objective(new, new, new, 1.0, norm_constant=0)

    def test_cli_constructor_and_objective_default_beta(self):
        with patch("sys.argv", ["w1-test"]):
            self.assertEqual(parse_args().beta_kl, 0.001)
        self.assertEqual(DenseGRPOTrainer().beta_kl, 0.001)
        self.assertEqual(inspect.signature(grpo_token_objective).parameters["beta_kl"].default, 0.001)


class TestGoldErrorRateContract(unittest.TestCase):
    def test_denominator_is_distinct_questions_and_waits_for_100(self):
        monitor = GoldExecutionMonitor()
        for _ in range(8):
            monitor.observe("grpo_train:0", failed=True)
        self.assertEqual(monitor.attempted, 1)
        self.assertEqual(monitor.failed, 1)
        monitor.check()  # No rate decision yet.
        for i in range(1, 100):
            monitor.observe(f"grpo_train:{i}", failed=False)
        monitor.check()  # Exactly 1% is permitted, not greater than 1%.
        monitor.observe("grpo_train:100", failed=True)
        with self.assertRaises(GoldErrorRateExceeded):
            monitor.check()

    def test_resume_keeps_distinct_error_counts(self):
        monitor = GoldExecutionMonitor()
        monitor.observe("grpo_train:7", failed=True)
        restored = GoldExecutionMonitor.from_state(monitor.state())
        restored.observe("grpo_train:7", failed=True)
        self.assertEqual(restored.attempted, 1)
        self.assertEqual(restored.failed, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

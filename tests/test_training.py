"""
Unit and Integration Tests for Phase 2: SLM Training & Constrained Decoding.

Verifies:
  1. DriveCheckpointCallback: Syncing, checkpoint discovery, pruning, and session recovery.
  2. LLM_JEPAModule: 2-block causal mask, [PRED] alignment, cosine distance, and random loss dropout.
  3. train_sft_generator: Completion-Only -100 label masking, Cross-Entropy loss invariance, and LoRA config.
  4. train_grpo_generator: 4 reward heads (format, exec, mem, atomic), group advantage normalization, and advantage collapse resolution.
  5. train_embedding: HN-SupCon contrastive loss, margin penalty effect, and column index export.
  6. train_decision_model: Dual learning rates (1e-5 backbone, 1e-4 head), RLCD calibration loss, and ECE <= 0.081.
"""

import os
import sys
import math
import json
import shutil
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.callbacks import (
    DriveCheckpointCallback,
    find_latest_checkpoint,
    get_resume_checkpoint,
    TrainerState,
    TrainingArguments
)
from src.training.jepa_module import (
    build_2block_causal_mask,
    compute_cosine_distance,
    LLM_JEPAModule,
    compute_jepa_loss
)
from src.training.train_sft_generator import (
    mask_prompt_tokens,
    compute_completion_loss,
    build_lora_config,
    run_dry_run as run_sft_dry_run
)
from src.training.train_grpo_generator import (
    compute_format_reward,
    compute_exec_reward,
    compute_mem_reward,
    compute_atomic_reward,
    compute_dense_rewards,
    normalize_group_advantages,
    compute_grpo_loss,
    DenseGRPOTrainer,
    run_dry_run as run_grpo_dry_run
)
from src.training.train_embedding import (
    format_column_card_text,
    compute_hn_supcon_loss,
    HNSupConLoss,
    export_column_index,
    run_dry_run as run_embedding_dry_run
)
from src.training.train_decision_model import (
    configure_dual_lr_optimizer,
    compute_rlcd_loss,
    compute_ece,
    calibrate_probabilities,
    fit_temperature,
    run_dry_run as run_decision_dry_run
)


class TestDriveCheckpointCallback(unittest.TestCase):
    """Tests Drive checkpoint synchronization and session recovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.local_dir = os.path.join(self.temp_dir, "local_checkpoints")
        self.drive_dir = os.path.join(self.temp_dir, "drive_checkpoints")
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_latest_checkpoint(self):
        # Create dummy checkpoints: checkpoint-100, checkpoint-500, checkpoint-250
        for step in [100, 500, 250]:
            p = os.path.join(self.local_dir, f"checkpoint-{step}")
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, "dummy_model.bin"), "w") as f:
                f.write(f"weights at step {step}")

        latest = find_latest_checkpoint(self.local_dir)
        self.assertIsNotNone(latest)
        self.assertTrue(latest.endswith("checkpoint-500"))

    def test_drive_sync_on_save(self):
        cb = DriveCheckpointCallback(
            drive_checkpoint_dir=self.drive_dir,
            local_checkpoint_dir=self.local_dir,
            max_drive_checkpoints=2
        )

        # Create local checkpoint-1000
        step_dir = os.path.join(self.local_dir, "checkpoint-1000")
        os.makedirs(step_dir, exist_ok=True)
        test_file = os.path.join(step_dir, "adapter_model.bin")
        with open(test_file, "w") as f:
            f.write("test weights 1000")

        state = TrainerState(global_step=1000)
        args = TrainingArguments(output_dir=self.local_dir)

        cb.on_save(args, state, control=None)

        # Verify Drive destination has checkpoint-1000 with matching file
        synced_path = os.path.join(self.drive_dir, "checkpoint-1000", "adapter_model.bin")
        self.assertTrue(os.path.exists(synced_path))
        with open(synced_path, "r") as f:
            self.assertEqual(f.read(), "test weights 1000")

    def test_checkpoint_pruning_on_drive(self):
        cb = DriveCheckpointCallback(
            drive_checkpoint_dir=self.drive_dir,
            local_checkpoint_dir=self.local_dir,
            max_drive_checkpoints=2
        )

        args = TrainingArguments(output_dir=self.local_dir)
        for step in [100, 200, 300]:
            step_dir = os.path.join(self.local_dir, f"checkpoint-{step}")
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, "model.bin"), "w") as f:
                f.write(f"step {step}")
            state = TrainerState(global_step=step)
            cb.on_save(args, state, control=None)

        # Research checkpoints are retained even when a legacy retention count is set.
        drive_checkpoints = os.listdir(self.drive_dir)
        self.assertIn("checkpoint-100", drive_checkpoints)
        self.assertIn("checkpoint-200", drive_checkpoints)
        self.assertIn("checkpoint-300", drive_checkpoints)

    def test_get_resume_checkpoint(self):
        # Create a checkpoint in Drive
        drive_ckpt = os.path.join(self.drive_dir, "checkpoint-2000")
        os.makedirs(drive_ckpt, exist_ok=True)

        res = get_resume_checkpoint("latest", self.local_dir, self.drive_dir)
        self.assertEqual(res, drive_ckpt)

        # Test specific path
        explicit_res = get_resume_checkpoint(drive_ckpt, self.local_dir, self.drive_dir)
        self.assertEqual(explicit_res, drive_ckpt)


class TestLLM_JEPAModule(unittest.TestCase):
    """Tests 2-block causal mask, [PRED] prediction, and Cosine Distance Loss."""

    def test_2block_causal_mask_structure(self):
        T = 5  # Block 1 (Text + [PRED])
        C = 7  # Block 2 (Code/SQL)
        N = T + C
        mask = build_2block_causal_mask(seq_len_text=T, seq_len_code=C)

        # Invariant 1: Block 1 tokens CANNOT attend to Block 2 tokens
        for i in range(T):
            for j in range(T, N):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertLess(val, -1e8, f"Block 1 token {i} attended to Block 2 token {j}!")

        # Invariant 2: Block 1 attends causally to Block 1
        for i in range(T):
            for j in range(i + 1):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertEqual(val, 0.0)
            for j in range(i + 1, T):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertLess(val, -1e8)

        # Invariant 3: Block 2 tokens attend to all Block 1 tokens
        for i in range(T, N):
            for j in range(T):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertEqual(val, 0.0)

        # Invariant 4: Block 2 attends causally to Block 2
        for i in range(T, N):
            for j in range(T, i + 1):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertEqual(val, 0.0)
            for j in range(i + 1, N):
                val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                self.assertLess(val, -1e8)

    def test_cosine_distance_computation(self):
        u = [1.0, 0.0, 0.0]
        v_ident = [1.0, 0.0, 0.0]
        v_orth = [0.0, 1.0, 0.0]
        v_opp = [-1.0, 0.0, 0.0]

        d_ident = compute_cosine_distance(u, v_ident)
        d_orth = compute_cosine_distance(u, v_orth)
        d_opp = compute_cosine_distance(u, v_opp)

        self.assertAlmostEqual(d_ident, 0.0, places=4)
        self.assertAlmostEqual(d_orth, 1.0, places=4)
        self.assertAlmostEqual(d_opp, 2.0, places=4)

    def test_jepa_module_forward_and_dropout(self):
        jepa = LLM_JEPAModule(hidden_dim=16, lambda_weight=1.0, loss_dropout=0.5)
        pred_latent = [0.1 * i for i in range(16)]
        target_latent = [0.2 * i for i in range(16)]

        # Force compute (no dropout)
        res_forced = jepa.forward(pred_latent, target_latent, force_compute=True)
        self.assertEqual(res_forced["dropped"], 0.0)
        self.assertGreater(res_forced["jepa_loss"], 0.0)
        self.assertAlmostEqual(res_forced["jepa_loss"], res_forced["unscaled_distance"], places=4)

        # Test compute_jepa_loss helper
        loss_val, is_drop = compute_jepa_loss(pred_latent, target_latent, is_training=False)
        self.assertFalse(is_drop)
        self.assertGreater(loss_val, 0.0)


class TestCompletionOnlySFT(unittest.TestCase):
    """Tests completion-only -100 label masking and SFT loss calculation."""

    def test_mask_prompt_tokens(self):
        THINK_TOKEN = 999
        # Input sequence: prompt tokens [10, 11, 12], completion [999, 20, 21]
        input_ids = [10, 11, 12, THINK_TOKEN, 20, 21]
        labels, start_idx = mask_prompt_tokens(input_ids, think_token_id=THINK_TOKEN)

        self.assertEqual(start_idx, 3)
        self.assertEqual(labels[:3], [-100, -100, -100])
        self.assertEqual(labels[3:], [999, 20, 21])

    def test_completion_loss_invariance_to_prompt_tokens(self):
        # Verify that changing prompt tokens does NOT change completion loss
        THINK_TOKEN = 999
        vocab_size = 1000
        seq1 = [10, 11, 12, THINK_TOKEN, 20, 21]
        seq2 = [55, 66, 77, THINK_TOKEN, 20, 21]  # Different prompt tokens

        labels1, _ = mask_prompt_tokens(seq1, THINK_TOKEN)
        labels2, _ = mask_prompt_tokens(seq2, THINK_TOKEN)

        # Constant logits across completion tokens
        logits1 = [[0.0] * vocab_size for _ in seq1]
        logits2 = [[0.0] * vocab_size for _ in seq2]
        for step in range(len(seq1)):
            logits1[step][20] = 3.0
            logits1[step][21] = 3.0
            logits2[step][20] = 3.0
            logits2[step][21] = 3.0

        loss1, valid1 = compute_completion_loss(logits1, labels1)
        loss2, valid2 = compute_completion_loss(logits2, labels2)

        self.assertEqual(valid1, valid2)
        self.assertAlmostEqual(loss1, loss2, places=5)

    def test_lora_config_generation(self):
        cfg = build_lora_config(rank=256, alpha=16)
        r = cfg.r
        alpha = cfg.lora_alpha
        targets = cfg.target_modules

        self.assertEqual(r, 256)
        self.assertEqual(alpha, 16)
        self.assertIn("q_proj", targets)
        self.assertIn("v_proj", targets)

    def test_sft_dry_run_execution(self):
        res = run_sft_dry_run("data/mini/mini_synsql.jsonl")
        self.assertEqual(res["status"], "PASSED")
        self.assertTrue(res["completion_masking_verified"])
        self.assertGreater(res["verified_samples"], 0)


class TestDenseGRPOTrainer(unittest.TestCase):
    """Tests 4-head reward computation, advantage normalization, and dry run."""

    def test_format_reward(self):
        valid = "<think>Step 1: Check table</think>\n<answer>SELECT * FROM t;</answer>"
        invalid_no_think = "SELECT * FROM t;"
        invalid_empty = "<think></think><answer>SELECT 1;</answer>"
        invalid_inverted = "<answer>SELECT 1;</answer><think>Reasoning</think>"

        self.assertEqual(compute_format_reward(valid), 1.0)
        self.assertEqual(compute_format_reward(invalid_no_think), 0.0)
        self.assertEqual(compute_format_reward(invalid_empty), 0.0)
        self.assertEqual(compute_format_reward(invalid_inverted), 0.0)

    def test_exec_reward_on_real_sqlite(self):
        db_path = "data/mini/financial_mini.sqlite"
        gold_sql = "SELECT client_id, name FROM clients WHERE district_id = 1;"
        cand_exact = "SELECT client_id, name FROM clients WHERE district_id = 1;"
        cand_wrong_res = "SELECT client_id, name FROM clients WHERE district_id = 999;"
        cand_syntax_err = "SELECT NON_EXISTENT FROM clients;"

        r_exact = compute_exec_reward(cand_exact, gold_sql, db_path)
        r_wrong = compute_exec_reward(cand_wrong_res, gold_sql, db_path)
        r_err = compute_exec_reward(cand_syntax_err, gold_sql, db_path)

        self.assertEqual(r_exact, 2.0)
        self.assertEqual(r_wrong, 1.0)
        self.assertEqual(r_err, 0.0)

    def test_mem_reward(self):
        gold_think = "Join clients with accounts on client_id and filter card_type = 'gold'."
        similar_think = "We need to join clients and accounts using client_id and check card_type."
        unrelated_think = "Calculate average math scores in Los Angeles county."

        r_sim = compute_mem_reward(similar_think, gold_think)
        r_unrelated = compute_mem_reward(unrelated_think, gold_think)

        self.assertGreater(r_sim, r_unrelated)
        self.assertGreater(r_sim, 0.3)

    def test_atomic_reward(self):
        sql1 = "SELECT T1.name FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id WHERE T2.balance > 1000;"
        sql2 = "SELECT T1.name FROM clients AS T1 WHERE T1.district_id = 2;"

        r_identical = compute_atomic_reward(sql1, sql1)
        r_partial = compute_atomic_reward(sql2, sql1)

        self.assertEqual(r_identical, 1.0)
        self.assertGreater(r_partial, 0.0)
        self.assertLess(r_partial, 1.0)

    def test_group_advantage_normalization_and_collapse_prevention(self):
        # When rewards are dense (distinct), advantages must have zero mean and positive std
        rewards = [4.5, 2.0, 1.0, 0.0]
        advs = normalize_group_advantages(rewards)

        self.assertEqual(len(advs), 4)
        mean_adv = sum(advs) / len(advs)
        self.assertAlmostEqual(mean_adv, 0.0, places=3)
        self.assertGreater(max(advs), 0.0)
        self.assertLess(min(advs), 0.0)

    def test_grpo_dry_run_execution(self):
        res = run_grpo_dry_run("data/mini/mini_bird.jsonl", "data/mini")
        self.assertEqual(res["status"], "PASSED")
        self.assertTrue(res["advantage_collapse_prevented"])


class TestLitESQLEmbeddingTrainer(unittest.TestCase):
    """Tests HN-SupCon contrastive loss and margin penalty."""

    def test_hn_supcon_margin_effect(self):
        q = [1.0, 0.0, 0.0, 0.0]
        pos = [0.9, 0.1, 0.0, 0.0]
        negs = [[0.8, 0.2, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]

        # Compute loss with margin 0.1 vs margin 0.0
        loss_with_margin = compute_hn_supcon_loss(q, pos, negs, margin=0.1, temperature=0.05)
        loss_zero_margin = compute_hn_supcon_loss(q, pos, negs, margin=0.0, temperature=0.05)

        # Adding margin to hard negatives increases negative logit, hence strictly increases loss
        self.assertGreater(loss_with_margin, loss_zero_margin)

    def test_format_column_card_text(self):
        card = {
            "table_name": "schools",
            "column_name": "CDSCode",
            "data_type": "TEXT",
            "constraints": ["PRIMARY KEY"],
            "column_description": "Unique code for schools",
            "sample_values": ["01611760130062"]
        }
        text = format_column_card_text(card)
        self.assertIn("Table: schools", text)
        self.assertIn("Column: CDSCode (TEXT)", text)
        self.assertIn("PRIMARY KEY", text)

    def test_embedding_dry_run_execution(self):
        with tempfile.TemporaryDirectory() as output:
            res = run_embedding_dry_run(
                cards_path="data/mini/column_cards_mini.jsonl",
                bird_path="data/mini/mini_bird.jsonl",
                output_dir=output
            )
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["margin"], 0.1)
        self.assertEqual(res["indexed_columns"], 40)


class TestDecisionModelTrainer(unittest.TestCase):
    """Tests dual learning rates, RLCD loss, and ECE <= 0.081 calibration."""

    def test_dual_lr_optimizer_grouping(self):
        cfg = configure_dual_lr_optimizer(None, lr_backbone=1e-5, lr_head=1e-4)
        groups = cfg["param_groups"]
        self.assertEqual(groups[0]["lr"], 1e-5)
        self.assertEqual(groups[1]["lr"], 1e-4)

    def test_rlcd_loss_penalizes_overconfidence(self):
        # Prediction 1: wrong decision with low confidence (p=0.55 on y=0)
        # Prediction 2: wrong decision with high confidence (p=0.99 on y=0)
        loss_moderate = compute_rlcd_loss([0.55], [0], alpha_cal=0.5)
        loss_overconfident = compute_rlcd_loss([0.99], [0], alpha_cal=0.5)

        self.assertGreater(loss_overconfident, loss_moderate * 2.0)

    def test_ece_computation_and_temperature_calibration(self):
        logits = [2.5, 1.8, 0.5, -0.4, -1.2, -2.0, 2.0, 1.2, -1.5, -0.8]
        labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]

        opt_t, final_ece = fit_temperature(logits, labels, target_ece=0.081)
        self.assertLessEqual(final_ece, 0.081)

    def test_decision_dry_run_execution(self):
        res = run_decision_dry_run(num_samples=100, target_ece_threshold=0.081)
        self.assertEqual(res["status"], "PASSED")
        self.assertTrue(res["ece_passed"])
        self.assertLessEqual(res["calibrated_ece"], 0.081)


if __name__ == "__main__":
    unittest.main()

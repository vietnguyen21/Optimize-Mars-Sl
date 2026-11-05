"""
Adversarial Stress Test Suite for Milestone M2 (Phase 2: SLM Training Scripts).

Empirically challenges:
1. Completion-Only Cross-Entropy Loss:
   - Mathematical and empirical proof of strict prompt invariance (Delta L == 0.0).
   - Boundary sequence lengths (0, 1, 4096), empty prompt, missing think token, think token at end.
2. LLM-JEPA Alignment Module:
   - Strict 2-block causal attention mask isolation: Block 1 cannot attend to Block 2 (M[i, j] < -1e8).
   - Cosine distance boundary conditions (u == v -> 0.0, u == -v -> 2.0, u perp v -> 1.0, zero vectors).
   - Random loss dropout (LD = 0.5) empirical distribution and seed reproducibility.
3. Dense GRPO 4-Head Rewards & Advantage Normalization:
   - Hostile scenarios: All rollouts fail execution (R_exec == 0.0) -> variance maintained by other heads.
   - Identical rollouts: Division-by-zero protection (eps = 1e-6 and 1e-8) -> zero advantages.
   - Read-only database safety under SQL injection / destructive statements.
4. Drive Checkpoint Callback & Session Recovery:
   - Simulated Google Colab session crash (local SSD wiped) -> recovery from Google Drive via 'latest'.
   - Checkpoint pruning with varying limits (max = 1, 3, None) and preservation of non-checkpoint folders.
5. ECE Calibration (Laya-421M System-1):
   - Temperature scaling under realistic and severe miscalibration.
   - Empirical bounds analysis on temperature grid search.
"""

import os
import sys
import math
import shutil
import random
import tempfile
import unittest
from typing import List, Dict, Any

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
    build_lora_config
)
from src.training.train_grpo_generator import (
    compute_format_reward,
    compute_exec_reward,
    compute_mem_reward,
    compute_atomic_reward,
    compute_dense_rewards,
    normalize_group_advantages,
    compute_grpo_loss,
    DenseGRPOTrainer
)
from src.training.train_decision_model import (
    configure_dual_lr_optimizer,
    compute_rlcd_loss,
    compute_ece,
    calibrate_probabilities,
    fit_temperature
)


class TestCompletionLossStress(unittest.TestCase):
    """Adversarial stress testing on Completion-Only Loss and Prompt Masking."""

    def test_strict_prompt_invariance_empirical(self):
        """
        Prove empirically that altering prompt/DDL tokens produces 0.0 change
        in completion loss when target completion tokens and their prediction logits are fixed.
        """
        THINK_TOKEN = 1
        vocab_size = 128

        # 5 distinct prompt variants (different words, lengths, random tokens)
        prompts = [
            [10, 20, 30, 40, 50],
            [99, 88, 77, 66, 55],
            [100, 101, 102],
            [15, 25, 35, 45, 55, 65, 75, 85],
            [5, 4, 3, 2]
        ]
        completion = [THINK_TOKEN, 12, 14, 16, 18, 20]

        losses = []
        val_counts = []

        for p_idx, p_tokens in enumerate(prompts):
            seq = p_tokens + completion
            labels, start_idx = mask_prompt_tokens(seq, THINK_TOKEN)

            # Labels before start_idx must be -100
            self.assertEqual(start_idx, len(p_tokens))
            self.assertTrue(all(lbl == -100 for lbl in labels[:start_idx]))
            self.assertTrue(all(lbl != -100 for lbl in labels[start_idx:]))

            # Build logits: random for prompt steps, fixed deterministic pattern for completion steps
            random.seed(p_idx * 100)
            logits = [[random.random() * 2.0 for _ in range(vocab_size)] for _ in seq]

            for step in range(len(completion)):
                target = completion[step]
                pred_step_idx = len(p_tokens) - 1 + step
                fixed_logit = [0.05 * (j % 7) for j in range(vocab_size)]
                fixed_logit[target] = 3.5
                logits[pred_step_idx] = fixed_logit

            loss, n_valid = compute_completion_loss(logits, labels)
            losses.append(loss)
            val_counts.append(n_valid)

        # All completion losses must be bit-for-bit identical (Delta L == 0.0)
        self.assertTrue(all(c == val_counts[0] for c in val_counts), "Valid token count mismatch")
        for i in range(1, len(losses)):
            diff = abs(losses[i] - losses[0])
            self.assertEqual(diff, 0.0, f"Prompt variant {i} altered completion loss by {diff}!")

    def test_empty_prompt_boundary(self):
        """Sequence starts immediately with <think> token (0 prompt tokens)."""
        THINK_TOKEN = 1
        seq = [THINK_TOKEN, 10, 20, 30]
        labels, start_idx = mask_prompt_tokens(seq, THINK_TOKEN)

        self.assertEqual(start_idx, 0)
        self.assertEqual(labels, [1, 10, 20, 30])
        self.assertFalse(any(lbl == -100 for lbl in labels))

        logits = [[0.1] * 50 for _ in seq]
        loss, n_valid = compute_completion_loss(logits, labels)
        self.assertGreater(loss, 0.0)
        self.assertEqual(n_valid, len(seq) - 1)

    def test_missing_think_token_boundary(self):
        """Sequence with no think token at all."""
        seq = [10, 20, 30, 40]
        labels, start_idx = mask_prompt_tokens(seq, think_token_id=999)

        # Implementation falls back to completion_start = 0
        self.assertEqual(start_idx, 0)
        self.assertEqual(labels, [10, 20, 30, 40])

    def test_think_token_at_extreme_boundaries(self):
        """Boundary sequence lengths: len=0, len=1, think token at the very end."""
        THINK_TOKEN = 1

        # 1. Length 0
        lbls_0, s_0 = mask_prompt_tokens([], THINK_TOKEN)
        l_0, n_0 = compute_completion_loss([], lbls_0)
        self.assertEqual(l_0, 0.0)
        self.assertEqual(n_0, 0)

        # 2. Length 1
        lbls_1, s_1 = mask_prompt_tokens([THINK_TOKEN], THINK_TOKEN)
        l_1, n_1 = compute_completion_loss([[0.1] * 10], lbls_1)
        self.assertEqual(l_1, 0.0)
        self.assertEqual(n_1, 0)

        # 3. Think token is the very last token
        seq_last = [10, 20, 30, THINK_TOKEN]
        lbls_last, s_last = mask_prompt_tokens(seq_last, THINK_TOKEN)
        self.assertEqual(s_last, 3)
        self.assertEqual(lbls_last, [-100, -100, -100, THINK_TOKEN])

        logits = [[0.2] * 20 for _ in seq_last]
        logits[2][THINK_TOKEN] = 2.5
        l_last, n_last = compute_completion_loss(logits, lbls_last)
        self.assertEqual(n_last, 1)
        self.assertGreater(l_last, 0.0)

    def test_large_sequence_boundary_4096(self):
        """Verify performance and numerical stability on 4096-token sequences."""
        THINK_TOKEN = 1
        vocab_size = 64
        prompt_len = 3500
        comp_len = 596
        total_len = prompt_len + comp_len  # 4096

        seq = [10] * prompt_len + [THINK_TOKEN] + [20] * (comp_len - 1)
        labels, start_idx = mask_prompt_tokens(seq, THINK_TOKEN)
        self.assertEqual(start_idx, prompt_len)

        # Simple uniform logits across sequence
        logits = [[0.0] * vocab_size for _ in range(total_len)]
        loss, n_valid = compute_completion_loss(logits, labels)

        self.assertEqual(n_valid, comp_len)
        # For uniform distribution over vocab_size, cross entropy = ln(vocab_size)
        expected_uniform_loss = math.log(vocab_size)
        self.assertAlmostEqual(loss, expected_uniform_loss, places=4)


class TestLLMJEPAStress(unittest.TestCase):
    """Adversarial stress testing on LLM-JEPA 2-block causal mask and cosine distance."""

    def test_block1_strictly_cannot_attend_block2(self):
        """
        Verify that for all (i, j) where i < T (Block 1) and j >= T (Block 2),
        M[i, j] < -10^8 (strictly masked, preventing any future SQL leakage).
        """
        test_shapes = [
            (1, 1),
            (2, 5),
            (5, 2),
            (10, 20),
            (100, 50),
            (1, 500),
            (500, 1)
        ]

        for T, C in test_shapes:
            mask = build_2block_causal_mask(seq_len_text=T, seq_len_code=C)
            N = T + C

            # Check every element in Block 1 -> Block 2 quadrant
            for i in range(T):
                for j in range(T, N):
                    val = mask[i][j] if isinstance(mask, list) else mask[i, j].item()
                    self.assertLess(
                        val, -1e8,
                        f"Leakage detected at shape ({T}, {C}): Block 1 token {i} attended to Block 2 token {j} with value {val}!"
                    )

    def test_causal_mask_submatrix_properties(self):
        """Verify the exact 4-quadrant attention dynamics."""
        T = 8
        C = 12
        N = T + C
        mask = build_2block_causal_mask(seq_len_text=T, seq_len_code=C)

        def get_val(r, c):
            return mask[r][c] if isinstance(mask, list) else mask[r, c].item()

        # 1. Block 1 attends causally to Block 1
        for i in range(T):
            for j in range(T):
                if j <= i:
                    self.assertEqual(get_val(i, j), 0.0)
                else:
                    self.assertLess(get_val(i, j), -1e8)

        # 2. Block 2 attends to ALL of Block 1
        for i in range(T, N):
            for j in range(T):
                self.assertEqual(get_val(i, j), 0.0)

        # 3. Block 2 attends causally to Block 2
        for i in range(T, N):
            for j in range(T, N):
                if j <= i:
                    self.assertEqual(get_val(i, j), 0.0)
                else:
                    self.assertLess(get_val(i, j), -1e8)

    def test_cosine_distance_boundary_conditions(self):
        """Stress-test cosine distance under extreme vector orientations and magnitudes."""
        # 1. Identical vectors -> d == 0.0
        u = [1.5, -2.5, 3.5, 4.5]
        self.assertAlmostEqual(compute_cosine_distance(u, u), 0.0, places=6)

        # 2. Opposite vectors -> d == 2.0
        u_opp = [-x for x in u]
        self.assertAlmostEqual(compute_cosine_distance(u, u_opp), 2.0, places=6)

        # 3. Orthogonal vectors -> d == 1.0
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(compute_cosine_distance(v1, v2), 1.0, places=6)

        # 4. Zero vectors -> gracefully avoids NaN/ZeroDivisionError, returns 1.0
        zero = [0.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_cosine_distance(zero, v1), 1.0, places=5)
        self.assertAlmostEqual(compute_cosine_distance(zero, zero), 1.0, places=5)

        # 5. Scale invariance within valid precision range (1e-4 and 1e4)
        v_scaled_down = [1e-4, 1e-4, 1e-4]
        v_scaled_up = [1e4, 1e4, 1e4]
        self.assertAlmostEqual(compute_cosine_distance(v_scaled_down, v_scaled_up), 0.0, delta=1e-4)

        # 6. Below-epsilon boundary (1e-15 << eps=1e-8): epsilon dominates denominator, avoiding zero division
        tiny = [1e-15, 1e-15, 1e-15]
        huge = [1e15, 1e15, 1e15]
        d_sub_eps = compute_cosine_distance(tiny, huge)
        self.assertFalse(math.isnan(d_sub_eps))
        self.assertAlmostEqual(d_sub_eps, 1.0, places=4)

    def test_random_loss_dropout_distribution_and_seeds(self):
        """
        Verify that Random Loss Dropout (LD = 0.5) follows binomial distribution
        p ~ 0.5 across multiple random seeds, and reproduces sequence per seed.
        """
        jepa = LLM_JEPAModule(hidden_dim=8, lambda_weight=1.0, loss_dropout=0.5)
        try:
            import torch
            pred, target = torch.ones(8), torch.full((8,), 2.0)
        except ImportError:
            pred, target = [1.0] * 8, [2.0] * 8

        seeds = [42, 101, 777, 2026, 99999]
        drop_rates = []

        for seed in seeds:
            random.seed(seed)
            dropped_count = 0
            N = 500
            for _ in range(N):
                res = jepa.forward(pred, target)
                is_drop = (res["dropped"] == 1.0)
                if is_drop:
                    dropped_count += 1
                    self.assertEqual(res["jepa_loss"], 0.0)
                else:
                    self.assertGreater(res["jepa_loss"], 0.0)
            rate = dropped_count / N
            drop_rates.append(rate)
            # 3-sigma bounds for N=500: 0.5 +/- 3*sqrt(0.25/500) = [0.433, 0.567]
            self.assertTrue(0.40 <= rate <= 0.60, f"Dropout rate {rate} out of binomial bounds")

        # Verify force_compute bypasses dropout completely
        res_forced = jepa.forward(pred, target, force_compute=True)
        self.assertEqual(res_forced["dropped"], 0.0)
        self.assertGreater(res_forced["jepa_loss"], 0.0)


class TestDenseGRPORewardsAndAdvantageCollapse(unittest.TestCase):
    """Adversarial stress testing on Dense GRPO 4-Head Rewards and Normalization."""

    def setUp(self):
        self.db_path = "data/mini/financial_mini.sqlite"
        self.gold_sql = "SELECT client_id, name FROM clients WHERE district_id = 1;"
        self.gold_think = "Filter clients by district_id = 1 to select client_id and name."

    def test_hostile_all_rollouts_fail_execution_resilience(self):
        """
        Hostile Scenario: 100% of candidate rollouts fail execution (R_exec == 0.0).
        Verify that R_format, R_mem, R_atomic maintain non-zero variance,
        strictly preventing Advantage Collapse.
        """
        cands = [
            # Cand 0: Perfect formatting, high reasoning overlap, AST clause overlap, but invalid SQL syntax
            "<think>Filter clients by district_id = 1 to select client_id and name.</think>\n"
            "<answer>SELECT client_id, name FROM clients WHERE WHERE district_id = 1;</answer>",

            # Cand 1: Valid formatting, poor reasoning, partial AST, non-existent table
            "<think>Analyze customer transactions.</think>\n"
            "<answer>SELECT * FROM invalid_table_xyz WHERE id = 1;</answer>",

            # Cand 2: Malformed tags (missing </answer>), execution fails
            "<think>Check schema</think>\n<answer>SELECT broken query",

            # Cand 3: Unstructured text, no tags, execution fails
            "I think the answer is SELECT * FROM clients;"
        ]

        trainer = DenseGRPOTrainer(group_size=4)
        res = trainer.evaluate_group(cands, self.gold_sql, self.gold_think, self.db_path)

        # 1. Empirically verify all candidate executions failed
        for i, ev in enumerate(res["evaluations"]):
            self.assertEqual(ev["r_exec"], 0.0, f"Candidate {i} unexpectedly had non-zero execution reward!")

        # 2. Empirically verify rewards are not all identical
        raw_rewards = res["raw_rewards"]
        self.assertTrue(res["advantage_collapse_prevented"], "Advantage collapse was not prevented!")
        self.assertGreater(max(raw_rewards), min(raw_rewards))

        # 3. Verify normalized advantages produce clean positive and negative policy updates
        advantages = res["advantages"]
        self.assertGreater(max(advantages), 0.0)
        self.assertLess(min(advantages), 0.0)
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0, places=3)

    def test_hostile_all_identical_rollouts_division_by_zero_safety(self):
        """
        Hostile Scenario: All 4 rollouts are completely identical.
        Variance is 0.0. Verify that division-by-zero is avoided via epsilon,
        and advantages are cleanly zeroed without NaN.
        """
        identical_cands = [
            "<think>Step 1</think>\n<answer>SELECT client_id FROM clients;</answer>"
        ] * 4

        trainer = DenseGRPOTrainer(group_size=4)
        res = trainer.evaluate_group(identical_cands, self.gold_sql, self.gold_think, self.db_path)

        # All rewards are identical
        self.assertEqual(len(set(res["raw_rewards"])), 1)
        self.assertFalse(res["advantage_collapse_prevented"])

        # Test both default eps and explicit eps=1e-6
        for eps_val in [1e-8, 1e-6, 1e-4]:
            advs = normalize_group_advantages(res["raw_rewards"], eps=eps_val)
            self.assertEqual(advs, [0.0, 0.0, 0.0, 0.0])
            self.assertFalse(any(math.isnan(a) for a in advs))

    def test_destructive_sql_injection_defense(self):
        """
        Verify that destructive SQL statements (DROP, DELETE, UPDATE, ALTER, INSERT)
        are blocked by read-only mode and execute_sqlite_query, yielding R_exec == 0.0
        without mutating the underlying database.
        """
        hostile_queries = [
            "DROP TABLE clients;",
            "DELETE FROM clients WHERE district_id = 1;",
            "UPDATE clients SET name = 'HACKED';",
            "INSERT INTO clients (client_id, name) VALUES (999, 'Hacker');",
            "ALTER TABLE clients DROP COLUMN name;"
        ]

        for q in hostile_queries:
            r_exec = compute_exec_reward(q, self.gold_sql, self.db_path)
            self.assertEqual(r_exec, 0.0, f"Destructive query '{q}' should receive 0.0 exec reward!")

    def test_atomic_reward_edge_cases(self):
        """Test atomic reward under empty queries, corrupt whitespace, and complex clauses."""
        # 1. Empty strings
        self.assertEqual(compute_atomic_reward("", self.gold_sql), 0.0)
        self.assertEqual(compute_atomic_reward(self.gold_sql, ""), 0.0)

        # 2. Case and whitespace insensitivity
        lower_sql = self.gold_sql.lower()
        self.assertAlmostEqual(compute_atomic_reward(lower_sql, self.gold_sql), 1.0, places=4)

        # 3. Query sharing SELECT and FROM but differing in WHERE clause
        cand_partial = "SELECT client_id, name FROM clients WHERE district_id = 999;"
        r = compute_atomic_reward(cand_partial, self.gold_sql)
        self.assertTrue(0.0 < r < 1.0)


class TestDriveCheckpointCallbackStress(unittest.TestCase):
    """Adversarial stress testing on Checkpoint Pruning and Colab Crash Recovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.local_dir = os.path.join(self.temp_dir, "local_ssd")
        self.drive_dir = os.path.join(self.temp_dir, "google_drive")
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_colab_session_crash_recovery_from_drive(self):
        """
        Simulate Google Colab session crash:
        1. Checkpoints 100, 200, 300 synced to Google Drive.
        2. Local SSD crashes and is completely wiped.
        3. --resume_from_checkpoint latest recovers checkpoint-300 from Drive.
        4. Model files inside checkpoint are verified intact.
        """
        cb = DriveCheckpointCallback(
            drive_checkpoint_dir=self.drive_dir,
            local_checkpoint_dir=self.local_dir,
            max_drive_checkpoints=5
        )
        args = TrainingArguments(output_dir=self.local_dir)

        # Save checkpoints
        for step in [100, 200, 300]:
            p = os.path.join(self.local_dir, f"checkpoint-{step}")
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, "adapter_model.bin"), "w") as f:
                f.write(f"weights_{step}")
            with open(os.path.join(p, "adapter_config.json"), "w") as f:
                f.write(f'{{"step": {step}}}')
            cb.on_save(args, TrainerState(global_step=step), control=None)

        # SIMULATE SESSION CRASH: Wipe local storage completely
        shutil.rmtree(self.local_dir)
        os.makedirs(self.local_dir)
        self.assertEqual(len(os.listdir(self.local_dir)), 0)

        # Resume from checkpoint 'latest'
        recovered = get_resume_checkpoint("latest", local_dir=self.local_dir, drive_dir=self.drive_dir)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered, os.path.join(self.drive_dir, "checkpoint-300"))

        # Verify integrity of recovered files
        with open(os.path.join(recovered, "adapter_model.bin")) as f:
            self.assertEqual(f.read(), "weights_300")
        with open(os.path.join(recovered, "adapter_config.json")) as f:
            self.assertEqual(f.read(), '{"step": 300}')

    def test_checkpoint_pruning_with_auxiliary_directories(self):
        """
        Ensure pruning:
        - Keeps exactly max_drive_checkpoints newest checkpoints.
        - Preserves auxiliary folders (best_model/, logs/, custom_data/) without deleting them.
        """
        # Create non-checkpoint folders in Drive
        os.makedirs(os.path.join(self.drive_dir, "best_model"))
        os.makedirs(os.path.join(self.drive_dir, "runs_tensorboard"))
        with open(os.path.join(self.drive_dir, "best_model", "best.bin"), "w") as f:
            f.write("best weights")

        cb = DriveCheckpointCallback(
            drive_checkpoint_dir=self.drive_dir,
            local_checkpoint_dir=self.local_dir,
            max_drive_checkpoints=2
        )
        args = TrainingArguments(output_dir=self.local_dir)

        # Save checkpoints 10, 20, 30, 40
        for step in [10, 20, 30, 40]:
            p = os.path.join(self.local_dir, f"checkpoint-{step}")
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, "w.bin"), "w") as f:
                f.write("w")
            cb.on_save(args, TrainerState(global_step=step), control=None)

        drive_items = set(os.listdir(self.drive_dir))
        # Checkpoints 10 and 20 must be pruned; 30 and 40 must remain
        self.assertNotIn("checkpoint-10", drive_items)
        self.assertNotIn("checkpoint-20", drive_items)
        self.assertIn("checkpoint-30", drive_items)
        self.assertIn("checkpoint-40", drive_items)

        # Auxiliary folders must remain intact!
        self.assertIn("best_model", drive_items)
        self.assertIn("runs_tensorboard", drive_items)


class TestECECalibrationStress(unittest.TestCase):
    """Adversarial stress testing on Expected Calibration Error (ECE) and Temperature Scaling."""

    def test_perfect_and_worst_case_ece(self):
        """Boundary properties of ECE: perfect calibration (0.0) vs total miscalibration (1.0)."""
        # Perfect calibration: confidence 0.8 on 80% accuracy, confidence 0.2 on 20% accuracy
        confs_perfect = [0.8] * 10 + [0.2] * 10
        labels_perfect = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8
        ece_perfect = compute_ece(confs_perfect, labels_perfect)
        self.assertAlmostEqual(ece_perfect, 0.0, places=3)

        # Worst-case calibration: 100% confidence on all wrong decisions
        confs_worst = [0.99] * 20
        labels_worst = [0] * 20
        ece_worst = compute_ece(confs_worst, labels_worst)
        self.assertGreater(ece_worst, 0.95)

    def test_temperature_scaling_achieves_target_ece_on_moderate_overconfidence(self):
        """
        Verify Temperature Scaling achieves ECE <= 0.081 on realistic overconfident logits.
        """
        random.seed(42)
        N = 200
        logits = []
        labels = []
        for _ in range(N):
            y = 1 if random.random() > 0.5 else 0
            # 20% error rate with high logit magnitude (+/- 3.5)
            is_err = random.random() < 0.20
            pred_pos = (y == 1 and not is_err) or (y == 0 and is_err)
            z = 3.5 if pred_pos else -3.5
            logits.append(z)
            labels.append(y)

        raw_probs = calibrate_probabilities(logits, temperature=1.0)
        initial_ece = compute_ece(raw_probs, labels)
        self.assertGreater(initial_ece, 0.081, "Initial model must be uncalibrated")

        opt_t, cal_ece = fit_temperature(logits, labels, target_ece=0.081)
        self.assertLessEqual(cal_ece, 0.081, f"Calibrated ECE {cal_ece} exceeds 0.081 threshold!")

    def test_temperature_grid_search_bound_finding(self):
        """
        EMPIRICAL CHALLENGE FINDING:
        Demonstrate that when logits are severely overconfident (magnitude >= 5.0 with 25% errors),
        the hardcoded grid search temps=[0.2..3.0] in fit_temperature stops at T=3.0 and achieves
        ECE > 0.081 (0.0911). However, higher temperatures (e.g. T=5.0) easily achieve ECE <= 0.081 (0.0192).
        This test documents the empirical upper bound of the current implementation.
        """
        random.seed(42)
        N = 200
        logits = []
        labels = []
        for _ in range(N):
            y = 1 if random.random() > 0.5 else 0
            is_err = random.random() < 0.25
            pred_pos = (y == 1 and not is_err) or (y == 0 and is_err)
            z = 5.0 if pred_pos else -5.0
            logits.append(z)
            labels.append(y)

        # Current fit_temperature stops at T=3.0
        best_t, current_ece = fit_temperature(logits, labels, target_ece=0.081)

        # Prove empirically that T=5.0 achieves ECE <= 0.081
        probs_t5 = calibrate_probabilities(logits, temperature=5.0)
        ece_t5 = compute_ece(probs_t5, labels)

        self.assertLessEqual(ece_t5, 0.081, f"Temperature T=5.0 should achieve ECE <= 0.081, got {ece_t5}")
        self.assertLess(ece_t5, current_ece, "T=5.0 should produce lower ECE than T=3.0 on severe overconfidence")


if __name__ == "__main__":
    unittest.main()

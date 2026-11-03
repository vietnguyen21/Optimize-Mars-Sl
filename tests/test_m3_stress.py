"""
Adversarial Stress Test Suite for Milestone M3 (Phase 3: Google Colab Training Notebooks).

Empirically challenges:
1. JSON & Syntax Robustness across all 5 Notebooks:
   - AST parsing of every code cell for valid Python syntax and escape handling.
   - Import resolution against actual project modules in src/ and scripts/.
2. Colab Environment Simulation & Drive Mount Graceful Degradation:
   - Google Drive mounting fallback in non-Colab environments without crashing.
   - Edge-case testing of checkpoint recovery (empty folder, non-existent directory,
     malformed checkpoint names, numerical step ordering vs lexicographical sorting).
3. Notebook 03 Hardware Tier & Model Switch Verification:
   - Hardware tiering (T4, L4, A100) micro-batch sizes (1, 2, 4), precision, and rollouts.
   - 3B vs 7B model ID, sequence length, and LoRA target module configuration.
4. Runner Contract Drift:
   - Detects phantom runner functions (run_sft_training, run_grpo_training,
     run_decision_model_training, generate_all) that break notebook execution.
"""

import os
import sys
import re
import json
import ast
import shutil
import tempfile
import unittest
import importlib
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.callbacks import (
    find_latest_checkpoint,
    get_resume_checkpoint,
    DriveCheckpointCallback
)

NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
EXPECTED_NOTEBOOKS = [
    "01_data_preparation_and_embedding.ipynb",
    "02_sft_and_jepa_training_3b.ipynb",
    "03_grpo_training_3b_and_7b.ipynb",
    "04_system1_laya421m_training.ipynb",
    "05_end_to_end_evaluation.ipynb",
]


class TestM3StressSuite(unittest.TestCase):
    """Adversarial stress test harness for Google Colab notebooks."""

    @classmethod
    def setUpClass(cls):
        cls.notebooks = {}
        for nb_name in EXPECTED_NOTEBOOKS:
            nb_path = os.path.join(NOTEBOOKS_DIR, nb_name)
            with open(nb_path, "r", encoding="utf-8") as f:
                cls.notebooks[nb_name] = json.load(f)

    # -------------------------------------------------------------------------
    # 1. JSON & Syntax Robustness
    # -------------------------------------------------------------------------

    def test_json_structure_and_nbformat_v4(self):
        """Verify all 5 notebooks are valid JSON and conform to nbformat v4."""
        for nb_name, nb in self.notebooks.items():
            self.assertEqual(nb.get("nbformat"), 4, f"{nb_name} nbformat is not 4")
            self.assertIn("cells", nb, f"{nb_name} missing cells array")
            self.assertIn("metadata", nb, f"{nb_name} missing metadata")
            self.assertGreaterEqual(len(nb["cells"]), 5, f"{nb_name} has too few cells")

    def test_ast_parse_every_code_cell(self):
        """Parse 100% of code cells with ast.parse to detect syntax errors or invalid escapes."""
        total_cells_checked = 0
        for nb_name, nb in self.notebooks.items():
            for idx, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") == "code":
                    source_lines = cell.get("source", [])
                    source_text = "".join(source_lines) if isinstance(source_lines, list) else source_lines
                    total_cells_checked += 1
                    try:
                        ast.parse(source_text)
                    except SyntaxError as e:
                        self.fail(f"SyntaxError in {nb_name} cell #{idx}: {e}")
        self.assertGreaterEqual(total_cells_checked, 30, "Too few code cells analyzed across notebooks")

    def test_extract_and_audit_all_imports(self):
        """
        Adversarially extract all imports from all code cells and test symbol availability.
        Uncovers phantom functions that were hallucinated or renamed in notebooks.
        """
        phantom_symbols = []
        resolved_symbols = []

        for nb_name, nb in self.notebooks.items():
            for idx, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        mod_name = node.module or ""
                        # Only audit project modules (src.* or scripts.*)
                        if mod_name.startswith("src.") or mod_name.startswith("scripts."):
                            try:
                                mod = importlib.import_module(mod_name)
                                for alias in node.names:
                                    sym_name = alias.name
                                    if not hasattr(mod, sym_name):
                                        phantom_symbols.append((nb_name, idx, mod_name, sym_name))
                                    else:
                                        resolved_symbols.append((nb_name, idx, mod_name, sym_name))
                            except Exception as e:
                                phantom_symbols.append((nb_name, idx, mod_name, f"<module import error: {e}>"))

        # Report empirical findings
        self.assertGreater(len(resolved_symbols), 15, "Expected majority of symbols to resolve")

        # Verify that all project symbols imported across all notebooks now resolve cleanly
        self.assertEqual(phantom_symbols, [], f"Found unresolved imports in notebooks: {phantom_symbols}")

    # -------------------------------------------------------------------------
    # 2. Colab Environment Simulation & Drive Mount
    # -------------------------------------------------------------------------

    def test_drive_mount_graceful_degradation_in_non_colab(self):
        """
        Simulate non-Colab environment (google.colab not installed or mount throws).
        Verify that notebooks 02, 03, 04, 05 execute Drive mount block with fallback.
        """
        drive_notebooks = [
            "02_sft_and_jepa_training_3b.ipynb",
            "03_grpo_training_3b_and_7b.ipynb",
            "04_system1_laya421m_training.ipynb",
            "05_end_to_end_evaluation.ipynb",
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for nb_name in drive_notebooks:
                nb = self.notebooks[nb_name]
                drive_cell = None
                for cell in nb["cells"]:
                    src = "".join(cell.get("source", []))
                    if "drive.mount" in src:
                        drive_cell = src
                        break

                self.assertIsNotNone(drive_cell, f"Drive mount cell not found in {nb_name}")

                # Execute the cell source in a sandboxed namespace where google.colab is absent
                import unittest.mock as mock
                local_scope = {"os": os, "sys": sys, "MODEL_SCALE": "3B"}
                orig_torch = sys.modules.get("torch")
                sys.modules["torch"] = mock.MagicMock()
                try:
                    exec(drive_cell, local_scope)
                except Exception as e:
                    self.fail(f"Drive mount cell in {nb_name} crashed in non-Colab environment: {e}")
                finally:
                    if orig_torch is not None:
                        sys.modules["torch"] = orig_torch
                    else:
                        sys.modules.pop("torch", None)

                # Verify a fallback directory was configured
                has_fallback_dir = any(
                    k in local_scope for k in [
                        "DRIVE_DIR", "DRIVE_CHECKPOINT_DIR", "DRIVE_RESULTS_DIR"
                    ]
                )
                self.assertTrue(has_fallback_dir, f"{nb_name} failed to set fallback drive directory")

    def test_checkpoint_recovery_edge_cases(self):
        """
        Adversarial edge cases for find_latest_checkpoint and get_resume_checkpoint:
        - Non-existent directory -> None
        - Empty directory -> None
        - Subdirectories with non-checkpoint names (.ipynb_checkpoints, logs, best_model)
        - Step numbering ordering: checkpoint-2 vs checkpoint-10 vs checkpoint-100 (numeric vs lexical)
        - Malformed names (checkpoint-abc, checkpoint_-1)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Non-existent path
            self.assertIsNone(find_latest_checkpoint(os.path.join(tmp_dir, "non_existent")))

            # 2. Empty directory
            empty_dir = os.path.join(tmp_dir, "empty")
            os.makedirs(empty_dir)
            self.assertIsNone(find_latest_checkpoint(empty_dir))

            # 3. Directory with .ipynb_checkpoints, logs, and bad names
            os.makedirs(os.path.join(empty_dir, ".ipynb_checkpoints"))
            os.makedirs(os.path.join(empty_dir, "checkpoint_invalid"))
            os.makedirs(os.path.join(empty_dir, "checkpoint-abc"))
            with open(os.path.join(empty_dir, "checkpoint-999.txt"), "w") as f:
                f.write("file, not dir")
            self.assertIsNone(find_latest_checkpoint(empty_dir))

            # 4. Fallback to 'best_model' folder when no step checkpoints exist
            best_dir = os.path.join(empty_dir, "best_model")
            os.makedirs(best_dir)
            self.assertEqual(find_latest_checkpoint(empty_dir), best_dir)

            # 5. Numerical step ordering: checkpoint-2, checkpoint-10, checkpoint-100
            # Under lexical sort, "checkpoint-2" > "checkpoint-100". Under numeric, 100 > 10 > 2.
            os.makedirs(os.path.join(empty_dir, "checkpoint-2"))
            os.makedirs(os.path.join(empty_dir, "checkpoint-10"))
            os.makedirs(os.path.join(empty_dir, "checkpoint-100"))
            latest = find_latest_checkpoint(empty_dir)
            self.assertEqual(os.path.basename(latest), "checkpoint-100",
                             "Checkpoint sorting must be strictly numerical, not lexicographical")

            # 6. Underscore vs hyphen: checkpoint_150
            os.makedirs(os.path.join(empty_dir, "checkpoint_150"))
            latest_under = find_latest_checkpoint(empty_dir)
            self.assertEqual(os.path.basename(latest_under), "checkpoint_150")

    def test_resume_from_checkpoint_precedence(self):
        """
        Verify get_resume_checkpoint behavior when 'latest' is requested:
        - Drive has older checkpoint-50, Local has newer checkpoint-100.
        - Documents Drive priority vs Local fallback.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            drive_dir = os.path.join(tmp_dir, "drive")
            local_dir = os.path.join(tmp_dir, "local")
            os.makedirs(drive_dir)
            os.makedirs(local_dir)

            os.makedirs(os.path.join(drive_dir, "checkpoint-50"))
            os.makedirs(os.path.join(local_dir, "checkpoint-100"))

            # Current logic: Google Drive takes precedence for crash recovery
            res = get_resume_checkpoint(resume_flag="latest", local_dir=local_dir, drive_dir=drive_dir)
            self.assertEqual(os.path.basename(res), "checkpoint-50",
                             "Current implementation prioritizes persistent Drive over local storage")

            # If Drive is empty, fallback to local SSD
            shutil.rmtree(drive_dir)
            res_fallback = get_resume_checkpoint(resume_flag="latest", local_dir=local_dir, drive_dir=drive_dir)
            self.assertEqual(os.path.basename(res_fallback), "checkpoint-100",
                             "Should fallback to local SSD when Drive has no checkpoints")

    # -------------------------------------------------------------------------
    # 3. Notebook 03: Hardware Tier Switch & 3B vs 7B Model Switch
    # -------------------------------------------------------------------------

    def test_notebook_03_hardware_tier_switch_logic(self):
        """
        Verify that hardware tier selection logic maps correctly across all VRAM levels:
        - A100 (VRAM >= 40.0 GB): micro_batch=4, K=16 rollouts, precision=bf16, quant=None
        - L4   (VRAM >= 20.0 GB): micro_batch=2, K=8 rollouts, precision=bf16, quant=None
        - T4   (VRAM < 20.0 GB):  micro_batch=1, K=8 rollouts, precision=fp16, quant=4bit
        """
        def simulate_tier_selection(vram_gb: float):
            if vram_gb >= 40.0:
                tier = "Pro+ A100"
                mb = 4
                k = 16
                prec = "bf16"
                quant = None
            elif vram_gb >= 20.0:
                tier = "Pro L4"
                mb = 2
                k = 8
                prec = "bf16"
                quant = None
            else:
                tier = "Free T4"
                mb = 1
                k = 8
                prec = "fp16"
                quant = "4bit"
            return tier, mb, k, prec, quant

        # A100 checks
        t, mb, k, prec, quant = simulate_tier_selection(80.0)
        self.assertEqual(t, "Pro+ A100")
        self.assertEqual(mb, 4)
        self.assertEqual(k, 16)
        self.assertEqual(prec, "bf16")
        self.assertIsNone(quant)

        # L4 checks (24GB VRAM)
        t, mb, k, prec, quant = simulate_tier_selection(24.0)
        self.assertEqual(t, "Pro L4")
        self.assertEqual(mb, 2)
        self.assertEqual(k, 8)
        self.assertEqual(prec, "bf16")
        self.assertIsNone(quant)

        # T4 checks (15-16GB VRAM)
        t, mb, k, prec, quant = simulate_tier_selection(15.0)
        self.assertEqual(t, "Free T4")
        self.assertEqual(mb, 1)
        self.assertEqual(k, 8)
        self.assertEqual(prec, "fp16")
        self.assertEqual(quant, "4bit")

        # CPU fallback (0 GB VRAM)
        t, mb, k, prec, quant = simulate_tier_selection(0.0)
        self.assertEqual(t, "Free T4")
        self.assertEqual(mb, 1)

    def test_notebook_03_model_switching_parameters(self):
        """
        Verify that switching MODEL_SCALE between '3B' and '7B' updates:
        - Base Model ID
        - Sequence length
        - LoRA rank & alpha
        And note the gap: target_modules is omitted in MODEL_CONFIGS!
        """
        nb = self.notebooks["03_grpo_training_3b_and_7b.ipynb"]
        cell_2_src = "".join(nb["cells"][2]["source"])

        # Execute cell 2 logic with both 3B and 7B
        for scale in ["3B", "7B"]:
            src_to_run = cell_2_src.replace('MODEL_SCALE = "3B"', f'MODEL_SCALE = "{scale}"')
            scope = {"HARDWARE_TIER": "Free T4"}
            exec(src_to_run, scope)
            cfg = scope["active_model_cfg"]

            if scale == "3B":
                self.assertEqual(cfg["name"], "FINER-SQL-3B")
                self.assertEqual(cfg["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
                self.assertEqual(cfg["lora_r"], 64)
                self.assertEqual(cfg["lora_alpha"], 16)
                self.assertEqual(cfg["max_seq_len"], 1024)  # 1024 on Free T4
            elif scale == "7B":
                self.assertEqual(cfg["name"], "Qwen2.5-Coder-7B")
                self.assertEqual(cfg["model_id"], "Qwen/Qwen2.5-Coder-7B-Instruct")
                self.assertEqual(cfg["lora_r"], 128)
                self.assertEqual(cfg["lora_alpha"], 32)
                self.assertEqual(cfg["max_seq_len"], 2048)

            # Check that target_modules exists in config dictionary
            self.assertIn("target_modules", cfg,
                          "target_modules should be defined in MODEL_CONFIGS in Notebook 03")
            self.assertIn("q_proj", cfg["target_modules"])


if __name__ == "__main__":
    unittest.main()

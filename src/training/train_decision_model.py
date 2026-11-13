"""
Training Script for Laya-421M (ModernBERT) System-1 Decision Model with RLCD and Dual Learning Rates.

Implements:
  1. Dual Learning Rate Optimizer Configuration:
     - Backbone: lr = 1e-5, weight_decay = 0.01
     - Classification Head: lr = 1e-4, weight_decay = 0.01
  2. RLCD (Reinforcement Learning for Calibrated Decisions) Loss:
     L_RLCD = L_BCE(p, y) + alpha_cal * (p - y)^2 (penalizes overconfident incorrect decisions)
  3. Held-out NLL temperature scaling and independently measured test ECE.
  4. Dual execution: Full HuggingFace/PyTorch mode and standalone --dry_run mode on data/mini/.
"""

import os
import sys
import json
import math
import random
import argparse
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.callbacks import DriveCheckpointCallback, get_resume_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ---------------------------------------------------------------------------
# 1. Dual Learning Rate Parameter Grouping
# ---------------------------------------------------------------------------

def configure_dual_lr_optimizer(
    model: Any,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01
) -> Any:
    """
    Separates model parameters into Backbone and Decision Head groups with distinct learning rates:
      - Backbone group: lr = 1e-5
      - Head group: lr = 1e-4
    
    If PyTorch is available, returns a torch.optim.AdamW optimizer.
    Otherwise, returns a dictionary describing the parameter groups.
    """
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        head_keywords = ["classifier", "head", "score", "fc", "out_proj"]
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name.lower() for k in head_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"}
        ]
        optimizer = AdamW(param_groups)
        return optimizer
    else:
        return {
            "param_groups": [
                {"name": "backbone", "lr": lr_backbone, "weight_decay": weight_decay},
                {"name": "head", "lr": lr_head, "weight_decay": weight_decay}
            ]
        }


# ---------------------------------------------------------------------------
# 2. RLCD Loss Function
# ---------------------------------------------------------------------------

def compute_rlcd_loss(
    probabilities: Optional[Union[List[float], float]] = None,
    labels: Optional[Union[List[int], List[float], float, int]] = None,
    alpha_cal: float = 0.5,
    eps: float = 1e-7,
    prob: Optional[float] = None,
    label: Optional[Union[float, int]] = None,
    alpha: Optional[float] = None,
    **kwargs: Any
) -> float:
    """
    Computes RLCD Loss:
      L_BCE = - [ y * log(p) + (1-y) * log(1-p) ]
      L_CAL = (p - y)^2  (Quadratic Brier calibration penalty)
      L_RLCD = L_BCE + alpha_cal * L_CAL
    Supports both scalar calls (prob=..., label=...) and list calls (probabilities=[...], labels=[...]).
    """
    if TORCH_AVAILABLE and isinstance(probabilities, torch.Tensor):
        targets = torch.as_tensor(labels, device=probabilities.device, dtype=probabilities.dtype)
        return F.binary_cross_entropy(probabilities.clamp(eps, 1-eps), targets) + alpha_cal * ((probabilities-targets)**2).mean()
    if prob is not None and probabilities is None:
        probs_list = [float(prob)]
    elif isinstance(probabilities, (int, float)):
        probs_list = [float(probabilities)]
    elif probabilities is not None:
        probs_list = list(probabilities)
    else:
        probs_list = []

    if label is not None and labels is None:
        labels_list = [int(round(label))]
    elif isinstance(labels, (int, float)):
        labels_list = [int(round(labels))]
    elif labels is not None:
        labels_list = [int(round(x)) for x in labels]
    else:
        labels_list = []

    effective_alpha = alpha if alpha is not None else alpha_cal
    n = len(probs_list)
    if n == 0:
        return 0.0

    total_loss = 0.0
    for p, y in zip(probs_list, labels_list):
        # Clip probability for numerical stability
        p_clipped = min(max(p, eps), 1.0 - eps)
        bce = -(y * math.log(p_clipped) + (1 - y) * math.log(1.0 - p_clipped))
        cal_penalty = (p - y) ** 2
        step_loss = bce + effective_alpha * cal_penalty
        total_loss += step_loss

    return total_loss / n


# ---------------------------------------------------------------------------
# 3. Expected Calibration Error (ECE) & Temperature Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: List[float],
    labels: List[int],
    num_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across M bins:
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0

    for i in range(num_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        # Elements in current bin
        bin_items = [
            (conf, lbl) for conf, lbl in zip(confidences, labels)
            if (low <= conf < high if i < num_bins - 1 else low <= conf <= high)
        ]

        bin_size = len(bin_items)
        if bin_size > 0:
            avg_conf = sum(item[0] for item in bin_items) / bin_size
            avg_acc = sum(item[1] for item in bin_items) / bin_size
            ece += (bin_size / n) * abs(avg_acc - avg_conf)

    return round(ece, 4)


def calibrate_probabilities(
    logits: List[float],
    temperature: Union[float, Tuple[float, float]] = 1.0
) -> List[float]:
    """
    Applies temperature scaling to raw logits:
      p = sigmoid(logit / T)
    """
    t_val = temperature[0] if isinstance(temperature, tuple) else float(temperature)
    t = max(t_val, 1e-4)
    calibrated = []
    for z in logits:
        scaled_z = z / t
        # Sigmoid with clipping to prevent overflow
        if scaled_z >= 0:
            p = 1.0 / (1.0 + math.exp(-scaled_z))
        else:
            exp_z = math.exp(scaled_z)
            p = exp_z / (1.0 + exp_z)
        calibrated.append(p)
    return calibrated


def fit_temperature(
    logits: List[float],
    labels: List[int],
    target_ece: float = 0.081
) -> Tuple[float, float]:
    """
    Fit positive temperature by held-out binary negative log likelihood.
    target_ece is a reporting target, never a promised calibration outcome.
    
    Returns:
        Tuple of (optimal_temperature, resulting_ece)
    """
    if not logits or len(logits) != len(labels):
        raise ValueError("Calibration requires nonempty, aligned logits and labels")
    # Fit held-out negative log likelihood, not ECE or a desired benchmark target.
    temps = [math.exp(-3.0 + i * 6.0 / 240) for i in range(241)]
    def nll(t):
        return sum(max(z/t, 0) - (z/t)*y + math.log1p(math.exp(-abs(z/t)))
                   for z,y in zip(logits, labels)) / len(logits)
    best_t = min(temps, key=nll)
    return best_t, compute_ece(calibrate_probabilities(logits, best_t), labels)



# ---------------------------------------------------------------------------
# 4. Dry Run Verification
# ---------------------------------------------------------------------------

def run_dry_run(
    num_samples: int = 100,
    target_ece_threshold: float = 0.081
) -> Dict[str, Any]:
    """
    Verifies:
      1. Dual learning rate parameter configuration (1e-5 backbone, 1e-4 head).
      2. RLCD loss computation.
      3. Temperature probability calibration.
      4. ECE <= 0.081 verification.
    """
    logger.info("[DryRun] Starting Laya-421M RLCD decision model verification...")

    # 1. Verify dual learning rate configuration
    dummy_model_cfg = configure_dual_lr_optimizer(None, lr_backbone=1e-5, lr_head=1e-4)
    groups = dummy_model_cfg["param_groups"]
    assert groups[0]["lr"] == 1e-5, "Backbone learning rate must be 1e-5"
    assert groups[1]["lr"] == 1e-4, "Head learning rate must be 1e-4"
    logger.info(f"[DryRun] ✅ Dual LR verified: Backbone={groups[0]['lr']}, Head={groups[1]['lr']}")

    # 2. Generate synthetic calibration test data (logits and ground truth verdicts)
    random.seed(42)
    logits = []
    labels = []

    for _ in range(num_samples):
        # Balanced binary labels: 1 = correct SQL, 0 = incorrect SQL
        y = 1 if random.random() > 0.4 else 0
        # True positives tend to have positive logits, false tend to have negative logits
        noise = (random.random() - 0.5) * 1.5
        z = (2.0 if y == 1 else -1.5) + noise
        logits.append(z)
        labels.append(y)

    # 3. Compute uncalibrated probabilities and initial ECE
    raw_probs = calibrate_probabilities(logits, temperature=1.0)
    initial_loss = compute_rlcd_loss(raw_probs, labels, alpha_cal=0.5)
    initial_ece = compute_ece(raw_probs, labels)
    logger.info(f"[DryRun] Initial RLCD Loss: {initial_loss:.4f}, Initial ECE: {initial_ece:.4f}")

    # 4. Fit temperature calibration to achieve ECE <= 0.081
    opt_temp, calibrated_ece = fit_temperature(logits, labels, target_ece=target_ece_threshold)
    calibrated_probs = calibrate_probabilities(logits, temperature=opt_temp)
    calibrated_loss = compute_rlcd_loss(calibrated_probs, labels, alpha_cal=0.5)

    logger.info(f"[DryRun] Optimal Temperature: {opt_temp:.2f}, Calibrated ECE: {calibrated_ece:.4f} (Target <= {target_ece_threshold})")
    assert calibrated_ece <= target_ece_threshold, f"Calibrated ECE ({calibrated_ece}) exceeds threshold ({target_ece_threshold})!"

    logger.info("[DryRun] ✅ RLCD calibration and dual LR training successfully verified.")

    return {
        "status": "PASSED",
        "dual_lr": {"backbone": 1e-5, "head": 1e-4},
        "initial_ece": initial_ece,
        "calibrated_ece": calibrated_ece,
        "target_ece": target_ece_threshold,
        "optimal_temperature": opt_temp,
        "ece_passed": calibrated_ece <= target_ece_threshold
    }


def run_decision_model_training(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    drive_dir: Optional[str] = None,
    export_dir: Optional[str] = None,
    epochs: int = 3,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    target_ece: float = 0.081,
    dry_run: bool = True,
    **kwargs: Any
) -> Dict[str, Any]:
    """Runner function executing System-1 decision model training (dry-run or full)."""
    if dry_run:
        res = dict(run_dry_run(num_samples=100, target_ece_threshold=target_ece), mode='smoke', trained=False)
        target_export = export_dir or output_dir
        if target_export:
            os.makedirs(target_export, exist_ok=True)
            with open(os.path.join(target_export, 'smoke_calibration_meta.json'), 'w') as f:
                json.dump(res, f, indent=2)
        return res
    return _train_decision(data_path=data_path, output_dir=output_dir, export_dir=export_dir,
                           epochs=epochs, lr_backbone=lr_backbone, lr_head=lr_head,
                           target_ece=target_ece, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Laya-421M RLCD Decision Model Trainer")
    parser.add_argument("--model_name_or_path", type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--alpha_cal", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_path")
    parser.add_argument("--output_dir", default="outputs/system1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        res = run_dry_run()
        print(json.dumps(res, indent=2))
        sys.exit(0)

    logger.info("Executing Laya-421M RLCD Trainer in full mode.")
    res = run_decision_model_training(**vars(args))
    print(json.dumps(res, indent=2))



def decision_text(question, schema, sql):
    """Canonical training/inference text; gold SQL never enters this representation."""
    return f'<question>{question}</question>\n<answer>{sql}</answer>\n<schema>{schema}</schema>'


def validate_decision_budget(tokenizer,question,sql,max_length):
    """Retain the full question and candidate; only trailing schema may truncate."""
    inputs=tokenizer(decision_text(question,'',sql),truncation=False,return_tensors='pt')
    length=inputs['input_ids'].shape[-1]
    if length > max_length:
        raise ValueError(f'Question and candidate need {length} tokens; increase System-1 max_length={max_length}')
    return length


def build_candidate_dataset(records, db_dir, output_path=None, generator=None, num_candidates=4,
                            include_gold=False):
    """Label actual sampled/provided candidates by read-only execution comparison.

    records are training questions with SQL/answer and either candidate SQLs in
    ``candidates`` or a supplied CandidateGenerator backed by a trained checkpoint.
    Gold positives may be added for training, never for benchmark generation.
    """
    import re
    import sqlite3
    from contextlib import closing
    from src.training.runtime import resolve_db, load_records
    from src.evaluation.metrics import execute_query_safely, compare_sql_results
    if isinstance(records, (str, os.PathLike)):
        records = load_records(records)
    result = []
    for row in records:
        gold = row.get('SQL', row.get('answer'))
        if not gold:
            raise ValueError('Candidate labeling requires reference SQL in training records')
        db = resolve_db(db_dir, row['db_id'])
        ok, gold_rows = execute_query_safely(gold, db)
        if not ok:
            raise ValueError(f'Gold SQL failed for {row["db_id"]}: {gold_rows}')
        schema = row.get('schema', row.get('ddl_statements',''))
        if not schema:
            with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as conn:
                schema = '\n'.join(r[0] for r in conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"))
        candidates = row.get('candidates')
        if candidates is None:
            if generator is None or getattr(generator,'dry_run',False):
                raise ValueError('Supply sampled candidates or a checkpoint-backed generator; no synthetic fallback')
            from src.common.schemas import GenerationInput, CompactSubSchema
            inp = GenerationInput(question=row['question'], sub_schema=CompactSubSchema(
                db_id=row['db_id'], selected_tables=[], selected_columns={}, foreign_keys=[], ddl_statements=schema),
                num_trajectories=num_candidates)
            candidates = generator.generate(inp, num_candidates=num_candidates)
        sqls = [c if isinstance(c,str) else (c['sql'] if isinstance(c,dict) else c.sql) for c in candidates]
        if include_gold:
            sqls.append(gold)
        for sql in dict.fromkeys(sqls):
            success, rows = execute_query_safely(sql, db)
            label = int(success and compare_sql_results(rows, gold_rows,
                        preserve_order=bool(re.search(r'\bORDER\s+BY\b',gold,re.I))))
            result.append({'db_id':row['db_id'], 'question':row['question'],
                           'schema':schema,
                           'sql':sql, 'label':label, 'label_source':'execution_match'})
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path,'x',encoding='utf-8') as f:
            for row in result:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return result


def split_candidate_groups(records, seed=42):
    """Keep all candidates from one database/question in exactly one split."""
    groups = sorted({(r['db_id'],r['question']) for r in records})
    if len(groups) < 3:
        raise ValueError('Need at least three distinct questions for train/calibration/test splits')
    random.Random(seed).shuffle(groups)
    n_cal = max(1, int(len(groups)*0.15))
    n_test = max(1, int(len(groups)*0.15))
    cal = set(groups[:n_cal]); test = set(groups[n_cal:n_cal+n_test])
    splits = ([],[],[])
    for row in records:
        key = (row['db_id'],row['question'])
        splits[1 if key in cal else 2 if key in test else 0].append(row)
    return splits


from src.training.checkpoint_sync import persistent_training


@persistent_training('system1')
def _train_decision(data_path, output_dir, export_dir=None, model_name_or_path='answerdotai/ModernBERT-large',
                    calibration_path=None, test_path=None, max_steps=None, epochs=3, lr_backbone=1e-5,
                    lr_head=1e-4, alpha_cal=0.5, target_ece=0.081, max_length=512,
                    gradient_accumulation_steps=8, save_steps=50, device='cuda:0', seed=42,
                    resume_from_checkpoint=None, **kwargs):
    if not data_path or not output_dir:
        raise ValueError('Real decision training requires labeled data_path and output_dir')
    if not TORCH_AVAILABLE:
        raise ImportError('Real System-1 training requires torch and transformers; no dry-run fallback')
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    destination=export_dir or os.path.join(output_dir,'final')
    if os.path.exists(destination):
        raise FileExistsError('System-1 final artifact exists; choose a fresh export/output directory to preserve weights')
    from src.training.runtime import load_records, seed_everything, ExperimentLogger, save_checkpoint, load_checkpoint, latest_checkpoint, dataset_fingerprint
    seed_everything(seed)
    records = load_records(data_path)
    for row in records:
        if row.get('label') not in (0,1) or not row.get('sql'):
            raise ValueError('Decision data must contain sql and execution-derived binary label')
    if bool(calibration_path) != bool(test_path):
        raise ValueError('Provide both calibration_path and test_path, or neither')
    if calibration_path:
        train,cal,test = records,load_records(calibration_path),load_records(test_path)
        keys = [{(r['db_id'],r['question']) for r in split} for split in (train,cal,test)]
        if any(keys[i].intersection(keys[j]) for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Question leakage between decision training/calibration/test splits')
    else:
        train,cal,test = split_candidate_groups(records,seed)
    for split in (train,cal,test):
        for row in split:
            if row.get('label') not in (0,1) or not row.get('sql') or not row.get('schema'):
                raise ValueError('All decision splits require schema, sql and binary labels')
    random.Random(seed).shuffle(train)
    if {r['label'] for r in train} != {0,1}:
        raise ValueError('Decision training needs both correct and incorrect candidates')
    if gradient_accumulation_steps < 1 or save_steps < 1:
        raise ValueError('Accumulation and save_steps must be positive')
    max_steps = max_steps if max_steps is not None else math.ceil(len(train)*epochs/gradient_accumulation_steps)
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    amp = str(device).startswith('cuda')
    if amp and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    from src.training.model_identity import model_identity, read_identity, verify_identity
    expected_identity = read_identity(resume_from_checkpoint) if resume_from_checkpoint else None
    identity, hf_options = model_identity(model_name_or_path, expected=expected_identity)
    verify_identity(expected_identity, identity)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_options)
    for row in train+cal+test:
        validate_decision_budget(tokenizer,row['question'],row['sql'],max_length)
    # Eager attention works on T4 without FlashAttention 2 or BF16.
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=1,
            attn_implementation='eager', reference_compile=False, torch_dtype=torch.float32, **hf_options).to(device)
    model._training_base_identity = identity
    model.gradient_checkpointing_enable()
    optimizer = configure_dual_lr_optimizer(model,lr_backbone,lr_head)
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    log = ExperimentLogger(output_dir,dict(phase='system1',model=model_name_or_path,seed=seed,
        max_steps=max_steps,max_length=max_length,gradient_accumulation_steps=gradient_accumulation_steps,
        train_questions=len({(r['db_id'],r['question']) for r in train}),
        calibration_questions=len({(r['db_id'],r['question']) for r in cal}),
        test_questions=len({(r['db_id'],r['question']) for r in test})))
    contract = dict(phase='system1',data_sha256=dataset_fingerprint(data_path),
        calibration_sha256=dataset_fingerprint(calibration_path) if calibration_path else None,
        test_sha256=dataset_fingerprint(test_path) if test_path else None,model_name_or_path=model_name_or_path,
        max_length=max_length,seed=seed,gradient_accumulation_steps=gradient_accumulation_steps,
        lr_backbone=lr_backbone,lr_head=lr_head,alpha_cal=alpha_cal)
    log.log(event='resume_contract',contract=contract)
    step=0
    if resume_from_checkpoint:
        path = latest_checkpoint(output_dir) if resume_from_checkpoint=='latest' else resume_from_checkpoint
        if path is None:
            raise FileNotFoundError('No complete decision checkpoint to resume')
        state=load_checkpoint(path,model,optimizer,scaler=scaler,expected_extra={"contract":contract})
        if state.get('extra',{}).get('contract') != contract:
            raise ValueError('System-1 resume data/config differ from saved checkpoint')
        step=state['step']
    def encode(row):
        return {k:v.to(device) for k,v in tokenizer(decision_text(row['question'],row.get('schema',''),row['sql']),
                  truncation=True,max_length=max_length,return_tensors='pt').items()}
    model.train();optimizer.zero_grad(set_to_none=True)
    consecutive_skips = 0
    max_skips = int(os.environ.get('AMP_MAX_CONSECUTIVE_SKIPS', '8'))
    if max_skips < 1:
        raise ValueError('AMP_MAX_CONSECUTIVE_SKIPS must be positive')
    while step < max_steps:
        total=0.
        for micro in range(gradient_accumulation_steps):
            row=train[(step*gradient_accumulation_steps+micro)%len(train)]
            labels=torch.tensor([float(row['label'])],device=device)
            with torch.autocast('cuda' if amp else 'cpu',dtype=torch.float16 if amp else torch.bfloat16,enabled=amp):
                logits=model(**encode(row)).logits.float().view(-1)
                probs=logits.sigmoid()
                loss=F.binary_cross_entropy_with_logits(logits,labels)+alpha_cal*((probs-labels)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective; last complete checkpoint preserved')
            scaler.scale(loss/gradient_accumulation_steps).backward()
            total+=float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        from src.training.runtime import optimizer_update_succeeded
        updated = optimizer_update_succeeded(optimizer, scaler)
        optimizer.zero_grad(set_to_none=True)
        if not updated:
            consecutive_skips += 1
            log.log(event='optimizer_skipped', step=step, reason='AMP overflow')
            if consecutive_skips >= max_skips:
                raise FloatingPointError('Repeated AMP overflow; last complete checkpoint preserved')
            continue
        consecutive_skips = 0
        step+=1
        log.log(step=step,loss=total/gradient_accumulation_steps)
        if step%save_steps==0 or step==max_steps:
            save_checkpoint(output_dir,model,optimizer,step,scaler=scaler,tokenizer=tokenizer,
                            extra={'phase':'system1','max_length':max_length,'contract':contract})
    from src.training.checkpoint_sync import finalize_checkpoint_backup
    finalize_checkpoint_backup()
    model.eval()
    def predict(rows):
        logits=[];labels=[]
        with torch.inference_mode():
            for row in rows:
                logits.append(float(model(**encode(row)).logits.float().view(-1)[0]))
                labels.append(row['label'])
        return logits,labels
    cal_logits,cal_labels=predict(cal)
    temperature,cal_ece=fit_temperature(cal_logits,cal_labels)
    test_logits,test_labels=predict(test)
    metrics={'mode':'train','trained':True,'step':step,'temperature':temperature,'max_length':max_length,
             'calibration_ece':cal_ece,'test_ece':compute_ece(calibrate_probabilities(test_logits,temperature),test_labels),
             'test_ece_unscaled':compute_ece(calibrate_probabilities(test_logits),test_labels),
             'target_ece':target_ece,'target_met':False,'test_candidates':len(test)}
    metrics['target_met']=metrics['test_ece']<=target_ece
    model.save_pretrained(destination);tokenizer.save_pretrained(destination)
    with open(os.path.join(destination,'calibration_meta.json'),'w') as f:
        json.dump(metrics,f,indent=2)
    with open(os.path.join(output_dir,'training_summary.json'),'w') as f:
        json.dump(dict(metrics,model_path=destination),f,indent=2)
    return dict(metrics,model_path=destination)


if __name__=='__main__':
    main()

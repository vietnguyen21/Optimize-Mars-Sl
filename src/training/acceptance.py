"""Isolated acceptance fixtures. No imports of ML packages until explicitly run.

The fixture is infrastructure evidence, never a pretrained-component benchmark.
Hub writes require a CLI opt-in and use a reserved, random acceptance namespace.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
STAGE = 'acceptance_fixture_v1'
MAX_UPLOAD_BYTES = 5 * 1024**2
COMPONENTS = ('grounding', 'sft_jepa_3b', 'grpo_3b', 'system1', 'evaluation')


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class AcceptanceFailure(RuntimeError):
    """Only application-authored diagnostics may be included in the report."""


def tree_digest(value):
    """Canonical digest of tensor state, including optimizer moments and RNG."""
    import numpy as np
    import torch
    def encode(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return ['tensor', str(item.dtype), list(item.shape), hashlib.sha256(data).hexdigest()]
        if isinstance(item, np.ndarray):
            return ['numpy', item.dtype.str, list(item.shape), hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, dict):
            return ['dict', [[repr(k), encode(v)] for k, v in sorted(item.items(), key=lambda pair:repr(pair[0]))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [encode(v) for v in item]]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture_rng(device):
    import random
    import numpy as np
    import torch
    cuda = torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    return dict(rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(), rng_cuda=cuda,
                cuda_training_device=torch.device(device).index if cuda else None)


def reset_rng(state, device):
    import random
    import numpy as np
    import torch
    from src.training.runtime import restore_cuda_rng
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    restore_cuda_rng(state, device)


def rng_probe(device):
    import random
    import numpy as np
    import torch
    return dict(python=random.random(), numpy=float(np.random.random()),
                torch_cpu=torch.rand(3).tolist(),
                torch_device=torch.rand(3, device=device).cpu().tolist())


def fixture_contract(device, gpu, component='sft_jepa_3b'):
    from src.training.checkpoint_sync import fingerprint, digest
    import torch
    files = ('acceptance.py', 'runtime.py', 'checkpoint_sync.py')
    return dict(fixture_version=1, dataset_fingerprint=fingerprint(list(range(32))),
                seed=1729, batch_size=2, dataset_size=8, gpu=gpu, component=component,
                torch_build=str(torch.__version__), cuda_build=torch.version.cuda,
                amp=str(device).startswith('cuda'), python=list(sys.version_info[:2]),
                torch=metadata.version('torch'), numpy=metadata.version('numpy'),
                source_fingerprint=fingerprint({name:digest(Path(__file__).parent/name) for name in files}))


def make_bundle(device):
    import torch
    from src.training.runtime import seed_everything
    seed_everything(1729)
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                               torch.nn.Dropout(0.25), torch.nn.Linear(8, 2))
            self.reference = torch.nn.Linear(4, 2)
            self.reference.requires_grad_(False)
    model = Fixture()
    model._training_base_identity = {'kind':'local', 'files':{'fixture-initial-weights':tree_digest(model.state_dict())}}
    model = model.to(device)
    predictor = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]
                                 + list(predictor.parameters()), lr=0.003, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda'))
    return dict(model=model, predictor=predictor, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, step=0, cursor=0)


def train_fixture_step(bundle, device):
    """One tiny synthetic optimizer update; exercises all RNGs and a frozen reference."""
    import torch
    from src.training.runtime import optimizer_update_succeeded
    data = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    for _ in range(8):  # AMP overflow cannot create an unlimited retry loop.
        probe = rng_probe(device)
        noise = (probe['python'] + probe['numpy'] + probe['torch_cpu'][0] + probe['torch_device'][0]) / 100
        indices = [(bundle['cursor'] + offset) % 8 for offset in range(2)]
        x = data[indices].to(device) + noise
        bundle['model'].train()
        with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu',
                            dtype=torch.float16, enabled=str(device).startswith('cuda')):
            output = bundle['model'].policy(x)
            target = bundle['model'].reference(x).detach()
            prediction = bundle['predictor'](output)
            loss = (output.float() - target.float()).square().mean() + (prediction.float() - target.float()).square().mean()
        if not torch.isfinite(loss):
            raise AcceptanceFailure('fixture_nonfinite_loss')
        bundle['scaler'].scale(loss).backward()
        bundle['scaler'].unscale_(bundle['optimizer'])
        updated = optimizer_update_succeeded(bundle['optimizer'], bundle['scaler'])
        bundle['optimizer'].zero_grad(set_to_none=True)
        if updated:
            bundle['scheduler'].step()
            bundle['step'] += 1
            bundle['cursor'] = (bundle['cursor'] + 2) % 8
            return
    raise AcceptanceFailure('fixture_repeated_amp_overflow')


def bundle_proof(bundle, device):
    import torch
    rng = capture_rng(device)
    active = torch.device(device).index or 0
    # Only the selected GPU RNG participates in this single-device fixture.
    # This allows 2xT4 -> 1xT4 mapping; different GPU architectures use separate runs.
    rng['rng_cuda'] = rng['rng_cuda'][active] if rng['rng_cuda'] else None
    rng.pop('cuda_training_device')
    fields = dict(model_weights=bundle['model'].state_dict(),
                  predictor_weights=bundle['predictor'].state_dict(),
                  frozen_reference=bundle['model'].reference.state_dict(),
                  optimizer=bundle['optimizer'].state_dict(), scheduler=bundle['scheduler'].state_dict(),
                  scaler=bundle['scaler'].state_dict(), rng=rng)
    moments = bundle['optimizer'].state_dict()['state']
    optimizer_steps = sorted({int(value['step']) for value in moments.values()})
    return dict(step=bundle['step'], next_data_cursor=bundle['cursor'],
                optimizer_steps=optimizer_steps, scheduler_last_epoch=bundle['scheduler'].last_epoch,
                hashes={key:tree_digest(value) for key,value in fields.items()})


def assert_proof(actual, expected, label):
    for key in ('step', 'next_data_cursor', 'optimizer_steps', 'scheduler_last_epoch'):
        if actual[key] != expected[key]:
            raise AcceptanceFailure(f'{label}_{key}_mismatch')
    for key, digest in expected['hashes'].items():
        if actual['hashes'][key] != digest:
            raise AcceptanceFailure(f'{label}_{key}_mismatch')


def save_fixture(bundle, manager, device, contract):
    """Store golden continuation digests inside the original immutable checkpoint."""
    from src.training.checkpoint_sync import ACTIVE_SYNC
    from src.training.runtime import save_checkpoint
    before_rng = capture_rng(device)
    saved_proof = bundle_proof(bundle, device)
    next_probe = rng_probe(device)
    reset_rng(before_rng, device)
    twin = copy.deepcopy(bundle)
    train_fixture_step(twin, device)
    golden_next = bundle_proof(twin, device)
    del twin
    reset_rng(before_rng, device)
    extra = dict(contract=contract, dataset_fingerprint=contract['dataset_fingerprint'],
                 predictor=bundle['predictor'].state_dict(), next_data_cursor=bundle['cursor'],
                 saved_proof=saved_proof, golden_next=golden_next, next_rng_probe=next_probe)
    context = ACTIVE_SYNC.set(manager)
    try:
        checkpoint = save_checkpoint(manager.output, bundle['model'], bundle['optimizer'], bundle['step'],
                                     scheduler=bundle['scheduler'], scaler=bundle['scaler'], extra=extra)
    finally:
        ACTIVE_SYNC.reset(context)
    return checkpoint


def restore_and_continue(checkpoint, device, contract):
    """Use the production runtime to restore, then compare against the saved golden branch."""
    from src.training.runtime import load_checkpoint
    from src.training.checkpoint_sync import verify
    verify(checkpoint)
    restored = make_bundle(device)
    state = load_checkpoint(checkpoint, restored['model'], restored['optimizer'],
        scheduler=restored['scheduler'], scaler=restored['scaler'],
        expected_extra={'contract':contract, 'dataset_fingerprint':contract['dataset_fingerprint']})
    restored['predictor'].load_state_dict(state['extra']['predictor'])
    restored['step'], restored['cursor'] = state['step'], state['extra']['next_data_cursor']
    proof = bundle_proof(restored, device)
    assert_proof(proof, state['extra']['saved_proof'], 'restored')
    before_rng = capture_rng(device)
    if rng_probe(device) != state['extra']['next_rng_probe']:
        raise AcceptanceFailure('restored_next_rng_probe_mismatch')
    reset_rng(before_rng, device)
    train_fixture_step(restored, device)
    continued = bundle_proof(restored, device)
    assert_proof(continued, state['extra']['golden_next'], 'continued')
    if continued['hashes']['model_weights']==proof['hashes']['model_weights']:
        raise AcceptanceFailure('fixture_optimizer_did_not_change_model_weights')
    if continued['hashes']['frozen_reference']!=proof['hashes']['frozen_reference']:
        raise AcceptanceFailure('fixture_frozen_reference_changed')
    return restored, dict(restored=proof, continued=continued, next_rng_probe='PASS',
                          exact_continuation='PASS', checkpoint_integrity='PASS')


def tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    core = Tokenizer(models.WordLevel({'[UNK]':0, '[BOS]':1, '[EOS]':2, '[PAD]':3,
                                      'question':4, 'SELECT':5, 'x':6, 'FROM':7, 't':8}, unk_token='[UNK]'))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token='[UNK]', bos_token='[BOS]',
        eos_token='[EOS]', pad_token='[PAD]', model_input_names=['input_ids','attention_mask'])
    tokenizer.save_pretrained(path)
    return tokenizer


def component_probe(component, device, output):
    """Tiny versions of actual architectures/APIs. No pretrained weights or datasets."""
    import torch
    from src.training.runtime import seed_everything
    seed_everything(42)
    assets = Path(output) / 'tiny-assets'
    assets.mkdir(parents=True, exist_ok=False)
    if component == 'grounding':
        from transformers import Qwen3Config, Qwen3Model
        from sentence_transformers import SentenceTransformer, models
        from src.training.train_embedding import compute_hn_supcon_loss
        base = assets/'qwen3';base.mkdir()
        tiny_tokenizer(base)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=64, pad_token_id=3, bos_token_id=1, eos_token_id=2,
            use_sliding_window=False, sliding_window=None)
        Qwen3Model(config).save_pretrained(base)
        model = SentenceTransformer(modules=[models.Transformer(str(base), max_seq_length=16,
            model_args={'attn_implementation':'eager','local_files_only':True}), models.Pooling(32)], device=device)
        features = {key:value.to(device) for key,value in model.tokenize(['question x','SELECT x','FROM t']).items()}
        vectors = model(features)['sentence_embedding']
        loss = compute_hn_supcon_loss(vectors[0], vectors[1], [vectors[2]])
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        architecture = 'tiny Qwen3 + SentenceTransformer + project HN-SupCon'
    elif component == 'system1':
        from transformers import ModernBertConfig, ModernBertForSequenceClassification
        from src.training.train_decision_model import configure_dual_lr_optimizer, compute_rlcd_loss
        config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=64,
            num_labels=1, pad_token_id=3, bos_token_id=1, eos_token_id=2,
            reference_compile=False, attn_implementation='eager')
        model = ModernBertForSequenceClassification(config).to(device)
        logits = model(input_ids=torch.tensor([[1,4,5,2]],device=device),
                       attention_mask=torch.ones(1,4,device=device,dtype=torch.long)).logits.view(-1)
        loss = compute_rlcd_loss(logits.sigmoid(), torch.ones_like(logits))
        optimizer = configure_dual_lr_optimizer(model)
        architecture = 'tiny ModernBERT + project dual-LR/RLCD; reference_compile=False'
    else:
        from transformers import LlamaConfig, LlamaForCausalLM
        from src.training.runtime import load_causal_model
        from src.training.train_sft_generator import joint_sft_jepa_loss
        from src.training.jepa_module import LLM_JEPAPredictor
        base = assets/'llama';base.mkdir()
        tiny_tokenizer(base)
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, pad_token_id=3, bos_token_id=1, eos_token_id=2)
        LlamaForCausalLM(config).save_pretrained(base)
        model, tokenizer = load_causal_model(str(base), training=component!='evaluation', device=device,rank=2)
        if str(device).startswith('cuda') and not getattr(model,'is_loaded_in_4bit',False):
            raise AcceptanceFailure('component_not_loaded_in_4bit')
        ids = torch.tensor([[1,4,5,6,7,8,5,2]],device=device)
        if component == 'grpo_3b':
            from peft import get_peft_model_state_dict
            from src.training.train_grpo_generator import _activate_adapter, completion_token_logprobs, grpo_token_objective
            initial = assets/'reference-init'
            model.save_pretrained(initial)
            model.load_adapter(str(initial), adapter_name='reference', is_trainable=False)
            _activate_adapter(model,'reference')
            with torch.no_grad(): reference = completion_token_logprobs(model,ids,4).detach()
            reference_hash = tree_digest(get_peft_model_state_dict(model, adapter_name='reference'))
            _activate_adapter(model,'default')
            current = completion_token_logprobs(model,ids,4)
            loss = grpo_token_objective(current,current.detach(),reference,1.0)
            architecture = 'tiny NF4/FP32 Llama + policy/reference PEFT + project GRPO token objective (synthetic advantage)'
        elif component == 'evaluation':
            with torch.inference_mode():
                generated = model.generate(input_ids=ids[:,:4],attention_mask=torch.ones_like(ids[:,:4]),
                    max_new_tokens=2,do_sample=False,pad_token_id=3,eos_token_id=2)
            scorer=component_probe('system1',device,assets/'evaluation-scorer')
            return dict(status='PASS',scope='TINY_COMPONENT_API_ONLY', architecture='tiny causal generation + classifier initialization',
                        device=str(device),output_tokens=int(generated.shape[-1]),classifier_probe=scorer,
                        full_evaluation='NOT_EXECUTED')
        else:
            predictor = LLM_JEPAPredictor(32,intermediate_dim=8).to(device)
            labels=ids.clone();labels[:,:4]=-100
            batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,
                       pred_positions=torch.tensor([3],device=device),sql_starts=torch.tensor([4],device=device),
                       sql_ends=torch.tensor([8],device=device))
            loss,_ = joint_sft_jepa_loss(model,batch,predictor,lambda_jepa=0.1,loss_dropout=0.0)
            architecture = 'tiny NF4/FP32 Llama + PEFT + project SFT/JEPA'
        parameters = [p for p in model.parameters() if p.requires_grad]
        if component == 'sft_jepa_3b':parameters += list(predictor.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=1e-4)
    if not torch.isfinite(loss):raise AcceptanceFailure('component_nonfinite_loss')
    # Tiny probes use unscaled FP32 loss to validate gradient connectivity; the
    # infrastructure lifecycle separately verifies GradScaler state/restoration.
    loss.backward()
    gradients=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise AcceptanceFailure('component_missing_or_nonfinite_gradient')
    optimizer.step();optimizer.zero_grad()
    result=dict(status='PASS',scope='TINY_COMPONENT_API_ONLY',architecture=architecture,device=str(device),
                quantization='NF4' if str(device).startswith('cuda') and component in ('sft_jepa_3b','grpo_3b') else 'NONE',
                finite_loss=float(loss.detach()), optimizer_updates=1, pretrained_component='NOT_EXECUTED')
    if component=='grpo_3b':
        if reference_hash!=tree_digest(get_peft_model_state_dict(model,adapter_name='reference')):
            raise AcceptanceFailure('component_reference_adapter_changed')
        result['immutable_reference_adapter']='PASS'
    return result


def assert_gpu_profile(info, selected):
    expected = {'t4': ('T4', [7, 5]), 'p100': ('P100', [6, 0])}[selected]
    if expected[0] not in info['gpu'].upper() or info['capability'] != expected[1]:
        raise AcceptanceFailure('selected_gpu_profile_does_not_match_hardware')


def inspect_imports(component, require_trl=False):
    packages={'transformers':'transformers','peft':'peft','trl':'trl',
              'accelerate':'accelerate','bitsandbytes':'bitsandbytes',
              'huggingface-hub':'huggingface_hub','tokenizers':'tokenizers',
              'numpy':'numpy','sentence-transformers':'sentence_transformers'}
    from src.training.environment import requirements,profile_path
    required=set(requirements(profile_path(ROOT,component)))
    if require_trl:required.add('trl')
    result={}
    for name,module in packages.items():
        try:version=metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name]={'status':'FAIL' if name in required else 'NOT_INSTALLED_OPTIONAL'}
            continue
        try:
            importlib.import_module(module)
            result[name]={'status':'PASS','version':version}
        except Exception as exc:
            result[name]={'status':'FAIL' if name in required else 'FAIL_OPTIONAL',
                          'version':version,'error_type':type(exc).__name__}
    if any(item['status']=='FAIL' for item in result.values()):
        raise AcceptanceFailure('required_import_failed: '+','.join(k for k,v in result.items() if v['status']=='FAIL'))
    return result


def checked(report, name, action):
    report['current_check']=name
    value=action()
    report['checks'][name]={'status':'PASS','details':value}
    return value


def run(args):
    """Report actual execution only. CPU mode cannot access Hub or claim GPU success."""
    if args.cpu_test and (args.hub_upload or args.phase!='local' or args.bootstrap or args.hub_read_check or args.check_model_secret):
        raise ValueError('--cpu-test is local/offline only; Hub/bootstrap/session flags are forbidden')
    if not 1<=args.steps<=3:raise ValueError('--steps must be 1..3')
    if args.phase=='session-a' and not args.hub_upload:
        raise ValueError('Session A requires explicit --hub-upload')
    if args.phase=='session-b':
        if not args.run_id or not re.fullmatch(r'[0-9a-f]{32}',args.run_id) or uuid.UUID(args.run_id).version!=4:
            raise ValueError('Session B requires the exact UUID4 --run-id printed by Session A')
        run_id=args.run_id
    else:
        if args.run_id:raise ValueError('--run-id is only accepted for Session B; fresh tests always generate a new namespace')
        run_id=uuid.uuid4().hex
    if not args.cpu_test and not args.gpu:raise ValueError('Select --gpu t4 or --gpu p100 explicitly')
    device='cpu' if args.cpu_test else args.device
    if not args.cpu_test and (not device.startswith('cuda:') or not device[5:].isdigit()):
        raise ValueError('GPU acceptance requires an explicit --device cuda:N')
    output=Path(args.output)/('acceptance-'+run_id)/args.phase
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(schema_version=1,run_id=run_id,experiment_id='acceptance-'+run_id,phase=args.phase,
        output=str(output),scope='LOCAL_CPU_TEST' if args.cpu_test else 'GPU_ACCEPTANCE_REQUESTED',
        execution_site='local_cpu' if args.cpu_test else 'unverified',
        result='FAIL',full_training_allowed=False,checks={name:{'status':'NOT_EXECUTED'} for name in
        ('bootstrap','checkpoint_secret','environment','imports','pip_check','nf4','model_secret','private_repo_read','component_probe',
         'fixture_lifecycle','private_hub_round_trip','cross_session_resume','pretrained_component','full_training')})
    values=dict(CHECKPOINT_EXPERIMENT_ID=report['experiment_id'],CHECKPOINT_SYNC_BACKEND='hf_hub',
        CHECKPOINT_SYNC_ENABLED='1' if args.hub_upload else '0',CHECKPOINT_SYNC_FAILURE_POLICY='stop',
        CHECKPOINT_SYNC_EVERY_N_SAVES='1000000',CHECKPOINT_RESUME_SOURCE='none',
        KAGGLE_OUTPUT_DIR=str(output),PIPELINE_MODE='train',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        with environment(values):
            if args.bootstrap:
                from src.training.kaggle import install_dependencies
                def bootstrap():
                    install_dependencies(ROOT,args.component)
                    snapshot=output/'dependency_snapshot.json'
                    data=json.loads(snapshot.read_text())
                    return dict(snapshot=str(snapshot),profile=data['profile'],protected_cuda_stack=data['protected_cuda_stack'])
                with environment({'INSTALL_DEPS':'1'}):
                    checked(report,'bootstrap',bootstrap)
            if args.hub_upload or args.phase=='session-b' or args.hub_read_check:
                from src.training.hf_auth import hf_token
                checked(report,'checkpoint_secret',lambda:bool(hf_token(required=True)))
            from src.training.kaggle import environment_report
            report['inventory']=environment_report(output)
            report['python']=platform.python_version()
            if not args.cpu_test:
                from src.training.environment import validate_environment
                info=checked(report,'environment',lambda:validate_environment(ROOT,args.component,output,device,quantization=True))
                assert_gpu_profile(info,args.gpu)
                report['scope']='GPU_INFRASTRUCTURE_ONLY'
                report['execution_site']='kaggle' if Path('/kaggle/working').is_dir() else 'other_cuda_host'
                if info.get('nf4_kernel')=='passed':
                    report['checks']['nf4']={'status':'PASS','details':'Existing environment gate verified NF4 forward/backward'}
                report['gpu_profile']=args.gpu
                report['python']=platform.python_version()
                report['native_bf16']=False
                checked(report,'imports',lambda:inspect_imports(args.component,args.require_trl))
                report['current_check']='pip_check'
                result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
                report['checks']['pip_check']={'status':'PASS' if result.returncode==0 else 'FAIL',
                    'details':{'returncode':result.returncode,'output':result.stdout.strip()[:8000]}}
                if result.returncode:raise AcceptanceFailure('dependency_conflicts_run_pip_check')
                driver=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],
                                      capture_output=True,text=True)
                if driver.returncode:raise AcceptanceFailure('nvidia_smi_failed')
                report['nvidia_smi']=driver.stdout.strip()
            if args.check_model_secret:
                from src.training.hf_auth import hf_token
                checked(report,'model_secret',lambda:bool(hf_token(model=True,required=True)))
            import torch
            torch.set_num_threads(1)
            if not args.cpu_test:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.cuda.reset_peak_memory_stats(device)
            from src.training.checkpoint_sync import CheckpointSync,fingerprint,verify
            contract=fixture_contract(device,'cpu' if args.cpu_test else args.gpu,args.component)
            compatibility=fingerprint(contract)
            if args.hub_upload:report['current_check']='private_repo_read'
            writer=CheckpointSync(output/'checkpoints',STAGE,compatibility)
            writer.configuration=contract
            if args.hub_upload:
                report['repository']=writer.repo
                checked(report,'private_repo_read',lambda:dict(private=writer._retry(writer._info).private))
            if args.hub_read_check or args.phase=='session-b':
                report['current_check']='private_repo_read'
                with environment({'CHECKPOINT_SYNC_ENABLED':'1'}):
                    reader=CheckpointSync(output/'download',STAGE,compatibility,read_only=True)
                checked(report,'private_repo_read',lambda:dict(private=reader._retry(reader._info).private))
            if args.component_probe:
                checked(report,'component_probe',lambda:component_probe(args.component,device,output))
            if args.phase=='session-b':
                # Start with an empty destination and require the original source
                # step, rather than silently picking another acceptance run/step.
                report['current_check']='cross_session_resume'
                report['repository']=reader.repo
                candidates=reader.remote_candidates()
                selected=next((c for c in candidates if c[0]==args.steps),None)
                if selected is None:raise AcceptanceFailure('expected_session_a_step_not_found')
                if sum(item['size'] for item in selected[3]['files'].values())>MAX_UPLOAD_BYTES:
                    raise AcceptanceFailure('acceptance_payload_exceeds_5_MiB')
                checkpoint=reader.fetch(selected)
                report['source_revision']=selected[2]
            else:
                bundle=make_bundle(device)
                for _ in range(args.steps):train_fixture_step(bundle,device)
                checkpoint=save_fixture(bundle,writer,device,contract)
                del bundle
            report['current_check']='fixture_lifecycle'
            restored,proof=restore_and_continue(checkpoint,device,contract)
            report['checks']['fixture_lifecycle']={'status':'PASS','details':proof}
            proof['checkpoint']=str(checkpoint)
            proof['manifest_fingerprint']=fingerprint(verify(checkpoint))
            if args.phase=='session-b':
                report['checks']['cross_session_resume']={'status':'PASS','details':proof,
                    'isolation':{'source':'private_hf','download_destination_initially_empty':True,
                                 'fresh_kaggle_runtime':'OPERATOR_ATTESTATION_REQUIRED'}}
                checkpoint=save_fixture(restored,writer,device,contract)
            if args.hub_upload:
                report['current_check']='private_hub_round_trip'
                manifest=verify(checkpoint)
                size=sum(item['size'] for item in manifest['files'].values())
                if size>MAX_UPLOAD_BYTES:raise AcceptanceFailure('acceptance_payload_exceeds_5_MiB')
                if not writer.sync(checkpoint):raise AcceptanceFailure('hub_upload_not_verified')
                with environment({'CHECKPOINT_SYNC_ENABLED':'1'}):
                    receiver=CheckpointSync(output/'round-trip-download',STAGE,compatibility,read_only=True)
                candidates=receiver.remote_candidates()
                selected=next((c for c in candidates if c[0]==manifest['step'] and c[3]==manifest),None)
                if selected is None:raise AcceptanceFailure('uploaded_checkpoint_not_discoverable')
                downloaded=receiver.fetch(selected)
                report['round_trip_revision']=selected[2]
                if verify(downloaded)!=manifest:raise AcceptanceFailure('round_trip_manifest_mismatch')
                _,roundtrip=restore_and_continue(downloaded,device,contract)
                report['checks']['private_hub_round_trip']={'status':'PASS','details':dict(bytes=size,
                    checkpoint=str(downloaded),manifest_fingerprint=fingerprint(manifest),proof=roundtrip)}
            report['result']='PASS_LOCAL_CPU_ONLY' if args.cpu_test else 'PASS_INFRASTRUCTURE'
            report['checks']['pretrained_component']={'status':'NOT_EXECUTED','reason':'Tiny random models do not validate pretrained memory/quality'}
            if not args.cpu_test:report['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated(device)
    except Exception as exc:
        failing=report.get('current_check')
        if failing in report['checks']:
            report['checks'][failing]['status']='FAIL'
        report['failure']={'check':report.get('current_check'),'error_type':type(exc).__name__,
            'http_status':getattr(exc,'http_status',None) or getattr(getattr(exc,'response',None),'status_code',None),
            'diagnostic':str(exc) if isinstance(exc,AcceptanceFailure) else 'See the named check and KAGGLE acceptance diagnostics; provider messages are suppressed',
            'hint':{'environment':'Run check_kaggle_environment.py for this component/GPU; compare dependency_snapshot.json',
                    'bootstrap':'Use a cold kernel and compatible image; never replace protected CUDA packages',
                    'private_repo_read':'Grant the configured Kaggle Secret, check private Model repo and read/write scope',
                    'checkpoint_secret':'Grant CHECKPOINT_HF_TOKEN_SECRET (default HF_TOKEN) to this Kaggle notebook',
                    'private_hub_round_trip':'Check sync JSONL HTTP category, network/cache space and explicit write permission',
                    'cross_session_resume':'Use the same repo, run ID, steps, component, GPU type and package snapshot as Session A'}.get(failing,'Check acceptance_report.json and the documented failure criteria')}
        report['result']='FAIL'
    finally:
        report.pop('current_check',None)
        report['elapsed_seconds']=time.monotonic()-started
        (output/'acceptance_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(report,indent=2,allow_nan=False))
    return report


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--phase',choices=('local','session-a','session-b'),default='local')
    result.add_argument('--gpu',choices=('t4','p100'))
    result.add_argument('--device',default='cuda:0')
    result.add_argument('--output',default='/kaggle/working/kaggle-acceptance')
    result.add_argument('--component',choices=COMPONENTS,default='sft_jepa_3b')
    result.add_argument('--component-probe',action='store_true')
    result.add_argument('--steps',type=int,default=2,help='Session A source updates, 1..3; Session B must use the same value')
    result.add_argument('--run-id')
    result.add_argument('--bootstrap',action='store_true',help='Explicitly run the existing constrained component installer')
    result.add_argument('--hub-upload',action='store_true',help='EXPLICIT OPT-IN: write only a small isolated acceptance artifact')
    result.add_argument('--hub-read-check',action='store_true')
    result.add_argument('--check-model-secret',action='store_true',help='Check Secret availability only, not gated model authorization')
    result.add_argument('--require-trl',action='store_true',help='TRL is optional for project custom loops; make its import mandatory')
    result.add_argument('--cpu-test',action='store_true',help='Offline fixture validation only; cannot run bootstrap/Hub actions or claim GPU acceptance')
    return result

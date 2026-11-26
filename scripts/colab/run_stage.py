"""
Runs one pipeline stage on Google Colab with data and outputs on Google Drive.

Every member uses this script instead of ad-hoc cells. It maps Drive paths onto the environment
variables the notebooks already read (KAGGLE_* names are kept on purpose), then executes the
committed notebook in a fresh kernel with nbconvert. The executed notebook, a log and launch.json
(commit, arguments, environment) are written next to the outputs on Drive.

  prepare    unzip BIRD from Drive, rebuild local databases, write JSONL splits to Drive
  grounding  notebooks/01   sft  notebooks/02   grpo  notebooks/03
  system1    notebooks/04   evaluation  notebooks/05

Rerunning the same --run-id resumes from the newest checkpoint in that run folder, but only with the
same code: the runner refuses to resume when the code hash differs from the newest launch json.
--overlay <dir> copies the files of <dir> over the repo before anything runs (members without git).
Each launch json records commit, dirty, code_hash and a patch (code-<stamp>.patch) that rebuilds the
working tree from that commit with `git apply`.

Example:
  python scripts/colab/run_stage.py --stage grounding --run-id m1-grounding-seed42
  python scripts/colab/run_stage.py --stage sft --run-id m2-sft-seed42 --set LAMBDA_JEPA=0
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Text2SQL_Colab"
LOCAL_DB_DIR = "/content/bird_db"
LOCAL_BIRD_ROOT = "/content/bird"
DEFAULT_GENERATOR = "Qwen/Qwen2.5-Coder-3B-Instruct"

NOTEBOOKS = {
    "grounding": "01_data_preparation_and_embedding.ipynb",
    "sft": "02_sft_and_jepa_training_3b.ipynb",
    "grpo": "03_grpo_training_3b_and_7b.ipynb",
    "system1": "04_system1_laya421m_training.ipynb",
    "evaluation": "05_end_to_end_evaluation.ipynb",
}


def has_checkpoint(run_dir: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("checkpoint") for p in run_dir.rglob("checkpoint*"))


def build_env(stage: str, drive_root: str, run_id: str, overrides: Dict[str, str],
              run_dir_has_checkpoint: bool, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Pure mapping from runner arguments to notebook environment variables (tested)."""
    env = dict(base if base is not None else os.environ)
    env.update({
        "KAGGLE_REPO_DIR": str(REPO),
        "KAGGLE_DATA_DIR": f"{drive_root}/data",
        "KAGGLE_OUTPUT_DIR": f"{drive_root}/runs/{run_id}",
        "DB_DIR": LOCAL_DB_DIR,
        "PIPELINE_MODE": "train",
        "INSTALL_DEPS": "1",
        "TRAIN_DEVICE": "cuda:0",
        "HF_HOME": "/content/hf-cache",
        "SEED": "42",
        "MAX_LENGTH": "2048" if stage in ("sft", "grpo") else "512",
        "MAX_STEPS": "100",
        "SAVE_STEPS": "25",
        "GRAD_ACCUM": "8",
        # Drive is the durable store; the optional HF Hub backup stays off unless set explicitly.
        "CHECKPOINT_SYNC_ENABLED": "0",
        "RESUME_FROM_CHECKPOINT": "latest" if run_dir_has_checkpoint else "none",
    })
    if stage in ("sft", "grpo", "evaluation"):
        env.setdefault("GENERATOR_MODEL", DEFAULT_GENERATOR)
    if stage == "grpo":
        env["MAX_NEW_TOKENS"] = "256"  # Same completion budget as evaluation.
    env.update(overrides)
    return env


def parse_overrides(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value
    return out


def colab_hf_token() -> Optional[str]:
    """Reads HF_TOKEN from Colab Secrets when available; never printed or written to disk."""
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def ensure_drive(drive_root: str) -> None:
    if not os.path.isdir(drive_root):
        raise SystemExit(f"Drive folder not found: {drive_root}. Mount Drive first: drive.mount('/content/drive').")


def ensure_databases(drive_root: str) -> None:
    """Rebuilds the local database folder from the BIRD zips on Drive when a new runtime lost it."""
    if os.path.isdir(LOCAL_DB_DIR) and any(Path(LOCAL_DB_DIR).glob("*.sqlite")):
        return
    zips = Path(drive_root) / "data" / "bird"
    os.makedirs(LOCAL_BIRD_ROOT, exist_ok=True)
    for name, marker in (("dev.zip", "dev_20240627"), ("train.zip", "train")):
        if not (Path(LOCAL_BIRD_ROOT) / marker).is_dir():
            subprocess.run(["unzip", "-q", "-o", str(zips / name), "-d", LOCAL_BIRD_ROOT], check=True)
    subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_bird.py"), "--root", LOCAL_BIRD_ROOT,
                    "--data_dir", f"{drive_root}/data", "--db_dir", LOCAL_DB_DIR, "--databases_only"], check=True)


def apply_overlay(overlay: Path, repo: Path = REPO) -> List[str]:
    """Copies every file of `overlay` over the repo; refuses .git, symlinks and paths leaving the repo."""
    overlay, root = Path(overlay).resolve(), Path(repo).resolve()
    if not overlay.is_dir():
        raise SystemExit(f"Overlay folder not found: {overlay}")
    plan = []
    for dirpath, dirnames, filenames in os.walk(overlay):
        for name in dirnames + filenames:
            src = Path(dirpath) / name
            rel = src.relative_to(overlay)
            if src.is_symlink():
                raise SystemExit(f"Overlay refused: symlink {rel} (copy the real file instead)")
            if name in filenames:
                plan.append((src, overlay_target(root, rel.as_posix())))
    for src, dest in plan:  # Copy only after every path passed the checks.
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return [str(dest.relative_to(root)) for _, dest in plan]


def overlay_target(root: Path, rel: str) -> Path:
    dest = (root / rel).resolve()
    if Path(rel).is_absolute() or ".." in Path(rel).parts or dest == root or root not in dest.parents:
        raise SystemExit(f"Overlay refused: {rel} escapes the repo")
    if dest.relative_to(root).parts[0].lower() == ".git":
        raise SystemExit(f"Overlay refused: {rel} writes into .git")
    return dest


def git(repo: Path, *args: str, env: Optional[Dict[str, str]] = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, env=env).stdout


def code_provenance(repo: Path = REPO) -> Dict[str, object]:
    """Exact commit, dirty flag and a hash over every tracked and untracked non-ignored file."""
    try:
        commit = git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot resolve the code commit of {repo} ({exc}). Run from a git clone with a "
                         "checked-out commit; results without a commit are not traceable.")
    files = sorted({os.fsdecode(p) for p in git(repo, "ls-files", "-z", "-co", "--exclude-standard").split(b"\0") if p})
    digest = hashlib.sha256()
    for rel in files:
        path = Path(repo) / rel
        if path.is_file():  # Deleted tracked files drop out of the hash.
            data = path.read_bytes()
            digest.update(rel.encode("utf-8", "surrogateescape") + b"\0" + str(len(data)).encode() + b"\0" + data)
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=all").strip())
    return {"commit": commit, "dirty": dirty, "code_hash": digest.hexdigest()}


def code_patch(repo: Path, commit: str) -> bytes:
    """Binary diff from `commit` to the working tree, untracked files included, without touching the real index."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        git(repo, "add", "-A", env=env)
        return git(repo, "diff", "--cached", "--binary", commit, env=env)


def check_resume(run_dir: Path, code_hash: str) -> None:
    """A checkpoint may only be resumed by the code that wrote it; otherwise use a new --run-id."""
    if not has_checkpoint(run_dir):
        return
    launches = sorted(run_dir.glob("launch-*.json"))
    previous = json.loads(launches[-1].read_text()).get("code_hash") if launches else None
    if previous != code_hash:
        raise SystemExit(f"Refusing to resume {run_dir}: its checkpoint was written by code_hash {previous or 'none'}, "
                         f"current code_hash is {code_hash}. Start a new --run-id or restore that code.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["prepare", *NOTEBOOKS])
    parser.add_argument("--run-id", help="Output folder name under <drive-root>/runs; required except for prepare")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a notebook environment variable, e.g. MAX_STEPS=50")
    parser.add_argument("--overlay", help="Folder whose files are copied over the repo before the run")
    args = parser.parse_args()

    if args.overlay:
        print("overlay:", ", ".join(apply_overlay(Path(args.overlay))), flush=True)

    ensure_drive(args.drive_root)
    ensure_databases(args.drive_root)
    if args.stage == "prepare":
        subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_bird.py"), "--root", LOCAL_BIRD_ROOT,
                        "--data_dir", f"{args.drive_root}/data", "--db_dir", LOCAL_DB_DIR], check=True)
        return
    if not args.run_id:
        raise SystemExit("--run-id is required for training and evaluation stages")

    run_dir = Path(args.drive_root) / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args.stage, args.drive_root, args.run_id, parse_overrides(args.set), has_checkpoint(run_dir))
    token = colab_hf_token()
    if token and "HF_TOKEN" not in env:
        env["HF_TOKEN"] = token

    provenance = code_provenance()
    check_resume(run_dir, provenance["code_hash"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    patch = f"code-{stamp}.patch"
    (run_dir / patch).write_bytes(code_patch(REPO, provenance["commit"]))
    launch = {"stage": args.stage, "run_id": args.run_id, **provenance, "overlay": args.overlay, "patch": patch,
              "started": stamp, "python": sys.version.split()[0], "overrides": parse_overrides(args.set),
              "env": {k: env[k] for k in sorted(env) if k.startswith(("KAGGLE_", "PIPELINE", "MAX_", "SAVE_",
                      "GRAD_", "SEED", "RESUME", "GENERATOR", "LAMBDA", "DB_DIR", "CHECKPOINT_SYNC"))}}
    (run_dir / f"launch-{stamp}.json").write_text(json.dumps(launch, indent=2))

    notebook = REPO / "notebooks" / NOTEBOOKS[args.stage]
    cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", str(notebook),
           "--ExecutePreprocessor.timeout=-1", "--output-dir", str(run_dir),
           "--output", f"{notebook.stem}-executed-{stamp}.ipynb"]
    print(json.dumps(launch, indent=2), flush=True)
    result = subprocess.run(cmd, env=env, cwd=str(REPO))
    print(f"STAGE_{args.stage.upper()}_{'DONE' if result.returncode == 0 else 'FAILED'}", flush=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

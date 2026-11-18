"""Lightweight environment gate. No model downloads or Hub writes.

--smoke: dependency-light offline wiring. Default: validate selected Kaggle profile.
--quantization: additionally run tiny NF4 forward/backward for Generation.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.environment import validate_environment, PROFILES, selected_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*PROFILES, 'core'), default='sft_jepa_3b')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='/kaggle/working/environment-check')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--profile', choices=('LEGACY', 'MODERN'))
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--resolve-modern-lock', action='store_true')
    action.add_argument('--create-modern-venv', metavar='PATH')
    action.add_argument('--install-modern-lock', action='store_true')
    args = parser.parse_args()
    profile = selected_profile(args.profile)
    repo = Path(__file__).resolve().parents[1]
    if args.resolve_modern_lock or args.create_modern_venv or args.install_modern_lock:
        if profile != 'MODERN' or args.smoke or args.quantization:
            parser.error('Modern environment actions require MODERN without smoke or quantization')
        if args.create_modern_venv:
            from src.training.kaggle_isolation import create_modern_venv
            print(create_modern_venv(args.create_modern_venv))
        elif args.resolve_modern_lock:
            from src.training.modern_environment import resolve_candidate
            resolve_candidate(repo, args.stage, args.output)
        else:
            from src.training.modern_environment import modern_operation
            modern_operation(repo, args.stage, args.output, install=True)
        return
    result = validate_environment(repo, args.stage,
        args.output, args.device, smoke=args.smoke, quantization=args.quantization, profile=profile)
    if args.smoke:
        print(result)


if __name__ == '__main__':
    main()

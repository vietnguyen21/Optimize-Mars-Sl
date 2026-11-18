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
    parser.add_argument('--resolve-modern-lock', action='store_true')
    args = parser.parse_args()
    profile = selected_profile(args.profile)
    if args.resolve_modern_lock:
        if profile != 'MODERN' or args.smoke or args.quantization:
            parser.error('--resolve-modern-lock requires MODERN without smoke or quantization')
        from src.training.modern_environment import resolve_candidate
        resolve_candidate(Path(__file__).resolve().parents[1], args.stage, args.output)
        return
    result = validate_environment(Path(__file__).resolve().parents[1], args.stage,
        args.output, args.device, smoke=args.smoke, quantization=args.quantization, profile=profile)
    if args.smoke:
        print(result)


if __name__ == '__main__':
    main()

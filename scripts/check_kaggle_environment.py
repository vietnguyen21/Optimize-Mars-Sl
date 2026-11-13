"""Lightweight environment gate. No model downloads or Hub writes.

--smoke: dependency-light offline wiring. Default: validate selected Kaggle profile.
--quantization: additionally run tiny NF4 forward/backward for Generation.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.environment import validate_environment, PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=PROFILES, default='sft_jepa_3b')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='/kaggle/working/environment-check')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--quantization', action='store_true')
    args = parser.parse_args()
    result = validate_environment(Path(__file__).resolve().parents[1], args.stage,
        args.output, args.device, smoke=args.smoke, quantization=args.quantization)
    if args.smoke:
        print(result)


if __name__ == '__main__':
    main()

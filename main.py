"""
main.py — run the full pipeline or resume from a given step.

Usage:
  python main.py                    # clean → features → train
  python main.py --from features    # features → train
  python main.py --from train       # train only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import clean
import features
import train

STEPS = [
    ('clean',    clean.main),
    ('features', features.main),
    ('train',    train.main),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from',
        dest='from_step',
        choices=[name for name, _ in STEPS],
        default=STEPS[0][0],
        help='resume pipeline from this step (default: clean)',
    )
    args = parser.parse_args()

    start = [name for name, _ in STEPS].index(args.from_step)
    for name, fn in STEPS[start:]:
        print(f'\n{"=" * 60}')
        print(f'  STEP: {name}')
        print(f'{"=" * 60}\n')
        fn()

    print('\nPipeline complete.')


if __name__ == '__main__':
    main()

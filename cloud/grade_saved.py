"""Grade immutable trajectories without starting a generation engine."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.offline import grade_saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--questions', default='questions')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sandbox', choices=('bwrap', 'rlimit', 'off'))
    args = parser.parse_args()
    grade_saved(args.source, args.out, args.questions, args.workers, args.sandbox)

"""
run_all.py
==========
Master script — runs the complete research pipeline in order.

Usage:
    python run_all.py                    # full pipeline
    python run_all.py --skip_preprocess  # skip if already preprocessed
    python run_all.py --ablation         # train & evaluate all baseline models too
    python run_all.py --search           # run hyperparameter search before training

Steps:
    1. Preprocess all 5 CSVs
    2. (Optional) Hyperparameter search
    3. Train GreenEyes++ (+ baselines if --ablation)
    4. Evaluate all trained models
    5. Print final research summary table
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from configs.config import *

set_seed(RANDOM_SEED)


def run(cmd: str, step: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {step}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd.split(),
        cwd=str(ROOT_DIR),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {step}")
        sys.exit(1)
    print(f"\n  ✓ Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_preprocess', action='store_true')
    parser.add_argument('--search',          action='store_true',
                        help='Run Optuna hyperparameter search')
    parser.add_argument('--n_trials',        type=int, default=50)
    parser.add_argument('--ablation',        action='store_true',
                        help='Train and evaluate all baseline models')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  AQI-Sense India — Full Research Pipeline")
    print("="*60)
    print(f"  Root dir : {ROOT_DIR}")
    print(f"  Data raw : {DATA_RAW}")
    print(f"  Processed: {DATA_PROCESSED}")
    print(f"  Results  : {RESULTS_DIR}")

    # Check data exists
    required = ['city_day.csv', 'city_hour.csv', 'station_day.csv',
                'station_hour.csv', 'stations.csv']
    for f in required:
        if not (DATA_RAW / f).exists():
            print(f"\n[ERROR] Missing: {DATA_RAW / f}")
            print("Place the 5 Kaggle CSVs in data/raw/ before running.")
            sys.exit(1)

    # Step 1: Preprocess
    if not args.skip_preprocess:
        run('preprocessing/preprocess.py', 'Preprocessing (all 5 CSVs)')
    else:
        print("\n[SKIP] Preprocessing (--skip_preprocess)")

    # Step 2: Hyperparameter search (optional)
    if args.search:
        run(f'training/hyperparameter_search.py --n_trials {args.n_trials}',
            f'Hyperparameter search ({args.n_trials} trials)')

    # Step 3: Train GreenEyes++
    run('training/train.py --model greeneyes', 'Training GreenEyes++')

    # Step 3b: Train baselines for ablation
    if args.ablation:
        for m in ['lstm', 'gru', 'transformer', 'wavenet']:
            run(f'training/train.py --model {m}', f'Training {m} baseline')

    # Step 4: Evaluate
    eval_cmd = 'evaluation/evaluate.py'
    if args.ablation:
        eval_cmd += ' --ablation'
    else:
        eval_cmd += ' --model greeneyes'
    run(eval_cmd, 'Evaluation')

    # Final summary
    import pandas as pd
    results_path = RESULTS_DIR / 'overall_metrics.csv'
    if results_path.exists():
        df = pd.read_csv(results_path)
        print("\n" + "="*60)
        print("  FINAL RESULTS SUMMARY")
        print("="*60)
        reg = df[df['horizon'] != 'classification']
        print(reg.groupby(['model', 'horizon'])[['MAE', 'RMSE', 'R2']].first().to_string())

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print(f"  Results: {RESULTS_DIR.resolve()}")
    print("="*60)


if __name__ == '__main__':
    main()

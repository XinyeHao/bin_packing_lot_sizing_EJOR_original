"""Train LightGBM LambdaRank: pretrain then optional fine-tune (transfer)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configuration import ML_BRANCH_MODEL_PATH, ML_BRANCH_PRETRAIN_PATH
from model_and_algo.ml_branch_selector import MLBranchSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-dir", type=Path, default=ROOT / "ml_train_data" / "pretrain")
    ap.add_argument("--finetune-dir", type=Path, default=ROOT / "ml_train_data" / "finetune")
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    sel = MLBranchSelector()
    print(f"Pretrain from {args.pretrain_dir}", flush=True)
    pre = sel.train_from_dir(
        args.pretrain_dir,
        out_path=ML_BRANCH_PRETRAIN_PATH,
        num_boost_round=250,
        learning_rate=0.05,
    )
    print(f"Saved pretrain -> {pre}", flush=True)

    if args.skip_finetune:
        # copy pretrain as deploy model
        ML_BRANCH_MODEL_PATH.write_text(pre.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Deploy model = pretrain copy -> {ML_BRANCH_MODEL_PATH}", flush=True)
        return

    if not args.finetune_dir.exists() or not any(args.finetune_dir.glob("*.csv")):
        print("No finetune data; using pretrain as deploy model", flush=True)
        ML_BRANCH_MODEL_PATH.write_text(pre.read_text(encoding="utf-8"), encoding="utf-8")
        return

    print(f"Fine-tune from {args.finetune_dir}", flush=True)
    out = sel.train_from_dir(
        args.finetune_dir,
        out_path=ML_BRANCH_MODEL_PATH,
        num_boost_round=80,
        learning_rate=0.02,
        init_model=pre,
    )
    print(f"Saved transferred model -> {out}", flush=True)


if __name__ == "__main__":
    main()

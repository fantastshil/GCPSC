"""Train, validate and test the GCPSC core-loss model.

Examples (run from this folder):
  python run.py --dataset mimic_full   --mode train
  python run.py --dataset skincap      --mode test --checkpoint outputs/skincap_seed42/checkpoint_best.pt --bootstrap
  python run.py --dataset mimic_full   --mode data-check

--mode train      train + validate every epoch, keep the validation-best checkpoint,
                  then test it (add --skip-test to stop after validation).
--mode test       evaluate a saved best checkpoint on validation and test.
--mode data-check load and summarise the splits only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gcpsc.pipeline import DATASETS, run, run_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mode", choices=("train", "test", "data-check"), default="train")
    parser.add_argument("--config", default="", help="Defaults to configs/<dataset>.yaml")
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/<dataset>_seed<seed>")
    parser.add_argument("--bert-model-name", default="", help="Local ClinicalBERT folder; overrides the config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Several fresh training runs in sequence, saved to <output-dir>/seed_<N>")
    parser.add_argument("--resume", default="", help="Training: resume from this checkpoint (default: <output-dir>/checkpoint_last.pt if present)")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to evaluate (default: <output-dir>/checkpoint_best.pt)")
    parser.add_argument("--skip-test", action="store_true", help="Stop after validating the best checkpoint")
    parser.add_argument("--bootstrap", action="store_true", help="Add 1000-draw bootstrap confidence intervals to test results")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set completion_nll_weight=0.1 (repeatable; "
                             "dotted keys such as optimizer.lr reach nested sections)")
    args = parser.parse_args()
    if args.seeds:
        run_seeds(args, Path(__file__).resolve())
    else:
        run(args)


if __name__ == "__main__":
    main()

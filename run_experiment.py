"""Launch FREQ and paper controls using the dataset configurations."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "a0": ["--disable_idf"],
    "idf": ["--idf_components", "htq", "--idf_weight_form", "log"],
    "rcqw": ["--use_rcqg", "--disable_idf"],
    "freq": ["--use_rcqg", "--idf_components", "htq", "--idf_weight_form", "log"],
    "random": ["--use_random_pair_weight", "--disable_idf"],
    "linear": ["--idf_components", "htq", "--idf_weight_form", "linear"],
    "sqrt": ["--idf_components", "htq", "--idf_weight_form", "sqrt"],
    "bilinear": [
        "--use_rcqg",
        "--use_rcqg_bilinear",
        "--idf_components",
        "htq",
        "--idf_weight_form",
        "log",
    ],
    "daqw": [
        "--use_rcqg",
        "--use_daqg",
        "--idf_components",
        "htq",
        "--idf_weight_form",
        "log",
    ],
}


def build_command(args):
    configs = json.loads((ROOT / "configs/paper.json").read_text(encoding="utf-8"))
    config = configs[args.dataset]
    command = [
        sys.executable,
        args.mode + ".py",
        "--dataset_name",
        args.dataset,
        "--setting",
        "Inductive",
        "--data_dir",
        str(ROOT / "data") + "/",
        "--exp",
        args.exp,
        "--log_name",
        args.log_name or f"{args.dataset}_{args.variant}_s{args.seed}",
    ]
    keys = ["dim", "num_init_layer", "num_layer", "num_head"]
    if args.mode == "train":
        keys += [
            "lr",
            "num_epoch",
            "val_dur",
            "batch_num",
            "train_graph_ratio",
            "model_dropout",
            "smoothing",
        ]
        command += ["--seed", str(args.seed), "--early_stop", "0"]
    elif args.epoch is None:
        raise ValueError(
            "test requires --epoch from the selected validation checkpoint"
        )
    else:
        command += ["--test_epoch", str(args.epoch)]
    for key in keys:
        value = (
            args.epochs
            if key == "num_epoch" and args.epochs is not None
            else config[key]
        )
        command += ["--" + key, str(value)]
    if config["msg_add_tr"]:
        command.append("--msg_add_tr")
    command += ["--eval_pair_freq_source", "inference"] + VARIANTS[args.variant]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument(
        "--dataset",
        choices=["WD20K100v1", "WD20K100v2", "WP-IND", "MFB-IND"],
        required=True,
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="freq")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp", default="FREQ_neurocomputing")
    parser.add_argument("--log-name")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command = build_command(args)
    except ValueError as error:
        parser.error(str(error))
    print("Working directory:", ROOT / "code")
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT / "code", check=True)


if __name__ == "__main__":
    main()

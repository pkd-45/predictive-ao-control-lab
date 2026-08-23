from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import make_plots, write_sqlite
from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate_all
from .report import markdown_table
from .train import fit_linear, train_mlp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic predictive AO control lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ["generate", "train", "evaluate", "all"]:
        s = sub.add_parser(cmd)
        s.add_argument("--config", default="configs/baseline.yaml")
        s.add_argument("--tag", default="baseline", help="Dataset/result regime tag")
        s.add_argument(
            "--model-tag",
            default=None,
            help="Model artifact tag for evaluation. Defaults to --tag. Use baseline here for distribution-shift tests.",
        )
    return p


def paths(tag: str, model_tag: str | None = None) -> dict[str, Path]:
    mt = model_tag or tag
    return {
        "data": Path("data") / f"{tag}.h5",
        "linear": Path("models") / f"linear_{mt}.npz",
        "model": Path("models") / f"mlp_{mt}.pt",
        "result": Path("results") / f"{tag}.json",
        "db": Path("results") / "benchmarks.sqlite3",
        "plots": Path("results") / tag,
        "report": Path("results") / f"{tag}.md",
    }


def do_generate(cfg, pth):
    generate_dataset(cfg, pth["data"])
    print(f"dataset={pth['data']}")


def do_train(cfg, pth):
    fit_linear(cfg, pth["data"], pth["linear"])
    train_mlp(cfg, pth["data"], pth["model"])
    print(f"linear_model={pth['linear']}")
    print(f"ml_model={pth['model']}")


def do_evaluate(cfg, pth, tag):
    evaluate_all(cfg, pth["data"], pth["linear"], pth["model"], pth["result"])
    write_sqlite(pth["result"], pth["db"], tag)
    make_plots(pth["result"], pth["plots"])
    table = markdown_table(pth["result"])
    pth["report"].write_text(f"# {tag} benchmark\n\n{table}\n")
    print(table)
    print(f"result_json={pth['result']}")
    print(f"sqlite={pth['db']}")


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    pth = paths(args.tag, args.model_tag)
    if args.cmd in ("generate", "all"):
        do_generate(cfg, pth)
    if args.cmd in ("train", "all"):
        if args.model_tag not in (None, args.tag):
            raise SystemExit("--model-tag different from --tag is only valid with evaluate")
        do_train(cfg, pth)
    if args.cmd in ("evaluate", "all"):
        do_evaluate(cfg, pth, args.tag)


if __name__ == "__main__":
    main()

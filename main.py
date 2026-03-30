"""
Entry point for Ba et al. ICCV 2015 reproduction.
Run training or evaluation; see Code/README for data setup.
"""
import argparse
import sys
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Zero-shot CNN from text (Ba et al. ICCV 2015)")
    parser.add_argument("mode", choices=["train", "eval", "check"], nargs="?", default="check",
                        help="train / eval / check (default: check model and deps)")
    args, rest = parser.parse_known_args()
    if args.mode == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + rest
        train_main()
    elif args.mode == "eval":
        from scripts.evaluate import main as eval_main
        sys.argv = [sys.argv[0]] + rest
        eval_main()
    else:
        # Quick sanity check
        import torch
        from models.zero_shot_model import ZeroShotModel
        m = ZeroShotModel(text_input_dim=9763, k=50)
        x = torch.randn(2, 3, 224, 224)
        t = torch.randn(10, 9763) * 0.01
        y = m(x, t)
        assert y.shape == (2, 10), y.shape
        print("Model check OK: ZeroShotModel(2 images, 10 classes) -> scores (2, 10).")
        print("Run: python main.py train  |  python main.py eval")


if __name__ == "__main__":
    main()

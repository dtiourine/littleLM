import argparse

from littlelm.config import DATA_DIR, MODEL_DIR, DataConfig, TrainConfig
from littlelm.generate import generate
from littlelm.network.transformer import LittleLM
from littlelm.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    data_cfg = DataConfig()
    train_cfg = TrainConfig()

    print(f"Loading model from {MODEL_DIR}")
    model = LittleLM.from_pretrained(MODEL_DIR)

    tokenizer_path = DATA_DIR / data_cfg.tokenizer_file
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        context_len=train_cfg.context_len,
        seed=args.seed,
    )
    print(output)


if __name__ == "__main__":
    main()

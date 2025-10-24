import json
import random
from pathlib import Path

import hydra
from datasets import load_dataset
from torch.utils.data import Dataset


def get_subset_from_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    indices = random.sample(range(len(dataset)), n_samples)
    subset = dataset.select(indices)
    return subset


def save_xnli_dataset(
    lang: str,
    n_examples: int,
    n_valid_samples: int,
    seed: int,
    output_dir: Path,
):
    save_dir = output_dir / "xnli"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"{lang}.json"
    if save_path.exists():
        print("Dataset already exists. Skip building.")
        return
    
    random.seed(seed)

    n_train_samples = n_valid_samples * n_examples
    train_dataset = load_dataset("xnli", lang, split="train")
    train_subset = get_subset_from_dataset(train_dataset, n_train_samples)
    
    valid_dataset = load_dataset("xnli", lang, split="validation")
    valid_subset = get_subset_from_dataset(valid_dataset, n_valid_samples)
    
    demonstration_data = [
        {"premise": p, "hypothesis": h, "label": l}
        for p, h, l in zip(train_subset["premise"], train_subset["hypothesis"], train_subset["label"])
    ]
    target_data = [
        {"premise": p, "hypothesis": h, "label": l}
        for p, h, l in zip(valid_subset["premise"], valid_subset["hypothesis"], valid_subset["label"])
    ]

    dataset = {
        "targets": target_data,
        "demonstrations": demonstration_data,
    }
    
    with open(save_path, "w") as f:
        json.dump(dataset, f)
    
    
@hydra.main(config_path="../config", config_name="prepare_icl_dataset_config", version_base=None)
def main(cfg):
    save_xnli_dataset(
        lang=cfg.lang,
        n_examples=cfg.n_examples,
        n_valid_samples=cfg.n_samples,
        seed=cfg.seed,
        output_dir=Path(cfg.output_dir)
    )

    
if __name__ == "__main__":
    main()

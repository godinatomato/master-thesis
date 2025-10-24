import json
import random
from pathlib import Path

from datasets import load_dataset


LANGS = ["de", "en", "es", "fr", "ja", "zh"]
LABELS = {
    "positive": 2,
    "neutral": 1,
    "negative": 0,
}
N_POSITIVE_SAMPLES = 500
N_NEGATIVE_SAMPLES = 1250
SEED = 0

random.seed(SEED)

root_dir = Path.cwd()
output_dir = root_dir / "assets/Sentiment"
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("Brand24/mms")

for lang in LANGS:
    print(f"Creating dataset for {lang}...")
    
    raw_data = {}
    subset = dataset.filter(lambda row: row["language"] == lang)
    for label_name, label in LABELS.items():
        raw_data[label_name] = subset.filter(lambda row: row["label"] == label)["train"]["text"]

    for label_name, label in LABELS.items():
        processed_data = {
            "lang": lang,
            "label": label_name,
            "sentences": {
                "positive": [],
                "negative": [],
            },
        }
        for l in LABELS.keys():
            if l == label_name:
                processed_data["sentences"]["positive"].extend(random.sample(raw_data[l], N_POSITIVE_SAMPLES))
            else:
                processed_data["sentences"]["negative"].extend(random.sample(raw_data[l], N_NEGATIVE_SAMPLES))

        with open(output_dir / f"{lang}_{label_name}.json", "w") as f:
            json.dump(processed_data, f)
    
    del subset

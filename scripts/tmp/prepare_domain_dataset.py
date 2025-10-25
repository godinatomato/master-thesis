import json
import random
from pathlib import Path

from datasets import load_dataset


root_dir = Path.cwd()

DOMAINS = ["financial", "medicine", "daily", "mathematics", "literature"]
N_POSITIVE_SAMPLES = 500
N_NEGATIVE_SAMPLES = 625
SEED = 0

def main():
    random.seed(SEED)
    output_dir = root_dir / "assets/Domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data = {}
    
    fin_dataset_all = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
    fin_dataset_75 = load_dataset("takala/financial_phrasebank", "sentences_75agree", trust_remote_code=True)
    fin_texts = fin_dataset_all["train"]["sentence"]
    fin_texts.extend(fin_dataset_75["train"]["sentence"][:3000 - len(fin_texts)])
    raw_data["financial"] = fin_texts
    
    med_dataset = load_dataset("lavita/MedQuad")
    raw_data["medicine"] = med_dataset["train"]["answer"][:3000]

    daily_dataset = load_dataset("stanfordnlp/imdb")
    raw_data["daily"] = daily_dataset["train"]["text"][:3000]

    math_dataset = load_dataset("openai/gsm8k", "main")
    raw_data["mathematics"] = math_dataset["train"]["question"][:3000]

    lit_dataset = load_dataset("mintujupally/ROCStories")
    raw_data["literature"] = lit_dataset["train"]["text"]
    
    for domain in DOMAINS:
        processed_data = {
            "lang": domain,
            "sentences": {
                "positive": [],
                "negative": [],
            }
        }
        
        for d in DOMAINS:
            if d == domain:
                processed_data["sentences"]["positive"].extend(
                    random.sample(raw_data[d], N_POSITIVE_SAMPLES)
                )
            else:
                processed_data["sentences"]["negative"].extend(
                    random.sample(raw_data[d], N_NEGATIVE_SAMPLES)
                )

        with open(output_dir / f"{domain}.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

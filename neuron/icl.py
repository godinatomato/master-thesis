import random

from torch.data.utils import Dataset
from typing import Dict, Any, List


LANG_CONFIG = {
    "ja": {
        "task_prefix": "以下の前提文と仮説文の関係性を、例示を参考にして「含意」「中立」「矛盾」から一つ判断し、回答を生成してください。",
        "premise": "前提",
        "hypothesis": "仮説",
        "relation": "関係性",
        "labels": {0: "含意", 1: "中立", 2: "矛盾"},
    },
    "en": {
        "task_prefix": "Determine the relationship between the premise and the hypothesis from the following three options: entailment, neutral, or contradiction, based on the examples provided.",
        "premise": "Premise",
        "hypothesis": "Hypothesis",
        "relation": "Relationship",
        "labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
    "de": {
        "task_prefix": "Bestimmen Sie die Beziehung zwischen der Prämisse und der Hypothese anhand der folgenden drei Optionen: Implikation, Neutral oder Widerspruch, basierend auf den bereitgestellten Beispielen.",
        "premise": "Prämisse",
        "hypothesis": "Hypothese",
        "relation": "Beziehung",
        "labels": {0: "Implikation", 1: "Neutral", 2: "Widerspruch"},
    },
    "es": {
        "task_prefix": "Determine la relación entre la premisa y la hipótesis de las siguientes tres opciones: implicación, neutralidad o contradicción, basándose en los ejemplos proporcionados.",
        "premise": "Premisa",
        "hypothesis": "Hipótesis",
        "relation": "Relación",
        "labels": {0: "Implicación", 1: "Neutral", 2: "Contradicción"},
    },
    "fr": {
        "task_prefix": "Déterminez la relation entre la prémisse et l'hypothèse à partir des trois options suivantes : implication, neutre ou contradiction, en vous basant sur les exemples fournis.",
        "premise": "Prémisse",
        "hypothesis": "Hypothèse",
        "relation": "Relation",
        "labels": {0: "Implication", 1: "Neutre", 2: "Contradiction"},
    },
    "zh": {
        "task_prefix": "根据提供的例子，从以下三种选项（蕴涵、中立、矛盾）中判断前提和假设之间的关系。",
        "premise": "前提",
        "hypothesis": "假设",
        "relation": "关系",
        "labels": {0: "蕴涵", 1: "中立", 2: "矛盾"},
    },
}


def create_icl_prompt(
    test_sample_index: int, all_dataset: Dataset, num_examples: int, lang: str
) -> str:
    config = LANG_CONFIG[lang]

    target_sample = all_dataset[test_sample_index]

    all_indices = list(range(len(all_dataset)))
    available_indices = [i for i in all_indices if i != test_sample_index]

    if len(available_indices) < num_examples:
        random_indices = available_indices
    else:
        random_indices = random.sample(available_indices, num_examples)

    # prefix
    prompt = config["task_prefix"] + "\n\n"

    # demonstrations
    for i, idx in enumerate(random_indices):
        example = all_dataset[idx]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = config["labels"][example["label"]]

        prompt += f"--- {config['premise']}: {premise}\n"
        prompt += f"--- {config['hypothesis']}: {hypothesis}\n"
        prompt += f"--- {config['relation']}: {label}\n\n"

    # query
    target_premise = target_sample["premise"]
    target_hypothesis = target_sample["hypothesis"]

    prompt += "--- Query ---\n"
    prompt += f"--- {config['premise']}: {target_premise}\n"
    prompt += f"--- {config['hypothesis']}: {target_hypothesis}\n"
    prompt += f"--- {config['relation']}: "

    return prompt


def preprocess_for_icl(
    dataset: Dataset, num_examples: int, lang: str
) -> List[Dict[str, Any]]:
    new_data = []

    dataset_with_index = dataset.add_column("original_index", list(range(len(dataset))))

    for i, sample in enumerate(dataset_with_index):
        icl_prompt_text = create_icl_prompt(
            sample["original_index"], dataset_with_index, num_examples, lang
        )

        true_label_string = LANG_CONFIG["en"]["labels"][sample["label"]]

        new_sample = {
            "prompt": icl_prompt_text,
            "true_label_id": sample["label"],
            "true_label_string": true_label_string,
        }
        new_data.append(new_sample)

    return new_data

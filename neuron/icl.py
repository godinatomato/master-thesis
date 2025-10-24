import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Any, List

from torch.utils.data import Dataset

from neuron.data import ICLPromptDataset


XNLI_CONFIG = {
    "en": {
        "task_prefix": "Determine the relationship between the premise and the hypothesis from the following three options: entailment, neutral, or contradiction, based on the examples provided.",
        "premise": "Premise",
        "hypothesis": "Hypothesis",
        "relation": "Relationship",
        "query": "Query",
        "labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
    "de": {
        "task_prefix": "Bestimmen Sie die Beziehung zwischen der Prämisse und der Hypothese anhand der folgenden drei Optionen: Implikation, Neutral oder Widerspruch, basierend auf den bereitgestellten Beispielen.",
        "premise": "Prämisse",
        "hypothesis": "Hypothese",
        "relation": "Beziehung",
        "query": "Frage",
        "labels": {0: "Implikation", 1: "Neutral", 2: "Widerspruch"},
    },
    "es": {
        "task_prefix": "Determine la relación entre la premisa y la hipótesis de las siguientes tres opciones: implicación, neutralidad o contradicción, basándose en los ejemplos proporcionados.",
        "premise": "Premisa",
        "hypothesis": "Hipótesis",
        "relation": "Relación",
        "query": "pregunta",
        "labels": {0: "Implicación", 1: "Neutral", 2: "Contradicción"},
    },
    "fr": {
        "task_prefix": "Déterminez la relation entre la prémisse et l'hypothèse à partir des trois options suivantes : implication, neutre ou contradiction, en vous basant sur les exemples fournis.",
        "premise": "Prémisse",
        "hypothesis": "Hypothèse",
        "relation": "Relation",
        "query": "question",
        "labels": {0: "Implication", 1: "Neutre", 2: "Contradiction"},
    },
    "zh": {
        "task_prefix": "根据提供的例子，从以下三种选项（蕴涵、中立、矛盾）中判断前提和假设之间的关系。",
        "premise": "前提",
        "hypothesis": "假设",
        "relation": "关系",
        "query": "问题",
        "labels": {0: "蕴涵", 1: "中立", 2: "矛盾"},
    },
}


def create_xnli_prompt(
    target: str, demonstrations: List[str], config: Dict[str, Any]
) -> str:
    # prefix
    prompt = config["task_prefix"] + "\n\n"

    # demonstrations
    for d in demonstrations:
        premise = d["premise"]
        hypothesis = d["hypothesis"]
        label = config["labels"][d["label"]]

        prompt += f"--- {config['premise']}: {premise}\n"
        prompt += f"--- {config['hypothesis']}: {hypothesis}\n"
        prompt += f"--- {config['relation']}: {label}\n\n"

    # query
    target_premise = target["premise"]
    target_hypothesis = target["hypothesis"]

    prompt += f"--- {config['query']} ---\n"
    prompt += f"--- {config['premise']}: {target_premise}\n"
    prompt += f"--- {config['hypothesis']}: {target_hypothesis}\n"
    prompt += f"--- {config['relation']}: "

    return prompt


def create_icl_dataset(
    data: Dict[str, Any], dataset_name: str, n_examples: int, lang: str
) -> List[str]:
    if dataset_name == "xnli":
        targets = data["targets"]
        demonstrations = data["demonstrations"]
        config = XNLI_CONFIG[lang]
        
        prompts = [
            create_xnli_prompt(targets[i], demonstrations[i * n_examples:i * (n_examples + 1)], config)
            for i in range(len(targets))
        ]
        labels = [t["label"] for t in targets]
        dataset = ICLPromptDataset(
            prompts=prompts,
            labels=labels,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return dataset
    
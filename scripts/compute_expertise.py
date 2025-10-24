#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import hydra

from neuron.expertise import ExpertiseResult
from neuron.responses import read_responses_from_cached


def analyze_expertise_for_lang(
    lang_dir: Path,
    lang: str,
):
    cached_responses_dir = lang_dir / "responses"
    lang_exp_dir = lang_dir / "expertise"

    if ExpertiseResult.exists_in_disk(lang_exp_dir):
        print("Results found, skipping building")
        return

    try:
        responses, labels_int, response_names = read_responses_from_cached(
            cached_responses_dir, lang
        )
    except RuntimeError:
        print(f"No responses found for lang {lang}")
        return

    if not responses:
        print(f"Found response files but could not load them for lang {lang}")
        return

    lang_exp_dir.mkdir(exist_ok=True, parents=True)

    expertise_result = ExpertiseResult()
    expertise_result.build(
        responses=responses,
        labels=labels_int,
        lang=lang,
        forcing=True,
    )
    expertise_result.save(lang_exp_dir)


@hydra.main(
    config_path="../config", config_name="compute_expertise_config", version_base=None
)
def main(cfg):
    model_short_name = cfg.model_name.rstrip("/").split("/")[-1]
    output_dir = Path(cfg.output_dir)
    lang_dir = output_dir / model_short_name / cfg.data_type / cfg.lang
    
    analyze_expertise_for_lang(
        lang_dir=lang_dir,
        lang=cfg.lang,
    )


if __name__ == "__main__":
    main()

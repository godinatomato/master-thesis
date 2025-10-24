#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
from pathlib import Path

import hydra
import pandas as pd


def make_limited_expert(
    num_units: int,
    output_dir: Path,
):
    top_file_path = output_dir / f"expertise_limited_{int(num_units / 2)}_top.csv"
    bottom_file_path = output_dir / f"expertise_limited_{int(num_units / 2)}_bottom.csv"
    both_file_path = output_dir / f"expertise_limited_{num_units}_both.csv"

    if (
        top_file_path.exists()
        and bottom_file_path.exists()
        and both_file_path.exists()
    ):
        print("expertise_limited files already exist. Skip.")
        return

    df = pd.read_csv(output_dir / "expertise.csv")

    # Top N
    df_top = df.sort_values("ap", ascending=False)
    df_top = df_top.head(int(num_units / 2))

    # Bottom N
    df_bottom = df.sort_values("ap", ascending=True)
    df_bottom = df_bottom.head(int(num_units / 2))

    # Top & Bottom
    df_both = pd.concat([df_top, df_bottom], axis=0, ignore_index=True)

    # Save to files
    df_top.to_csv(top_file_path, index=False)
    df_bottom.to_csv(bottom_file_path, index=False)
    df_both.to_csv(both_file_path, index=False)


@hydra.main(
    config_path="../config",
    config_name="make_limited_expert_config",
    version_base=None,
)
def main(cfg):
    model_short_name = cfg.model_name.rstrip("/").split("/")[-1]
    output_dir = Path(cfg.output_dir) / model_short_name / cfg.data_type / cfg.lang / "expertise"
    
    make_limited_expert(
        num_units=cfg.num_units,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

import os

import hydra
import pandas as pd


def make_limited_expert(
    model_name,
    language,
    threshold,
    base_path="/groups/gcb50389/takeshi.kojima/lang_neuron/Language/",
):
    top_file = f"{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{int(threshold/2)}_top.csv"
    bottom_file = f"{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{int(threshold/2)}_bottom.csv"
    both_file = f"{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{threshold}_both.csv"

    if (
        os.path.isfile(top_file)
        and os.path.isfile(bottom_file)
        and os.path.isfile(both_file)
    ):
        print("expertise_limited files already exist. Skip.")
        return

    df = pd.read_csv(
        f"{base_path}{model_name}/sense/{language}/expertise/expertise.csv"
    )

    # Top N
    df2 = df.sort_values("ap", ascending=False)
    df2 = df2.head(int(threshold / 2))

    # Bottom N
    df3 = df.sort_values("ap", ascending=True)
    df3 = df3.head(int(threshold / 2))

    # Top & Bottom
    df4 = pd.concat([df2, df3], axis=0, ignore_index=True)

    # Save to files
    df2.to_csv(top_file, index=False)
    df3.to_csv(bottom_file, index=False)
    df4.to_csv(both_file, index=False)


@hydra.main(
    config_path="../config",
    config_name="make_limited_expertise_config",
    version_base=None,
)
def main(cfg):
    make_limited_expert(
        model_name=cfg.model_name,
        language=cfg.language,
        threshold=cfg.num_units,
        base_path=cfg.base_path,
    )


if __name__ == "__main__":
    main()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import json
import pathlib
import typing as t
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tqdm import tqdm


def _single_response_ap(
    unit_response: t.Sequence[float], labels: t.Sequence[int]
) -> float:
    return average_precision_score(y_true=labels, y_score=unit_response)


def average_precision(
    responses: t.Mapping[str, t.Sequence[float]],
    labels: t.Sequence[int],
    cpus: int = None,
) -> t.Dict[str, t.List[float]]:
    aps = {}
    cpus = cpu_count() - 1 if cpus is None else cpus
    pool = Pool(processes=cpus)
    sorted_layers = sorted(responses.keys())
    for layer in tqdm(
        sorted_layers, total=len(responses), desc=f"Av. Precision [{cpus} workers]"
    ):
        aps[layer] = pool.map(
            partial(_single_response_ap, labels=labels), responses[layer]
        )
    pool.close()
    return aps


class ExpertiseResult:

    def __init__(self) -> None:
        self.concept: str = ""
        self.concept_group: str = ""
        self.response_names: t.List[str] = []
        self._num_responses_per_layer: t.List[int] = []
        self.ap: t.Dict = {}
        # Forcing
        self.forcing: bool = True
        self.on_values_p50: t.Dict = {}
        self.on_values_p90: t.Dict = {}
        self.off_values_mean: t.Dict = {}
        self.off_values_p50: t.Dict = {}

    def build(
        self,
        concept: str,
        concept_group: str,
        responses: t.Dict,
        labels: t.Sequence[int],
        forcing: bool = True,
    ) -> None:
        self.concept = concept
        self.concept_group = concept_group
        self.response_names = sorted(list(responses.keys()))

        self.forcing = forcing
        self._num_responses_per_layer = [len(responses[r]) for r in self.response_names]

        labels = np.array(labels, dtype=int)

        self.ap = average_precision(responses, labels)

        if self.forcing:
            pos_label = 1
            for r_name, resp in responses.items():
                if np.sum(labels != pos_label) == 0:
                    print("[WARNING]: NO DATA WITH NEGATIVE LABEL FOUND")
                if np.sum(labels == pos_label) == 0:
                    print("[WARNING]: NO DATA WITH POSITIVE LABEL FOUND")
                self.off_values_mean[r_name] = np.mean(
                    resp[:, labels != pos_label], axis=1
                ).tolist()
                self.off_values_p50[r_name] = np.percentile(
                    resp[:, labels != pos_label], q=50, axis=1
                ).tolist()
                self.on_values_p50[r_name] = np.percentile(
                    resp[:, labels == pos_label], q=50, axis=1
                ).tolist()
                self.on_values_p90[r_name] = np.percentile(
                    resp[:, labels == pos_label], q=90, axis=1
                ).tolist()

    @staticmethod
    def exists_in_disk(path: pathlib.Path) -> bool:
        table_file = path / "expertise.csv"
        info_json_file = path / "expertise_info.json"
        return table_file.exists() and info_json_file.exists()

    def load(self, dir: pathlib.Path) -> None:
        df = pd.read_csv(dir / "expertise.csv")
        self.response_names = df["layer"].unique()
        self._num_responses_per_layer = []
        self.ap = {}
        self.on_values_p50 = {}
        self.on_values_p90 = {}
        self.off_values_mean = {}
        self.off_values_p50 = {}

        self.forcing = "on_p50" in df.columns

        for r_name, df_layer in df.groupby("layer", sort=False):
            self.ap[r_name] = df_layer["ap"].values
            self._num_responses_per_layer.append(len(self.ap[r_name]))
            if self.forcing:
                self.on_values_p50[r_name] = df_layer["on_p50"].values
                self.on_values_p90[r_name] = df_layer["on_p90"].values
                self.off_values_mean[r_name] = df_layer["off_mean"].values
                self.off_values_p50[r_name] = df_layer["off_p50"].values

        with (dir / "expertise_info.json").open("r") as fp:
            json_data = json.load(fp)
            self.concept = json_data["concept"]
            self.concept_group = json_data["group"]

    def export_as_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["ap"] = np.concatenate([self.ap[r] for r in self.response_names]).astype(
            np.float32
        )
        if self.forcing:
            df["off_mean"] = np.concatenate(
                [self.off_values_mean[r] for r in self.response_names]
            ).astype(np.float32)
            df["off_p50"] = np.concatenate(
                [self.off_values_p50[r] for r in self.response_names]
            ).astype(np.float32)
            df["on_p50"] = np.concatenate(
                [self.on_values_p50[r] for r in self.response_names]
            ).astype(np.float32)
            df["on_p90"] = np.concatenate(
                [self.on_values_p90[r] for r in self.response_names]
            ).astype(np.float32)
        df["layer"] = np.concatenate(
            [
                [r] * r_len
                for r, r_len in zip(self.response_names, self._num_responses_per_layer)
            ]
        )
        df["unit"] = np.concatenate(
            [range(r_len) for r_len in self._num_responses_per_layer]
        ).astype(np.uint32)
        df["uuid"] = np.arange(len(df))
        df["concept"] = self.concept
        df["group"] = self.concept_group
        return df

    def export_extra_info_json(self) -> t.Dict:
        aps_list = np.concatenate([v for v in self.ap.values()])

        info_json = {
            "concept": self.concept,
            "group": self.concept_group,
            "max_ap": float(np.max(aps_list)),
            "layer_names": self.response_names,
            "total_neurons": int(len(aps_list)),
        }

        ap_thresholds = np.linspace(0.5, 1.0, 501)

        def to_str(x: float) -> str:
            return f"{x:0.5f}"

        def unit_at_metric(
            metric: t.Sequence[float], thresholds: t.Sequence[float]
        ) -> t.Dict[str, int]:
            units_at_m = {}
            nd_vals = np.array(metric)
            for a in thresholds:
                units_at_m[to_str(a)] = int(np.sum(nd_vals > a))
            return units_at_m

        val_list = np.concatenate([v for v in self.ap.values()])
        info_json["neurons_at_ap"] = unit_at_metric(val_list, ap_thresholds)

        return info_json

    def save(self, out_dir: pathlib.Path) -> None:
        df = self.export_as_pandas()
        df.to_csv(out_dir / "expertise.csv", index=False)

        json_data = self.export_extra_info_json()
        with (out_dir / "expertise_info.json").open("w") as fp:
            json.dump(json_data, fp, indent=4)

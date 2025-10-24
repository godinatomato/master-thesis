import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import pandas as pd

from neuron.plot import plot_neuron_counts, plot_neuron_counts_by_attn_ffn


root_dir = Path.cwd()

MODEL_NAMES = ["xglm-564M", "bloom-1b7"]
LANGS = ["de", "en", "es", "fr", "ja", "zh"]

N_LAYERS = {
    "xglm-564M": 24,
    "bloom-1b7": 24,
}
MODULE_NAMES = {
    "xglm-564M": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"],
    "bloom-1b7": ["self_attention.query_key_value", "self_attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
}
ATTN_MODULES = {
    "xglm-564M": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"],
    "bloom-1b7": ["self_attention.query_key_value", "self_attention.dense"],
}
FFN_MODULES = {
    "xglm-564M": ["fc1", "fc2"],
    "bloom-1b7": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
}

def get_layer_idx(layer_str: str) -> int:
    return int(layer_str.replace(":0", "").split(".")[2])

def get_module_basename(layer_str: str, keys: list) -> str:
    for key in keys:
        if key in layer_str:
            return key


for model_name in MODEL_NAMES:
    expertise_dir = root_dir / f"output/{model_name}/Language"
    output_dir = root_dir / f"output/{model_name}/Figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_keys = [i for i in range(N_LAYERS[model_name])]
    module_keys = MODULE_NAMES[model_name]
    attn_keys = ATTN_MODULES[model_name]
    ffn_keys = FFN_MODULES[model_name]
    
    layer_neuron_count = {}
    module_neuron_count = {}
    attn_count = [0 for _ in range(N_LAYERS[model_name])]
    ffn_count = [0 for _ in range(N_LAYERS[model_name])]

    for lang in LANGS:
        expert_df = pd.read_csv(expertise_dir / f"{lang}/expertise/expertise_limited_2000_both.csv")
        expert_df["layer_idx"] = expert_df["layer"].apply(get_layer_idx)
        expert_df["module_basename"] = expert_df["layer"].apply(lambda x: get_module_basename(x, module_keys))

        layer_idx_count = expert_df["layer_idx"].value_counts()
        module_basename_count = expert_df["module_basename"].value_counts()

        layer_neuron_count[lang] = []
        for key in layer_keys:
            try:
                layer_neuron_count[lang].append(layer_idx_count[key])
            except KeyError:
                layer_neuron_count[lang].append(0)
                
        module_neuron_count[lang] = []
        for key in module_keys:
            try:
                module_neuron_count[lang].append(module_basename_count[key])
            except KeyError:
                module_neuron_count[lang].append(0)
                
        for key in layer_keys:
            df_ = expert_df[expert_df["layer_idx"] == key]
            for name in df_["module_basename"]:
                if name in attn_keys:
                    attn_count[key] += 1
                elif name in ffn_keys:
                    ffn_count[key] += 1

    plot_neuron_counts(layer_neuron_count, layer_keys, output_dir / "num_neurons_per_layer.png")
    plot_neuron_counts(module_neuron_count, module_keys, output_dir / "num_neurons_per_module.png")
    plot_neuron_counts_by_attn_ffn(attn_count, ffn_count, layer_keys, output_dir / "num_neurons_by_attn_ffn.png")
    
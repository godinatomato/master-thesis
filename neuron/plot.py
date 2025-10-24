from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_neuron_counts(neuron_count_dict: dict, x_labels: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cmap = cm.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(neuron_count_dict))] 
    x = [i for i in range(len(list(neuron_count_dict.values())[0]))]
    
    max_n = 0
    for i, lang in enumerate(neuron_count_dict.keys()):
        ax.plot(x, neuron_count_dict[lang], color=colors[i], label=lang)
        
        if max(neuron_count_dict[lang]) > max_n:
            max_n = max(neuron_count_dict[lang])
        
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max_n + 20)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Number of neurons")
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(save_path)


def plot_neuron_counts_by_attn_ffn(attn_counts: list, ffn_counts: list, x_labels: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(12,7))
    cmap = cm.get_cmap("tab10")
    colors = [cmap(i) for i in range(2)] 
    x = [i for i in range(len(attn_counts))]
    
    max_n = max(max(attn_counts), max(ffn_counts))
    ax.plot(x, attn_counts, color=colors[0], label="self-attention")
    ax.plot(x, ffn_counts, color=colors[1], label="ffn")
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max_n + 20)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Number of neurons")
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(save_path)
    
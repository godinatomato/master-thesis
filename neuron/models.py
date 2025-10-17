#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import typing as t
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoConfig,
    XGLMForCausalLM,
    BloomForCausalLM,
)

MODEL_INPUT_FIELDS = ["input_ids", "attention_mask"]
LABELS_FIELD = "labels"


@dataclass(frozen=True)
class ResponseInfo:
    name: str
    dtype: np.dtype
    shape: t.Tuple[t.Optional[int], ...]
    layer: "ResponseInfo.Layer"

    @dataclass(frozen=True)
    class Layer:
        name: str
        kind: str


class TorchModel:
    def __init__(
        self,
        module: torch.nn.Module,
        input_size: t.Mapping[str, t.Tuple],
        input_type: t.Mapping[str, torch.dtype],
        name: str,
        device: str = None,
    ) -> None:
        self.name = name
        self._pytorch_module = module.eval()

        if set(input_size.keys()) != set(input_type.keys()):
            raise RuntimeError(
                "Model input keys for size and type must be the same."
                f"{input_size.keys()} != {input_type.keys()}."
            )

        self._forward_hooks: t.List[RemovableHandle] = []
        self._input_size: t.Mapping[str, t.Tuple] = input_size
        self._input_types: t.Mapping[str, torch.dtype] = input_type
        self._response_infos: t.List[ResponseInfo] = []
        self._compute_response_infos()

    @property
    def module(self) -> nn.Module:
        return self._pytorch_module

    def _compute_response_infos(self) -> None:
        def hook(module_name, module, module_input, module_output) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            outputs = (
                module_output
                if isinstance(module_output, (list, tuple))
                else [module_output]
            )

            for output_idx, o in enumerate(outputs):
                if o is None or type(o) is not torch.Tensor:
                    continue

                response_name = "{}:{}".format(module_name, output_idx)
                ri = ResponseInfo(
                    name=response_name,
                    dtype=o.dtype,
                    shape=(o.size())[1:],
                    layer=ResponseInfo.Layer(
                        name=module_name,
                        kind=class_name,
                    ),
                )

                self._response_infos.append(ri)

        hooks = []
        for module_name, module in self._pytorch_module.named_modules():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                continue

            if module == self._pytorch_module:
                continue

            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        self._perform_dummy_inference()

        for h in hooks:
            h.remove()

    def _perform_dummy_inference(self) -> None:
        arg_names = list(self._input_types.keys())
        fixed_shaped_list: t.List[int] = [2]

        x = {
            input_name: torch.rand(
                tuple(fixed_shaped_list + [*self._input_size[input_name]])
            )
            .type(self._input_types[input_name])
            .to(torch.device("cuda", 0))
            for input_name in arg_names
        }

        with torch.no_grad():  # type: ignore
            self._pytorch_module(**x)

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        return self._response_infos

    def _set_units_hook_wrapper(
        self,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool,
    ) -> t.Callable:
        assert len(units) == len(
            values
        ), "The number of values must match the number of units."

        def forward_hook(module, input, output):
            if only_last_token:
                output[:, -1, units] = values.to(output.device)
            else:
                output[:, :, units] = values.to(output.device)
            return output

        return forward_hook

    def set_units_in_layer(
        self,
        layer_name: str,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool = False,
    ) -> None:
        layer_name = layer_name.replace(":0", "")
        for iterated_module_name, layer in self._pytorch_module.named_modules():
            if iterated_module_name == layer_name:
                handle = layer.register_forward_hook(
                    self._set_units_hook_wrapper(
                        units=units,
                        values=values,
                        only_last_token=only_last_token,
                    )
                )
                self._forward_hooks.append(handle)

    def restore_units(self):
        for h in self._forward_hooks:
            h.remove()
        self._forward_hooks.clear()

    def run_inference(
        self, inputs: t.Mapping[str, torch.Tensor], outputs: t.AbstractSet[str]
    ) -> t.Dict[str, np.ndarray]:
        a_key = list(inputs.keys())[0]
        torch_inputs: t.MutableMapping[str, torch.Tensor] = {}
        if isinstance(inputs[a_key][0], torch.Tensor):
            torch_inputs = {k: v.to(torch.device("cuda", 0)) for k, v in inputs.items()}

        response_dict: t.Dict[str, t.Any] = {}

        def hook(module_name, module, module_input, module_output) -> None:  # type: ignore
            module_output = (
                module_output
                if isinstance(module_output, (list, tuple))
                else [module_output]
            )

            for output_idx, o in enumerate(module_output):
                response_name = "{}:{}".format(module_name, output_idx)
                if response_name in outputs:
                    if o.dtype == torch.float32:
                        tensor = o.detach().cpu().numpy()
                    else:
                        tensor = o.detach().cpu().to(dtype=torch.float32).numpy()

                    response_dict[response_name] = tensor

        hooks = []

        for module_name, module in self._pytorch_module.named_modules():

            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                continue

            if module == self._pytorch_module:
                continue

            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        with torch.no_grad():  # type: ignore
            self._pytorch_module(**torch_inputs)

        for h in hooks:
            h.remove()

        return response_dict


class PytorchTransformersModel(TorchModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: t.Optional[pathlib.Path],
        seq_len: int,
        device: str = None,
    ) -> None:
        torch_model = transformers_class_from_name(model_name, cache_dir=cache_dir)
        super().__init__(
            module=torch_model,
            input_size={input_name: (seq_len,) for input_name in MODEL_INPUT_FIELDS},
            input_type={input_name: torch.long for input_name in MODEL_INPUT_FIELDS},
            name=model_name,
            device=device,
        )


def transformers_model_name_to_family(model_name: str) -> str:
    if model_name.startswith("bert"):
        return "bert"
    elif model_name.startswith("openai"):
        return "openai"
    elif model_name.startswith("gpt2"):
        return "gpt2"
    elif model_name.startswith("xlnet"):
        return "xlnet"
    elif model_name.startswith("xlm"):
        return "xlm"
    elif model_name.startswith("roberta"):
        return "roberta"
    elif model_name.startswith("distilbert"):
        return "distilbert"
    elif model_name.startswith("ctrl"):
        return "ctrl"
    elif "bloom" in model_name:
        return "bloom"
    elif "Llama-2" in model_name:
        return "Llama-2"
    elif "llama" in model_name:
        return "llama"
    elif "falcon" in model_name:
        return "falcon"
    elif "xglm" in model_name:
        return "xglm"
    else:
        raise NotImplementedError(f"Model name to type not considered: {model_name}")


def transformers_class_from_name(
    model_name: str,
    cache_dir: t.Optional[pathlib.Path] = None,
    rand_weights: bool = False,
) -> nn.Module:
    try:
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
        max_memory = f"{free_in_GB-2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        if rand_weights:
            config = AutoConfig.from_pretrained(model_name)
            m = AutoModelForPreTraining.from_config(config)
        else:
            if "Llama-2" in model_name:
                m = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    device_map="auto",
                )
            elif "xglm" in model_name:
                m = XGLMForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                ).to("cuda")
            elif "bloom" in model_name:
                m = BloomForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                ).to("cuda")
            else:
                raise ValueError("error! model_name is not properly defined.")

            try:
                print(m.hf_device_map)
            except:
                pass

    except OSError:
        raise NotImplementedError(f"Model {model_name} could not be loaded.")
    assert m is not None
    return m


def get_layer_regex(model_name: str) -> t.Optional[t.List[str]]:
    family = transformers_model_name_to_family(model_name)
    layer_types = None
    if family == "gpt2":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_attn",
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_proj",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_fc",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_proj",
        ]
    elif family == "bloom":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.query_key_value",
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.dense",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_h_to_4h",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_4h_to_h",
        ]
    elif family in ["llama", "Llama-2"]:
        layer_types = [
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.o_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.gate_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.down_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.up_proj",
        ]
    if family == "xglm":
        layer_types = [
            "model.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.out_proj",
            "model.layers.([0-9]|[0-9][0-9]).fc1",
            "model.layers.([0-9]|[0-9][0-9]).fc2",
        ]
    # Extend to other model families here if needed
    return layer_types


def _print_responses(ri: t.List[ResponseInfo]) -> None:
    assert len(ri), "No responses selected"
    print(f"Found {len(ri)} responses from model.")
    for r in ri:
        print("\t", r.name, r.shape)


def _collect_responses_info_for_model(
    model: TorchModel, model_family: str
) -> t.List[ResponseInfo]:
    mapping = {
        "gpt2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "bloom": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "llama": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "Llama-2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "falcon": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "xglm": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
    }
    return mapping[model_family]


def collect_responses_info(model_name: str, model: TorchModel) -> t.List[ResponseInfo]:
    family = transformers_model_name_to_family(model_name)
    responses_info = _collect_responses_info_for_model(model, family)
    # _print_responses(responses_info)
    return responses_info


def concatenate_responses(
    responses: t.Dict[str, np.ndarray],
    response_fields: t.Set[str],
    output_field: str,
    axis: int,
) -> t.Dict[str, np.ndarray]:
    data = [tensor for field, tensor in responses.items() if field in response_fields]
    responses[output_field] = np.concatenate(data, axis=axis)
    for field in response_fields:
        del responses[field]
    return responses


def pool_responses(
    responses: t.Dict[str, np.ndarray],
    response_fields: t.Optional[t.Set[str]],
    axis: t.Tuple[int],
    pooling_type: str = "max",
) -> t.Dict[str, np.ndarray]:
    assert pooling_type in ["mean", "sum", "max", "median", "min"]
    pooler_fn = getattr(np, pooling_type)
    fields = response_fields or responses.keys()
    for field in fields:
        responses[field] = pooler_fn(responses[field], axis=axis)
    return responses


def processors_per_model(model: TorchModel) -> t.List[t.Callable]:
    pool_args: t.List[t.Dict] = [
        dict(response_fields=None, axis=1, pooling_type="mean")
    ]
    process_fns: t.List[t.Callable] = []
    process_fns += [partial(pool_responses, **args) for args in pool_args]
    return process_fns

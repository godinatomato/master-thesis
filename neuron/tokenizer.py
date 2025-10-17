#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer


class PytorchTransformersTokenizer:
    def __init__(self, model_name: str, cache_dir: pathlib.Path):
        print(f"Creating tokenizer {model_name} from {cache_dir}")
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, cache_dir=cache_dir
        )
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def pad_indexed_tokens(
        self, indexed_tokens: List[int], min_num_tokens: int
    ) -> List[int]:
        assert min_num_tokens is not None
        assert min_num_tokens > 0
        pad_token_id: int = self._tokenizer.pad_token_id

        num_effective_tokens = len(indexed_tokens)
        pad_tokens: int = max(min_num_tokens - num_effective_tokens, 0)
        return indexed_tokens + [pad_token_id] * pad_tokens

    def pre_process_sequence(
        self, text: str, min_num_tokens: int = None
    ) -> Dict[str, List]:
        indexed_tokens: List[int] = self._tokenizer.encode(text)

        num_effective_tokens = len(indexed_tokens)
        if min_num_tokens is not None:
            indexed_tokens = self.pad_indexed_tokens(indexed_tokens, min_num_tokens)

        attention_mask: List[int] = [1] * num_effective_tokens + [0] * (
            len(indexed_tokens) - num_effective_tokens
        )

        named_data = {"input_ids": indexed_tokens, "attention_mask": attention_mask}
        return named_data

    def preprocess_dataset(
        self, sentence_list: List[str], min_num_tokens: int = None
    ) -> Dict[str, List]:
        named_data: Dict[str, List] = defaultdict(list)
        for seq in tqdm(sentence_list, desc="Preprocessing", total=len(sentence_list)):
            if type(seq) != str:
                continue
            named_data_seq = self.pre_process_sequence(
                text=seq,
                min_num_tokens=min_num_tokens,
            )

            for k, v in named_data_seq.items():
                named_data[k].append(v)
        return named_data

    @property
    def model_name(self) -> str:
        return self._model_name

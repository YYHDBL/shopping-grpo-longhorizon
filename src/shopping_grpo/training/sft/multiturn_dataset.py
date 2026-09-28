# 购物轨迹多轮 SFT 数据集（veRL custom_cls 注入用）
# veRL MultiTurnSFTDataset 逐条渲染消息，Qwen 模板不接受单独的 tool 消息；
# 本类对每个 assistant 段做"前缀(+generation prompt) vs 含该段"两次渲染差分，
# 渲染输入永远是合法对话前缀。训练入口：scripts/train_sft.sh 的
# data.custom_cls.path=shopping_grpo.training.sft.multiturn_dataset
from __future__ import annotations

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

import pandas as pd


class ShoppingMultiTurnSFTDataset(Dataset):
    def __init__(
        self,
        parquet_files,
        tokenizer,
        config: DictConfig | None = None,
        processor=None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.tokenizer = tokenizer
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        assert self.truncation in {"error", "left", "right"}
        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools")
        self.shuffle = config.get("shuffle", False)
        self.max_samples = max_samples
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(path) for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if self.shuffle:
            self.dataframe = self.dataframe.sample(
                frac=1, random_state=config.get("seed"))
        if self.max_samples and self.max_samples > 0:
            self.dataframe = self.dataframe[: self.max_samples]

    def __len__(self):
        return len(self.dataframe)

    def _render_ids(self, messages, tools, add_generation_prompt):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @staticmethod
    def _maps_to_dicts(messages):
        # arrow 的 map 类型读回是 (key, value) 元组列表，模板要求 dict
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, list):
                    function["arguments"] = dict(arguments)
        return messages

    def __getitem__(self, index):
        from verl.utils.py_functional import convert_nested_value_to_list_recursive
        row = self.dataframe.iloc[index]
        messages = self._maps_to_dicts(
            convert_nested_value_to_list_recursive(row[self.messages_key]))
        tools = None
        if self.tools_key in self.dataframe.columns:
            tools = convert_nested_value_to_list_recursive(row[self.tools_key])
            for tool in tools or []:
                parameters = (tool.get("function") or {}).get("parameters") or {}
                properties = parameters.get("properties")
                if isinstance(properties, list):
                    parameters["properties"] = dict(properties)

        full_ids = self._render_ids(messages, tools, add_generation_prompt=False)
        # 差分：对每个 assistant 段计算 [前缀+生成提示) 与 [含该段) 的 token 差
        segment_mask = torch.zeros(len(full_ids), dtype=torch.long)
        rendered_prefix = []
        for i, message in enumerate(messages):
            rendered_prefix.append(message)
            if message.get("role") != "assistant":
                continue
            prefix_ids = self._render_ids(
                rendered_prefix[:-1], tools, add_generation_prompt=True)
            with_assistant_ids = self._render_ids(
                rendered_prefix, tools, add_generation_prompt=False)
            start, end = len(prefix_ids), len(with_assistant_ids)
            if end > len(full_ids) or with_assistant_ids != full_ids[:end]:
                raise ValueError(
                    f"token boundary mismatch at message {i} of row {index}; "
                    "chat template is not prefix-consistent")
            segment_mask[start:end] = 1
        if len(full_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"{len(full_ids)=} is larger than {self.max_length=}")
            if self.truncation == "right":
                full_ids = full_ids[: self.max_length]
                segment_mask = segment_mask[: self.max_length]
            else:
                full_ids = full_ids[-self.max_length:]
                segment_mask = segment_mask[-self.max_length:]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "position_ids": torch.arange(len(full_ids), dtype=torch.long),
            "loss_mask": segment_mask,
        }

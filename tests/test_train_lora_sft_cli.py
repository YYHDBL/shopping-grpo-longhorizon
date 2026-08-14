"""验证 LoRA SFT 入口的关键默认值。"""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_lora_sft import (
    DEFAULT_TARGET_MODULES,
    _load_preprocessing_components,
    _loss_only_eval_trainer_class,
    _model_load_kwargs,
    _prepare_model_for_training,
    _resolve_dtype,
    _swanlab_config,
    _curriculum_task_ids,
    parse_args,
)


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _FakeAutoConfig:
    @staticmethod
    def from_pretrained(model_name, trust_remote_code):
        del model_name, trust_remote_code
        return _FakeConfig("qwen3_5")


class _FakeTokenizer:
    pass


class _FakeAutoTokenizer:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeTokenizer()


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeAutoProcessor:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeProcessor()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"use_cache": True})()
        self.input_grads_enabled = False

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class _FakeTrainer:
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        return model, inputs, prediction_loss_only, ignore_keys


class TrainLoraSftCliTest(unittest.TestCase):
    def test_curriculum_manifest_expands_cumulative_stage_ids(self):
        manifest = {
            "stages": {"b": {"buckets": ["foundation", "constraints"]}},
            "buckets": {
                "foundation": {
                    "train_task_ids": [1],
                    "validation_task_ids": [2],
                },
                "constraints": {
                    "train_task_ids": [3],
                    "validation_task_ids": [4],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(_curriculum_task_ids(path, "b", "train"), {1, 3})
            self.assertEqual(_curriculum_task_ids(path, "b", "validation"), {2, 4})

    def test_defaults_are_suitable_for_small_qwen_lora_warmup(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "/models/Qwen3.5-0.8B",
                "--train",
                "outputs/batch/train.jsonl",
                "--output",
                "checkpoints/qwen-shopping-lora",
                "--curriculum-manifest",
                "data/sft_curriculum/manifest.json",
                "--curriculum-stage",
                "b",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.model, "/models/Qwen3.5-0.8B")
        self.assertEqual(args.train, Path("outputs/batch/train.jsonl"))
        self.assertEqual(args.max_length, 24576)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.save_total_limit, 3)
        self.assertEqual(args.dtype, "auto")
        self.assertFalse(args.bf16)
        self.assertFalse(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-grpo")
        self.assertEqual(
            args.curriculum_manifest,
            Path("data/sft_curriculum/manifest.json"),
        )
        self.assertEqual(args.curriculum_stage, "b")

    def test_swanlab_flags_are_opt_in_and_keep_a_stable_run_name(self):
        """国内监控必须显式启用，且实验名可由调用方固定以便对比。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--swanlab",
                "--swanlab-project",
                "shopping-agent",
                "--swanlab-run-name",
                "qwen35-2b-lora-v1",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-agent")
        self.assertEqual(args.swanlab_run_name, "qwen35-2b-lora-v1")

    def test_swanlab_config_returns_a_stable_default_run_name(self):
        """SwanLab 由 main 中的显式 init 配置；此处只验证纯配置函数。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/run/adapter",
                "--swanlab",
                "--swanlab-mode",
                "local",
            ],
        ):
            args = parse_args()

        with patch.dict(sys.modules, {"swanlab": object()}), patch.dict(os.environ, {}, clear=True):
            report_to, run_name = _swanlab_config(args)
            self.assertEqual(report_to, "swanlab")
            self.assertIn("lora-r16", run_name)

    def test_qwen35_uses_processor_template_and_underlying_tokenizer(self):
        """Qwen3.5 是多模态检查点，不能只加载 AutoTokenizer。"""
        tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
            "Qwen/Qwen3.5-2B",
            auto_config=_FakeAutoConfig,
            auto_tokenizer=_FakeAutoTokenizer,
            auto_processor=_FakeAutoProcessor,
        )

        self.assertTrue(is_multimodal)
        self.assertIs(chat_template.tokenizer, tokenizer)
        self.assertTrue(_FakeAutoProcessor.called)
        self.assertFalse(_FakeAutoTokenizer.called)

    def test_default_lora_targets_cover_qwen35_linear_attention_layers(self):
        """Qwen3.5 的 3/4 层是 Gated DeltaNet，不能只训练少数全注意力层。"""
        self.assertIn("in_proj_qkv", DEFAULT_TARGET_MODULES)
        self.assertIn("out_proj", DEFAULT_TARGET_MODULES)

    def test_acceleration_flags_build_liger_sdpa_and_standard_qlora_configuration(self):
        """D 组必须在 C 的 SDPA 基础上显式添加 NF4 QLoRA，而非传递未验证的 dict。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "Qwen/Qwen3.5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--liger-kernel",
                "--attention-implementation", "sdpa",
                "--qlora",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(args, dtype="bf16", bits_and_bytes_config=_FakeBitsAndBytesConfig)
        self.assertTrue(args.liger_kernel)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertIsInstance(kwargs["quantization_config"], _FakeBitsAndBytesConfig)
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_compute_dtype"], "bf16")

    def test_dtype_auto_prefers_bf16_then_fp16_and_cpu_fp32(self):
        class FakeCuda:
            available = True
            bf16_supported = True

            @classmethod
            def is_available(cls):
                return cls.available

            @classmethod
            def is_bf16_supported(cls):
                return cls.bf16_supported

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": FakeCuda,
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            },
        )
        args = type("Args", (), {"dtype": "auto", "bf16": False})()

        self.assertEqual(_resolve_dtype(args, fake_torch), ("bf16", "bf16"))
        FakeCuda.bf16_supported = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp16", "fp16"))
        FakeCuda.available = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp32", "fp32"))

    def test_model_revision_is_forwarded_to_loader(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--revision",
                "frozen-revision",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(
            args,
            dtype="bf16",
            bits_and_bytes_config=_FakeBitsAndBytesConfig,
        )
        self.assertEqual(kwargs["revision"], "frozen-revision")

    def test_qlora_prepares_model_before_lora_and_keeps_gradient_checkpointing_compatible(self):
        """量化基座必须先做 PEFT 标准预处理，再由后续 LoRA 注入 adapter。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "Qwen/Qwen3.5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--qlora",
                "--gradient-checkpointing",
            ],
        ):
            args = parse_args()
        model = _FakeModel()
        prepared = _FakeModel()
        prepare = unittest.mock.MagicMock(return_value=prepared)

        result = _prepare_model_for_training(model, args, prepare)

        self.assertIs(result, prepared)
        prepare.assert_called_once_with(model, use_gradient_checkpointing=True)
        self.assertFalse(result.config.use_cache)

    def test_liger_qwen_loss_only_eval_skips_full_vocabulary_logits(self):
        """纯 eval_loss 必须显式走 Liger fused loss，避免 20K×248K logits。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=True,
        )
        original_inputs = {"input_ids": [1, 2], "labels": [1, 2]}

        _, forwarded_inputs, prediction_loss_only, ignore_keys = trainer_class().prediction_step(
            model="model",
            inputs=original_inputs,
            prediction_loss_only=True,
            ignore_keys=["past_key_values"],
        )

        self.assertTrue(forwarded_inputs["skip_logits"])
        self.assertNotIn("skip_logits", original_inputs)
        self.assertTrue(prediction_loss_only)
        self.assertEqual(ignore_keys, ["past_key_values"])

    def test_eval_that_needs_predictions_does_not_skip_logits(self):
        """若调用方需要 predictions/metrics，则仍必须返回真实 logits。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=True,
        )

        _, forwarded_inputs, _, _ = trainer_class().prediction_step(
            model="model",
            inputs={"input_ids": [1, 2], "labels": [1, 2]},
            prediction_loss_only=False,
        )

        self.assertNotIn("skip_logits", forwarded_inputs)

    def test_non_liger_training_keeps_standard_eval_forward(self):
        """未启用兼容的 Liger Qwen forward 时不能传入专用参数。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=False,
        )

        _, forwarded_inputs, _, _ = trainer_class().prediction_step(
            model="model",
            inputs={"input_ids": [1, 2], "labels": [1, 2]},
            prediction_loss_only=True,
        )

        self.assertNotIn("skip_logits", forwarded_inputs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

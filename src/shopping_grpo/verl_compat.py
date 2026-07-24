"""veRL 0.8 的窄范围运行时兼容。"""


def install_torch_padding_fallback():
    """用 veRL 自带的纯 PyTorch 实现替代 FlashAttention padding 工具。"""
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions

import os


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    # Ray 分配 GPU 前执行此 hook；设备可用性查询必须避免初始化 CUDA。
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions

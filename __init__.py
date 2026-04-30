import os
import subprocess
import sys
import hashlib

# 插件版本号
__version__ = "2.0.3"

# 在启动时打印版本信息
print(f"[Comfyui-MyApi] 插件版本: {__version__}")


def _ensure_requirements_installed():
    """
    在首次加载节点时自动安装依赖。
    若 requirements.txt 或 pip 执行失败，仅打印警告，继续加载，避免阻断 ComfyUI。
    可以通过环境变量 COMFYUI_MYAPI_SKIP_AUTO_INSTALL=1 跳过自动安装。
    """
    if os.environ.get("COMFYUI_MYAPI_SKIP_AUTO_INSTALL") == "1":
        print("[Comfyui-MyApi] 跳过依赖自动安装（检测到环境变量 COMFYUI_MYAPI_SKIP_AUTO_INSTALL=1）")
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    requirements_path = os.path.join(base_dir, "requirements.txt")
    marker_path = os.path.join(base_dir, ".requirements_installed")

    if not os.path.exists(requirements_path):
        return

    try:
        with open(requirements_path, "rb") as req_file:
            requirements_hash = hashlib.sha256(req_file.read()).hexdigest()
    except Exception:
        requirements_hash = ""

    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as marker:
                if marker.read().strip() == requirements_hash:
                    return
        except Exception:
            pass

    try:
        print("[Comfyui-MyApi] 正在自动安装依赖...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        )
        with open(marker_path, "w", encoding="utf-8") as marker:
            marker.write(requirements_hash)
        print("[Comfyui-MyApi] 依赖安装完成。")
    except Exception as exc:
        print(f"[Comfyui-MyApi] 自动安装依赖失败：{exc}")
        print("请手动执行：pip install -r requirements.txt")


_ensure_requirements_installed()

from .bizyair_nanobanana_node import NODE_CLASS_MAPPINGS as NANOBANANA_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NANOBANANA_NODE_DISPLAY_NAME_MAPPINGS
from .bizyair_gpt_image2_node import NODE_CLASS_MAPPINGS as GPT_IMAGE2_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GPT_IMAGE2_NODE_DISPLAY_NAME_MAPPINGS
from .deepseek_v4_node import NODE_CLASS_MAPPINGS as DEEPSEEK_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DEEPSEEK_NODE_DISPLAY_NAME_MAPPINGS
from .doubao_node import NODE_CLASS_MAPPINGS as DOUBAO_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DOUBAO_NODE_DISPLAY_NAME_MAPPINGS
from .doubao_seedream5_node import NODE_CLASS_MAPPINGS as DOUBAO_SEEDREAM5_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DOUBAO_SEEDREAM5_NODE_DISPLAY_NAME_MAPPINGS
from .doubao_seed_translation_node import NODE_CLASS_MAPPINGS as DOUBAO_SEED_TRANSLATION_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DOUBAO_SEED_TRANSLATION_NODE_DISPLAY_NAME_MAPPINGS
from .autodl_api_node import NODE_CLASS_MAPPINGS as AUTODL_API_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as AUTODL_API_NODE_DISPLAY_NAME_MAPPINGS
from .autodl_nano_banana_image_node import NODE_CLASS_MAPPINGS as AUTODL_NANO_IMAGE_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as AUTODL_NANO_IMAGE_NODE_DISPLAY_NAME_MAPPINGS
from .text_segmentation_node import NODE_CLASS_MAPPINGS as TEXT_SEGMENTATION_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TEXT_SEGMENTATION_NODE_DISPLAY_NAME_MAPPINGS
from .ssh_file_upload_node import NODE_CLASS_MAPPINGS as SSH_FILE_UPLOAD_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SSH_FILE_UPLOAD_NODE_DISPLAY_NAME_MAPPINGS
# 合并所有节点映射（按类别分组排序）
NODE_CLASS_MAPPINGS = {}
# BizyAir 系列
NODE_CLASS_MAPPINGS.update(NANOBANANA_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(GPT_IMAGE2_NODE_CLASS_MAPPINGS)
# DeepSeek 系列
NODE_CLASS_MAPPINGS.update(DEEPSEEK_NODE_CLASS_MAPPINGS)
# 豆包系列
NODE_CLASS_MAPPINGS.update(DOUBAO_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DOUBAO_SEEDREAM5_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DOUBAO_SEED_TRANSLATION_NODE_CLASS_MAPPINGS)
# AutodL
NODE_CLASS_MAPPINGS.update(AUTODL_API_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(AUTODL_NANO_IMAGE_NODE_CLASS_MAPPINGS)
# 文本处理系列
NODE_CLASS_MAPPINGS.update(TEXT_SEGMENTATION_NODE_CLASS_MAPPINGS)
# 文件处理系列
NODE_CLASS_MAPPINGS.update(SSH_FILE_UPLOAD_NODE_CLASS_MAPPINGS)

# 初始化显示名称映射并合并（按类别分组排序）
NODE_DISPLAY_NAME_MAPPINGS = {}
# BizyAir 系列
NODE_DISPLAY_NAME_MAPPINGS.update(NANOBANANA_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GPT_IMAGE2_NODE_DISPLAY_NAME_MAPPINGS)
# DeepSeek 系列
NODE_DISPLAY_NAME_MAPPINGS.update(DEEPSEEK_NODE_DISPLAY_NAME_MAPPINGS)
# 豆包系列
NODE_DISPLAY_NAME_MAPPINGS.update(DOUBAO_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DOUBAO_SEEDREAM5_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DOUBAO_SEED_TRANSLATION_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AUTODL_API_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AUTODL_NANO_IMAGE_NODE_DISPLAY_NAME_MAPPINGS)
# 文本处理系列
NODE_DISPLAY_NAME_MAPPINGS.update(TEXT_SEGMENTATION_NODE_DISPLAY_NAME_MAPPINGS)
# 文件处理系列
NODE_DISPLAY_NAME_MAPPINGS.update(SSH_FILE_UPLOAD_NODE_DISPLAY_NAME_MAPPINGS)

# 暴露前端扩展目录，加载自定义 JS（用于动态图片输入端口等交互）
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY', '__version__']

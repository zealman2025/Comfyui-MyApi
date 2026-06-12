import traceback
from typing import Dict, Any, List, Optional

from .bizyair_common import (
    call_modelzoo_task_async,
    decode_image_from_url_async,
    upload_comfyui_images_to_oss_async,
    calculate_official_wh,
    NANOBANANA2_STANDARD_ASPECT_RATIOS,
    normalize_aspect_ratio,
    validate_prompt_length,
)
from .myapi_keys import (
    PROVIDER_BIZYAIR,
    missing_api_key_message,
    resolve_api_key,
    USE_NODE_API_KEY_INPUT,
)

O2_BASE_T2I = "bza-image-o2-base/text-to-image"
O2_BASE_I2I = "bza-image-o2-base/image-to-image"
O2_OFFICIAL_T2I = "bza-image-o2-official/text-to-image"
O2_OFFICIAL_I2I = "bza-image-o2-official/image-to-image"

O2_BASE_I2I_MAX = 10
O2_OFFICIAL_I2I_MAX = 16

GPT_IMAGE_2_ASPECT_RATIOS = NANOBANANA2_STANDARD_ASPECT_RATIOS


def _require_bizyair_key(api_key: str, use_node_api_key: bool) -> str:
    key = resolve_api_key(PROVIDER_BIZYAIR, api_key, use_node_api_key)
    if not key:
        raise Exception(missing_api_key_message(PROVIDER_BIZYAIR))
    return key


async def _run_task_async(api_key: str, endpoint: str, payload: dict, log_prefix: str):
    images_out, request_id = await call_modelzoo_task_async(api_key, endpoint, payload, log_prefix)
    if not images_out:
        raise Exception("接口调用成功但未返回任何生成图片地址")
    output = await decode_image_from_url_async(images_out[0])
    return output, request_id


def _collect_images(inputcount: int, max_count: int, slots: List[Optional[object]]) -> List[object]:
    count = max(1, min(int(inputcount), max_count))
    valid = [img for img in slots[:count] if img is not None]
    if not valid:
        raise Exception("图生图至少需要连接一张参考图 (image)")
    return valid


def _resolve_o2_base_resolution(resolution: str, aspect_ratio: str, log_prefix: str) -> str:
    if resolution == "4K" and aspect_ratio == "1:1":
        print(f"[{log_prefix}] 4K+1:1 组合不受支持，已自动降为 2K。")
        return "2K"
    return resolution


class BizyAirGPTImage2ThirdPartyT2INode:
    """bza-image-o2-base/text-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(self, api_key, use_node_api_key, prompt, aspect_ratio, resolution):
        log_prefix = "BizyAirGPTImage2ThirdPartyT2I"
        key = _require_bizyair_key(api_key, use_node_api_key)
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        ratio = normalize_aspect_ratio(aspect_ratio, GPT_IMAGE_2_ASPECT_RATIOS)
        used_resolution = _resolve_o2_base_resolution(resolution, ratio, log_prefix)

        payload = {
            "prompt": clean_prompt,
            "aspect_ratio": ratio,
            "resolution": used_resolution,
        }
        try:
            output, request_id = await _run_task_async(key, O2_BASE_T2I, payload, log_prefix)
            status = (
                f"✅ GPT-IMAGE-2 第三方渠道版 文生图成功\n"
                f"比例: {ratio}, 分辨率: {used_resolution}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirGPTImage2ThirdPartyI2INode:
    """bza-image-o2-base/image-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": O2_BASE_I2I_MAX, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(
        self, api_key, use_node_api_key, prompt, aspect_ratio, resolution, inputcount,
        image=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
    ):
        log_prefix = "BizyAirGPTImage2ThirdPartyI2I"
        key = _require_bizyair_key(api_key, use_node_api_key)
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        ratio = normalize_aspect_ratio(aspect_ratio, GPT_IMAGE_2_ASPECT_RATIOS)
        used_resolution = _resolve_o2_base_resolution(resolution, ratio, log_prefix)

        slots = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        valid_images = _collect_images(inputcount, O2_BASE_I2I_MAX, slots)

        payload = {
            "prompt": clean_prompt,
            "aspect_ratio": ratio,
            "resolution": used_resolution,
        }
        try:
            payload["image_urls"] = await upload_comfyui_images_to_oss_async(valid_images, key, log_prefix)
            output, request_id = await _run_task_async(key, O2_BASE_I2I, payload, log_prefix)
            status = (
                f"✅ GPT-IMAGE-2 第三方渠道版 图生图成功\n"
                f"比例: {ratio}, 分辨率: {used_resolution}, 参考图: {len(valid_images)}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirGPTImage2OfficialT2INode:
    """bza-image-o2-official/text-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(self, api_key, use_node_api_key, prompt, aspect_ratio, resolution, quality):
        log_prefix = "BizyAirGPTImage2OfficialT2I"
        key = _require_bizyair_key(api_key, use_node_api_key)
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        width, height = calculate_official_wh(aspect_ratio, resolution)
        payload = {
            "prompt": clean_prompt,
            "width": width,
            "height": height,
            "quality": quality,
        }
        try:
            output, request_id = await _run_task_async(key, O2_OFFICIAL_T2I, payload, log_prefix)
            status = (
                f"✅ GPT-IMAGE-2 官方版 文生图成功\n"
                f"尺寸: {width}x{height}, 质量: {quality}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirGPTImage2OfficialI2INode:
    """bza-image-o2-official/image-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": O2_OFFICIAL_I2I_MAX, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "image15": ("IMAGE",),
                "image16": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(
        self, api_key, use_node_api_key, prompt, aspect_ratio, resolution, quality, inputcount,
        image=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
        image11=None, image12=None, image13=None, image14=None, image15=None, image16=None,
    ):
        log_prefix = "BizyAirGPTImage2OfficialI2I"
        key = _require_bizyair_key(api_key, use_node_api_key)
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        width, height = calculate_official_wh(aspect_ratio, resolution)
        slots = [
            image, image2, image3, image4, image5, image6, image7, image8,
            image9, image10, image11, image12, image13, image14, image15, image16,
        ]
        valid_images = _collect_images(inputcount, O2_OFFICIAL_I2I_MAX, slots)

        payload = {
            "prompt": clean_prompt,
            "width": width,
            "height": height,
            "quality": quality,
        }
        try:
            payload["image_urls"] = await upload_comfyui_images_to_oss_async(valid_images, key, log_prefix)
            output, request_id = await _run_task_async(key, O2_OFFICIAL_I2I, payload, log_prefix)
            status = (
                f"✅ GPT-IMAGE-2 官方版 图生图成功\n"
                f"尺寸: {width}x{height}, 质量: {quality}, 参考图: {len(valid_images)}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


NODE_CLASS_MAPPINGS = {
    "BizyAirGPTImage2ThirdPartyT2INode": BizyAirGPTImage2ThirdPartyT2INode,
    "BizyAirGPTImage2ThirdPartyI2INode": BizyAirGPTImage2ThirdPartyI2INode,
    "BizyAirGPTImage2OfficialT2INode": BizyAirGPTImage2OfficialT2INode,
    "BizyAirGPTImage2OfficialI2INode": BizyAirGPTImage2OfficialI2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirGPTImage2ThirdPartyT2INode": "🌐BizyAir GPT-IMAGE-2 第三方渠道版 文生图",
    "BizyAirGPTImage2ThirdPartyI2INode": "🌐BizyAir GPT-IMAGE-2 第三方渠道版 图生图",
    "BizyAirGPTImage2OfficialT2INode": "🌐BizyAir GPT-IMAGE-2 官方版 文生图",
    "BizyAirGPTImage2OfficialI2INode": "🌐BizyAir GPT-IMAGE-2 官方版 图生图",
}

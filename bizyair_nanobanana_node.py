import traceback
from typing import Dict, Any, List, Optional

from .bizyair_common import (
    call_modelzoo_task_async,
    decode_image_from_url_async,
    upload_comfyui_images_to_oss_async,
    NANOBANANA2_EXTENDED_ASPECT_RATIOS,
    normalize_aspect_ratio,
    validate_prompt_length,
)

B2_BASE_T2I = "bza-image-b2-base/text-to-image"
B2_BASE_I2I = "bza-image-b2-base/image-to-image"
B2_OFFICIAL_T2I = "bza-image-b2-official/text-to-image"
B2_OFFICIAL_I2I = "bza-image-b2-official/image-to-image"

B2_I2I_MAX_IMAGES = 10


def _get_clean_api_key(api_key: str) -> str:
    invalid = {"", "YOUR_API_KEY", "你的apikey", "your_api_key_here", "请输入API密钥", "请输入你的API密钥"}
    key = (api_key or "").strip()
    return key if key and key not in invalid else ""


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


def _optional_seed(payload: Dict[str, Any], seed: int) -> None:
    if int(seed) > 0:
        payload["seed"] = int(seed)


def _optional_web_search(payload: Dict[str, Any], web_search: bool) -> None:
    if web_search:
        payload["web_search"] = True


class BizyAirNanoBanana2ThirdPartyT2INode:
    """bza-image-b2-base/text-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_EXTENDED_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(self, api_key, prompt, aspect_ratio, resolution):
        log_prefix = "BizyAirNanoBanana2ThirdPartyT2I"
        key = _get_clean_api_key(api_key)
        if not key:
            raise Exception("请在节点中填写 BizyAir API 密钥。")
        try:
            clean_prompt = validate_prompt_length(prompt, 20000)
        except ValueError as e:
            raise Exception(str(e)) from e

        payload = {
            "prompt": clean_prompt,
            "resolution": resolution,
            "aspect_ratio": normalize_aspect_ratio(aspect_ratio, NANOBANANA2_EXTENDED_ASPECT_RATIOS),
        }
        try:
            output, request_id = await _run_task_async(key, B2_BASE_T2I, payload, log_prefix)
            status = (
                f"✅ NanoBanana2 第三方渠道版 文生图成功\n"
                f"比例: {payload['aspect_ratio']}, 分辨率: {resolution}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirNanoBanana2ThirdPartyI2INode:
    """bza-image-b2-base/image-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_EXTENDED_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": B2_I2I_MAX_IMAGES, "step": 1}),
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
        self, api_key, prompt, aspect_ratio, resolution, inputcount,
        image=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
    ):
        log_prefix = "BizyAirNanoBanana2ThirdPartyI2I"
        key = _get_clean_api_key(api_key)
        if not key:
            raise Exception("请在节点中填写 BizyAir API 密钥。")
        try:
            clean_prompt = validate_prompt_length(prompt, 20000)
        except ValueError as e:
            raise Exception(str(e)) from e

        slots = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        valid_images = _collect_images(inputcount, B2_I2I_MAX_IMAGES, slots)

        payload = {
            "prompt": clean_prompt,
            "resolution": resolution,
            "aspect_ratio": normalize_aspect_ratio(aspect_ratio, NANOBANANA2_EXTENDED_ASPECT_RATIOS),
        }
        try:
            payload["image_urls"] = await upload_comfyui_images_to_oss_async(valid_images, key, log_prefix)
            output, request_id = await _run_task_async(key, B2_BASE_I2I, payload, log_prefix)
            status = (
                f"✅ NanoBanana2 第三方渠道版 图生图成功\n"
                f"比例: {payload['aspect_ratio']}, 分辨率: {resolution}, 参考图: {len(valid_images)}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirNanoBanana2OfficialT2INode:
    """bza-image-b2-official/text-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_EXTENDED_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "web_search": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    async def generate(self, api_key, prompt, aspect_ratio, resolution, seed, web_search):
        log_prefix = "BizyAirNanoBanana2OfficialT2I"
        key = _get_clean_api_key(api_key)
        if not key:
            raise Exception("请在节点中填写 BizyAir API 密钥。")
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        payload = {
            "prompt": clean_prompt,
            "resolution": resolution,
            "aspect_ratio": normalize_aspect_ratio(aspect_ratio, NANOBANANA2_EXTENDED_ASPECT_RATIOS),
        }
        _optional_seed(payload, seed)
        _optional_web_search(payload, web_search)

        try:
            output, request_id = await _run_task_async(key, B2_OFFICIAL_T2I, payload, log_prefix)
            status = (
                f"✅ NanoBanana2 官方版 文生图成功\n"
                f"比例: {payload['aspect_ratio']}, 分辨率: {resolution}\n"
                f"种子: {seed}, 联网搜索: {'是' if web_search else '否'}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


class BizyAirNanoBanana2OfficialI2INode:
    """bza-image-b2-official/image-to-image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_EXTENDED_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "web_search": ("BOOLEAN", {"default": False}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": B2_I2I_MAX_IMAGES, "step": 1}),
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
        self, api_key, prompt, aspect_ratio, resolution, seed, web_search, inputcount,
        image=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
    ):
        log_prefix = "BizyAirNanoBanana2OfficialI2I"
        key = _get_clean_api_key(api_key)
        if not key:
            raise Exception("请在节点中填写 BizyAir API 密钥。")
        try:
            clean_prompt = validate_prompt_length(prompt, 2500)
        except ValueError as e:
            raise Exception(str(e)) from e

        slots = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        valid_images = _collect_images(inputcount, B2_I2I_MAX_IMAGES, slots)

        payload = {
            "prompt": clean_prompt,
            "resolution": resolution,
            "aspect_ratio": normalize_aspect_ratio(aspect_ratio, NANOBANANA2_EXTENDED_ASPECT_RATIOS),
        }
        _optional_seed(payload, seed)
        _optional_web_search(payload, web_search)

        try:
            payload["image_urls"] = await upload_comfyui_images_to_oss_async(valid_images, key, log_prefix)
            output, request_id = await _run_task_async(key, B2_OFFICIAL_I2I, payload, log_prefix)
            status = (
                f"✅ NanoBanana2 官方版 图生图成功\n"
                f"比例: {payload['aspect_ratio']}, 分辨率: {resolution}, 参考图: {len(valid_images)}\n"
                f"种子: {seed}, 联网搜索: {'是' if web_search else '否'}\n"
                f"请求ID: {request_id}"
            )
            return (output, status)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"BizyAir 节点执行失败: {e}") from e


NODE_CLASS_MAPPINGS = {
    "BizyAirNanoBanana2ThirdPartyT2INode": BizyAirNanoBanana2ThirdPartyT2INode,
    "BizyAirNanoBanana2ThirdPartyI2INode": BizyAirNanoBanana2ThirdPartyI2INode,
    "BizyAirNanoBanana2OfficialT2INode": BizyAirNanoBanana2OfficialT2INode,
    "BizyAirNanoBanana2OfficialI2INode": BizyAirNanoBanana2OfficialI2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirNanoBanana2ThirdPartyT2INode": "🌐BizyAir NanoBanana2 第三方渠道版 文生图",
    "BizyAirNanoBanana2ThirdPartyI2INode": "🌐BizyAir NanoBanana2 第三方渠道版 图生图",
    "BizyAirNanoBanana2OfficialT2INode": "🌐BizyAir NanoBanana2 官方版 文生图",
    "BizyAirNanoBanana2OfficialI2INode": "🌐BizyAir NanoBanana2 官方版 图生图",
}

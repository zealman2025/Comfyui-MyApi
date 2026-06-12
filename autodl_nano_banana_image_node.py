"""
AutodL Nano Banana 2 — 官方 Gemini generateContent API（经 AutoDL 中转）

- 文生图 / 图生图：Gemini `v1beta` `generateContent`（`autodl_common.call_gemini_image`）
"""

import asyncio
import random
import traceback

try:
    from .autodl_common import (
        blank_image,
        call_gemini_image,
        check_image_deps,
        status_json,
    )
    from .myapi_keys import (
        PROVIDER_AUTODL,
        missing_api_key_message,
        resolve_api_key,
        USE_NODE_API_KEY_INPUT,
    )
except ImportError:
    from autodl_common import (
        blank_image,
        call_gemini_image,
        check_image_deps,
        status_json,
    )
    from myapi_keys import (
        PROVIDER_AUTODL,
        missing_api_key_message,
        resolve_api_key,
        USE_NODE_API_KEY_INPUT,
    )

NANOBANANA2_MODEL = "nano-banana-2"

NANOBANANA2_ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
]

NANOBANANA2_IMAGE_SIZES = ("0.5K", "1K", "2K", "4K")


class AutodlNanoBanana2T2INode:
    """AutodL Nano Banana 2 文生图（Gemini generateContent）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_ASPECT_RATIOS, {"default": "16:9"}),
                "image_size": (list(NANOBANANA2_IMAGE_SIZES), {"default": "1K"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(cls, api_key, use_node_api_key, prompt, aspect_ratio, image_size, seed):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (key_seed, prompt or "", aspect_ratio, image_size)

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(self, api_key, use_node_api_key, prompt, aspect_ratio, image_size, seed):
        _ = seed
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = resolve_api_key(PROVIDER_AUTODL, api_key, use_node_api_key)
        if not key:
            return blank_image(), f"Error: {missing_api_key_message(PROVIDER_AUTODL)}"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        try:
            image, info = call_gemini_image(
                key,
                NANOBANANA2_MODEL,
                str(prompt).strip(),
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
            info["mode"] = "文生图"
            info["protocol"] = "gemini generateContent"
            return image, status_json(**info)
        except Exception as e:
            print(f"[AutodL NanoBanana2 T2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


class AutodlNanoBanana2I2INode:
    """AutodL Nano Banana 2 图生图（Gemini generateContent，最多 14 张参考图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (NANOBANANA2_ASPECT_RATIOS, {"default": "16:9"}),
                "image_size": (list(NANOBANANA2_IMAGE_SIZES), {"default": "1K"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 14, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(
        cls,
        api_key,
        use_node_api_key,
        prompt,
        aspect_ratio,
        image_size,
        inputcount,
        seed,
        image=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        image11=None,
        image12=None,
        image13=None,
        image14=None,
    ):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        refs = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14]
        return (key_seed, prompt or "", aspect_ratio, image_size, int(inputcount), tuple(x is not None for x in refs))

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(
        self,
        api_key,
        use_node_api_key,
        prompt,
        aspect_ratio,
        image_size,
        inputcount,
        seed,
        image=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        image11=None,
        image12=None,
        image13=None,
        image14=None,
    ):
        _ = seed
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = resolve_api_key(PROVIDER_AUTODL, api_key, use_node_api_key)
        if not key:
            return blank_image(), f"Error: {missing_api_key_message(PROVIDER_AUTODL)}"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        inputcount = max(1, min(int(inputcount), 14))
        ref_images = [
            image,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
            image11,
            image12,
            image13,
            image14,
        ]
        ref_images = ref_images[:inputcount]
        ref_images = [img for img in ref_images if img is not None]

        if not ref_images:
            return blank_image(), "Error: 图生图至少需要一张参考图 (image)。"

        try:
            image_out, info = call_gemini_image(
                key,
                NANOBANANA2_MODEL,
                str(prompt).strip(),
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                reference_images=ref_images,
            )
            info["mode"] = "图生图"
            info["protocol"] = "gemini generateContent"
            info["inputcount"] = len(ref_images)
            return image_out, status_json(**info)
        except Exception as e:
            print(f"[AutodL NanoBanana2 I2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


NODE_CLASS_MAPPINGS = {
    "AutodlNanoBanana2T2INode": AutodlNanoBanana2T2INode,
    "AutodlNanoBanana2I2INode": AutodlNanoBanana2I2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutodlNanoBanana2T2INode": "🍎AutodL Nano Banana 2 文生图",
    "AutodlNanoBanana2I2INode": "🍎AutodL Nano Banana 2 图生图",
}

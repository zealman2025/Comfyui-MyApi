"""
AutodL GPT-IMAGE-2 — OpenAI Image API（经 AutoDL 中转，与 OpenAI / ComfyUI 内置一致）

端点:
  文生图: https://www.autodl.art/api/v1/images/generations  (application/json)
  图生图: https://www.autodl.art/api/v1/images/edits        (multipart/form-data)

- model 直接为 gpt-image-2，无需 orchestrator（编排）主模型
- UI：resolution + aspect_ratio 映射官方 size；quality 与官方一致
- 图生图：参考图作为文件 multipart 上传（单图 image / 多图 image[]）
"""

import asyncio
import random
import traceback

try:
    from .autodl_common import (
        GPT_IMAGE2_ASPECT_RATIOS,
        GPT_IMAGE2_RESOLUTIONS,
        blank_image,
        call_images_edit,
        call_images_generation,
        check_image_deps,
        get_api_key,
        map_gpt_image2_size,
        status_json,
    )
except ImportError:
    from autodl_common import (
        GPT_IMAGE2_ASPECT_RATIOS,
        GPT_IMAGE2_RESOLUTIONS,
        blank_image,
        call_images_edit,
        call_images_generation,
        check_image_deps,
        get_api_key,
        map_gpt_image2_size,
        status_json,
    )

GPT_IMAGE2_MODEL = "gpt-image-2"

GPT_IMAGE2_QUALITIES = ["low", "medium", "high", "auto"]


class AutodlGPTImage2T2INode:
    """AutodL GPT-IMAGE-2 文生图（Image API /images/generations）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "quality": (GPT_IMAGE2_QUALITIES, {"default": "medium"}),
                "resolution": (GPT_IMAGE2_RESOLUTIONS, {"default": "1K"}),
                "aspect_ratio": (GPT_IMAGE2_ASPECT_RATIOS, {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, quality, resolution, aspect_ratio, seed):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (key_seed, prompt or "", quality, resolution, aspect_ratio)

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(self, api_key, prompt, quality, resolution, aspect_ratio, seed):
        _ = seed
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = get_api_key(api_key)
        if not key:
            return blank_image(), "Error: 请在节点中填写 AutodL API 密钥。"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        try:
            size = map_gpt_image2_size(resolution, aspect_ratio)
            image, raw = call_images_generation(
                key,
                str(prompt).strip(),
                image_model=GPT_IMAGE2_MODEL,
                quality=quality,
                size=size,
                n=1,
            )
            return image, status_json(
                mode="文生图",
                protocol="images/generations",
                model=GPT_IMAGE2_MODEL,
                quality=quality,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        except Exception as e:
            print(f"[AutodL GPT-IMAGE-2 T2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


class AutodlGPTImage2I2INode:
    """AutodL GPT-IMAGE-2 图生图（Image API /images/edits，多参考图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "quality": (GPT_IMAGE2_QUALITIES, {"default": "medium"}),
                "resolution": (GPT_IMAGE2_RESOLUTIONS, {"default": "1K"}),
                "aspect_ratio": (GPT_IMAGE2_ASPECT_RATIOS, {"default": "16:9"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
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
        prompt,
        quality,
        resolution,
        aspect_ratio,
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
    ):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        refs = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        return (
            key_seed,
            prompt or "",
            quality,
            resolution,
            aspect_ratio,
            int(inputcount),
            tuple(x is not None for x in refs),
        )

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(
        self,
        api_key,
        prompt,
        quality,
        resolution,
        aspect_ratio,
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
    ):
        _ = seed
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = get_api_key(api_key)
        if not key:
            return blank_image(), "Error: 请在节点中填写 AutodL API 密钥。"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        inputcount = max(1, min(int(inputcount), 10))
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
        ]
        ref_images = ref_images[:inputcount]
        ref_images = [img for img in ref_images if img is not None]

        if not ref_images:
            return blank_image(), "Error: 图生图至少需要一张参考图 (image)。"

        try:
            size = map_gpt_image2_size(resolution, aspect_ratio)
            image_out, raw = call_images_edit(
                key,
                str(prompt).strip(),
                image_model=GPT_IMAGE2_MODEL,
                quality=quality,
                size=size,
                reference_images=ref_images,
                n=1,
            )
            return image_out, status_json(
                mode="图生图",
                protocol="images/edits",
                model=GPT_IMAGE2_MODEL,
                quality=quality,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                size=size,
                inputcount=len(ref_images),
            )
        except Exception as e:
            print(f"[AutodL GPT-IMAGE-2 I2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


NODE_CLASS_MAPPINGS = {
    "AutodlGPTImage2T2INode": AutodlGPTImage2T2INode,
    "AutodlGPTImage2I2INode": AutodlGPTImage2I2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutodlGPTImage2T2INode": "🍎AutodL GPT-IMAGE-2 文生图",
    "AutodlGPTImage2I2INode": "🍎AutodL GPT-IMAGE-2 图生图",
}

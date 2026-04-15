"""
AutodL OpenAI 兼容图像：nano-banana-2
- 文生图：未连接输入图像 → POST /v1/images/generations
- 图生图：已连接输入图像 → POST /v1/images/edits（multipart）；不对输入图做裁切/缩放，仅编码为 PNG 原样上传。

输出尺寸（本节点走 OpenAI images API，用字符串 "WxH"）——优先级：
1) custom_width 与 custom_height 均 >0 → 仅使用「宽x高」，image_resolution / aspect_ratio 不参与请求（界面里仍可留着，但被忽略）。
2) 二者均为 0 → 由 image_resolution + aspect_ratio 查表得到 WxH。
3) 仅一个为 0、另一个 >0 → 报错（避免歧义）。

不会把「比例/分辨率档位」与「自定义宽高」合并计算（OpenAI 只有一个 size 字符串）。

banana2 官方（Gemini generateContent）：无自定义像素宽高，仅 generationConfig.imageConfig 里
枚举 imageSize（0.5K/1K/2K/4K）与 aspectRatio（1:1 等），见 banana2.html。
"""

import base64
import io
import json
import random
import traceback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

AUTODL_OPENAI_BASE = "https://www.autodl.art/api/v1"
AUTODL_IMAGE_MODEL = "nano-banana-2"
URL_GENERATIONS = f"{AUTODL_OPENAI_BASE.rstrip('/')}/images/generations"
URL_EDITS = f"{AUTODL_OPENAI_BASE.rstrip('/')}/images/edits"

# 与 banana2.html 一致：分辨率档位 + 宽高比 → OpenAI 风格 size 字符串（宽x高）
# 若网关不支持某组合会返回 400，可改用 custom_width/custom_height 或换档位。
IMAGE_RESOLUTIONS = ("0.5K", "1K", "2K", "4K")
ASPECT_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4")


def _openai_size_from_resolution_and_aspect(resolution: str, aspect: str) -> str:
    """将 0.5K/1K/2K/4K 与 宽高比 映射为 API 的 WxH（短边对齐 banana2 的 K 概念）。"""
    r = str(resolution).strip()
    a = str(aspect).strip()
    # 行：分辨率；列：比例。数值为常见 OpenAI/生图网关可接受的像素组合（可被上游拒绝时请换档或自定义）。
    table = {
        ("0.5K", "1:1"): "512x512",
        ("0.5K", "16:9"): "896x512",
        ("0.5K", "9:16"): "512x896",
        ("0.5K", "4:3"): "640x512",
        ("0.5K", "3:4"): "512x640",
        ("1K", "1:1"): "1024x1024",
        ("1K", "16:9"): "1792x1024",
        ("1K", "9:16"): "1024x1792",
        ("1K", "4:3"): "1024x768",
        ("1K", "3:4"): "768x1024",
        ("2K", "1:1"): "2048x2048",
        ("2K", "16:9"): "2048x1152",
        ("2K", "9:16"): "1152x2048",
        ("2K", "4:3"): "2048x1536",
        ("2K", "3:4"): "1536x2048",
        ("4K", "1:1"): "4096x4096",
        ("4K", "16:9"): "3840x2160",
        ("4K", "9:16"): "2160x3840",
        ("4K", "4:3"): "4096x3072",
        ("4K", "3:4"): "3072x4096",
    }
    return table.get((r, a), "1024x1024")


class AutodlNanoBanana2ImageNode:
    def _normalize_api_key(self, key: str) -> str:
        if not key:
            return ""
        k = key.strip()
        if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
            k = k[1:-1].strip()
        low = k.lower()
        if low.startswith("bearer "):
            k = k[7:].strip()
        return k

    def _get_api_key(self, input_api_key: str) -> str:
        invalid = {
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥",
            "",
        }
        if input_api_key and input_api_key.strip() and input_api_key.strip() not in invalid:
            return self._normalize_api_key(input_api_key)
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_resolution": (
                    list(IMAGE_RESOLUTIONS),
                    {"default": "1K"},
                ),
                "aspect_ratio": (
                    list(ASPECT_RATIOS),
                    {"default": "1:1"},
                ),
                # 均为 0：用上方分辨率+比例；均 >0：仅此 WxH 生效，分辨率/比例被忽略
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(
        cls,
        api_key,
        prompt,
        image_resolution,
        aspect_ratio,
        custom_width,
        custom_height,
        response_format,
        seed,
        image=None,
    ):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (
            key_seed,
            prompt or "",
            image_resolution,
            aspect_ratio,
            int(custom_width),
            int(custom_height),
            response_format,
            image is not None,
        )

    def _check_deps(self):
        missing = []
        if not HAS_PIL or not HAS_NUMPY:
            missing.append("Pillow 与 numpy")
        if not HAS_TORCH:
            missing.append("torch")
        if not HAS_REQUESTS:
            missing.append("requests")
        return missing

    def _tensor_to_png_bytes(self, image) -> bytes:
        """ComfyUI IMAGE → PNG 字节，不裁切、不按 size 缩放（保持原图幅面）。"""
        if not HAS_PIL or not HAS_NUMPY:
            raise RuntimeError("缺少 Pillow/numpy")
        if HAS_TORCH and isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()
        else:
            arr = np.asarray(image)
        if len(arr.shape) == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _b64_or_url_to_tensor(self, b64, url):
        if b64:
            raw = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        elif url:
            r = requests.get(url, timeout=120, proxies={"http": None, "https": None})
            r.raise_for_status()
            pil = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            raise ValueError("无图像数据")

        arr = np.array(pil).astype(np.float32) / 255.0
        t = arr[np.newaxis, ...]
        if HAS_TORCH:
            return torch.from_numpy(t)
        return t

    def _blank_image(self):
        if HAS_TORCH:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return np.zeros((1, 64, 64, 3), dtype=np.float32)

    def _resolve_size(self, image_resolution, aspect_ratio, custom_width, custom_height):
        """
        返回 (size_str, source, err_msg)。
        custom_width 与 custom_height 均为 0 → 预设；均 >0 → f"{w}x{h}"；仅一侧 >0 → 错误。
        """
        w = int(custom_width)
        h = int(custom_height)
        if w == 0 and h == 0:
            return (
                _openai_size_from_resolution_and_aspect(image_resolution, aspect_ratio),
                "preset",
                None,
            )
        if w > 0 and h > 0:
            return f"{w}x{h}", "custom", None
        return (
            None,
            None,
            "custom_width 与 custom_height 须同时为 0（用预设）或同时 > 0（自定义宽高）",
        )

    def generate(
        self,
        api_key,
        prompt,
        image_resolution,
        aspect_ratio,
        custom_width,
        custom_height,
        response_format,
        seed,
        image=None,
    ):
        _ = seed
        missing = self._check_deps()
        if missing:
            return (self._blank_image(), f"Error: 缺少依赖: {', '.join(missing)}")

        key = self._get_api_key(api_key)
        if not key:
            return (self._blank_image(), "Error: 请在节点中填写 AutodL API 密钥。")

        if not (prompt and str(prompt).strip()):
            return (self._blank_image(), "Error: 请填写 prompt。")

        size, size_source, size_err = self._resolve_size(
            image_resolution, aspect_ratio, custom_width, custom_height
        )
        if size_err:
            return (self._blank_image(), f"Error: {size_err}")

        if size_source == "custom":
            print(
                f"[AutodL NanoBanana2] API size={size} 来自 custom_width×custom_height；"
                f"本次忽略 image_resolution={image_resolution!r}、aspect_ratio={aspect_ratio!r}。"
            )

        headers_json = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers_bearer_only = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

        try:
            if image is None:
                body = {
                    "model": AUTODL_IMAGE_MODEL,
                    "prompt": str(prompt).strip(),
                    "n": 1,
                    "size": size,
                    "response_format": response_format,
                }
                r = requests.post(
                    URL_GENERATIONS,
                    headers=headers_json,
                    json=body,
                    timeout=600,
                    proxies={"http": None, "https": None},
                )
            else:
                png_bytes = self._tensor_to_png_bytes(image)
                files = {"image": ("input.png", png_bytes, "image/png")}
                data = {
                    "model": AUTODL_IMAGE_MODEL,
                    "prompt": str(prompt).strip(),
                    "n": 1,
                    "size": size,
                }
                if response_format:
                    data["response_format"] = response_format
                r = requests.post(
                    URL_EDITS,
                    headers=headers_bearer_only,
                    files=files,
                    data=data,
                    timeout=600,
                    proxies={"http": None, "https": None},
                )

            if r.status_code == 401:
                return (
                    self._blank_image(),
                    "Error: 401 密钥无效或未授权图像接口。请检查 https://www.autodl.art/large-model/tokens 。",
                )

            try:
                data = r.json()
            except Exception:
                return (self._blank_image(), f"Error: HTTP {r.status_code} 非 JSON: {r.text[:500]}")

            if not r.ok:
                err = data.get("error") if isinstance(data, dict) else None
                msg = err.get("message", str(err)) if isinstance(err, dict) else r.text[:500]
                return (self._blank_image(), f"Error: HTTP {r.status_code} - {msg}")

            items = data.get("data") if isinstance(data, dict) else None
            if not items or not isinstance(items, list):
                return (self._blank_image(), f"Error: 响应无 data: {json.dumps(data, ensure_ascii=False)[:800]}")

            first = items[0] or {}
            b64 = first.get("b64_json")
            url = first.get("url")
            img_t = self._b64_or_url_to_tensor(b64, url)
            info = json.dumps(
                {
                    "model": AUTODL_IMAGE_MODEL,
                    "mode": "图生图" if image is not None else "文生图",
                    "endpoint": URL_EDITS if image is not None else URL_GENERATIONS,
                    "size": size,
                    "size_source": size_source,
                    "preset_used_for_size": size_source == "preset",
                    "custom_overrides_preset": size_source == "custom",
                    "image_resolution": image_resolution,
                    "aspect_ratio": aspect_ratio,
                    "custom_width": int(custom_width),
                    "custom_height": int(custom_height),
                },
                ensure_ascii=False,
            )
            return (img_t, info)
        except Exception as e:
            print(f"[AutodL NanoBanana2 Image] {e}")
            print(traceback.format_exc())
            return (self._blank_image(), f"Error: {e}")


NODE_CLASS_MAPPINGS = {
    "AutodlNanoBanana2ImageNode": AutodlNanoBanana2ImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutodlNanoBanana2ImageNode": "🍎AutodL Nano Banana 2 出图",
}

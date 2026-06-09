"""
Geeknow GPT-IMAGE-2 — 统一图像生成入口（OpenAI Images 兼容）

文档:
  https://docs.geeknow.top/api-reference/images/gpt-image-2/generation
  https://docs.geeknow.top/api-reference/images/gpt-image-2-pro/generation

端点: POST https://www.geeknow.top/v1/images/generations
  - 文生图：JSON {model, prompt, n, size, response_format}
  - 图生图：同一端点附带 image（Base64 字符串/数组）作为参考图
  - model: gpt-image-2（基础档）/ gpt-image-2-pro（额外 2K/4K 档）
  - 返回 data[0].b64_json 或 data[0].url
"""

import asyncio
import base64
import random
import traceback

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from .autodl_common import (
        REQUEST_PROXIES,
        REQUEST_TIMEOUT,
        blank_image,
        bytes_to_image_tensor,
        check_image_deps,
        get_api_key,
        status_json,
        tensor_to_compressed_image_bytes,
    )
except ImportError:
    from autodl_common import (
        REQUEST_PROXIES,
        REQUEST_TIMEOUT,
        blank_image,
        bytes_to_image_tensor,
        check_image_deps,
        get_api_key,
        status_json,
        tensor_to_compressed_image_bytes,
    )

# 可选 API 线路（显示标签 -> 实际 URL，以 /v1 结尾）
GEEKNOW_LINE_OPTIONS = {
    "https://geeknow.ai/v1 (cn2线路)": "https://geeknow.ai/v1",
    "https://api.geeknow.ai/v1 (cdn线路推荐国内用户)": "https://api.geeknow.ai/v1",
}
GEEKNOW_LINES = list(GEEKNOW_LINE_OPTIONS.keys())
GEEKNOW_LINE_DEFAULT = GEEKNOW_LINES[0]


def _resolve_line(line: str) -> str:
    return GEEKNOW_LINE_OPTIONS.get(line, line)


def _line_root(base: str) -> str:
    root = str(base).strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def geeknow_generations_url(base: str) -> str:
    return str(base).strip().rstrip("/") + "/images/generations"


def geeknow_presign_url(base: str) -> str:
    return _line_root(base) + "/api/upload/presign"


GEEKNOW_MODELS = ["gpt-image-2", "gpt-image-2-pro"]
GEEKNOW_MODEL_DEFAULT = "gpt-image-2"

GEEKNOW_QUALITIES = ["auto", "low", "medium", "high"]

# 图生图参考图传递方式
GEEKNOW_REF_MODES = ["base64 内嵌", "上传获取URL"]

# 基础档（gpt-image-2 与 pro 均支持）
GEEKNOW_ASPECT_RATIOS = ["1:1", "4:3", "3:2", "2:3", "16:9", "9:16"]

# 仅 gpt-image-2-pro 支持 2K / 4K
GEEKNOW_RESOLUTIONS = ["1K", "2K", "4K"]

GEEKNOW_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "4:3": "1536x1152",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
        "16:9": "1920x1080",
        "9:16": "1080x1920",
    },
    "2K": {
        "1:1": "2048x2048",
        "4:3": "2048x1536",
        "3:2": "2560x1712",
        "2:3": "1712x2560",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
    },
    "4K": {
        "1:1": "2880x2880",
        "4:3": "3840x2880",
        "3:2": "3840x2560",
        "2:3": "2560x3840",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
    },
}


def map_geeknow_size(model: str, resolution: str, aspect_ratio: str) -> str:
    res = str(resolution).strip()
    ratio = str(aspect_ratio).strip()
    # 基础模型不支持 2K/4K，回退到 1K
    if str(model).strip() == "gpt-image-2" and res in ("2K", "4K"):
        print(f"[Geeknow] gpt-image-2 不支持 {res}，已回退 1K（如需高分请用 gpt-image-2-pro）")
        res = "1K"
    tier = GEEKNOW_SIZE_MAP.get(res) or GEEKNOW_SIZE_MAP["1K"]
    return tier.get(ratio, tier["1:1"])


def _format_geeknow_error(response, data) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = err.get("message", str(err))
        code = err.get("code")
        if code:
            msg = f"{msg} (code={code})"
        return msg
    if isinstance(data, dict) and data.get("message"):
        return str(data.get("message"))
    return response.text[:500]


# 部分图床/CDN 拒绝非浏览器请求，下载图片时附带浏览器请求头（参考官方插件做法）
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _download_image_bytes(url: str) -> bytes:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    resp = requests.get(
        url, headers=_DOWNLOAD_HEADERS, timeout=REQUEST_TIMEOUT, proxies=REQUEST_PROXIES, stream=True
    )
    resp.raise_for_status()
    return resp.content


def _extract_image(data: dict):
    items = data.get("data") if isinstance(data, dict) else None
    if not items:
        raise ValueError("Geeknow 响应中未找到 data")
    first = items[0] or {}
    b64 = first.get("b64_json")
    if b64:
        return bytes_to_image_tensor(base64.b64decode(b64))
    url = first.get("url")
    if url:
        return bytes_to_image_tensor(_download_image_bytes(url))
    raise ValueError("Geeknow 响应中既无 b64_json 也无 url")


def _post_generations(base: str, api_key: str, payload: dict) -> dict:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(
        geeknow_generations_url(base),
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        proxies=REQUEST_PROXIES,
    )
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP {response.status_code} 非 JSON: {response.text[:500]}") from exc
    if response.status_code == 401:
        raise RuntimeError("401 密钥无效或未授权。请检查 Geeknow API Key。")
    if not response.ok:
        detail = _format_geeknow_error(response, data)
        print(f"[Geeknow] HTTP {response.status_code} 响应: {response.text[:1500]}")
        raise RuntimeError(f"HTTP {response.status_code} - {detail}")
    return data


def _ref_images_to_base64(reference_images) -> list:
    refs = [img for img in (reference_images or []) if img is not None]
    encoded = []
    for img in refs:
        img_bytes, _mime = tensor_to_compressed_image_bytes(
            img, max_bytes=4 * 1024 * 1024, target_ratio=0.6, min_dim=256, max_long_edge=1536
        )
        encoded.append(base64.b64encode(img_bytes).decode("utf-8"))
    return encoded


def _presign_upload(base: str, api_key: str, file_name: str, content_type: str, expires_in: int = 900) -> dict:
    """调用 /api/upload/presign 取得预签名 PUT 地址与 public_url。"""
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {"file_name": file_name, "content_type": content_type, "expires_in": int(expires_in)}
    resp = requests.post(
        geeknow_presign_url(base), headers=headers, json=body,
        timeout=REQUEST_TIMEOUT, proxies=REQUEST_PROXIES,
    )
    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"presign HTTP {resp.status_code} 非 JSON: {resp.text[:300]}") from exc
    if resp.status_code == 401:
        raise RuntimeError("401 密钥无效或未授权（presign）。")
    # /api/* 接口用 success 字段表示业务结果
    if not resp.ok or not data.get("success"):
        msg = data.get("message") or resp.text[:300]
        raise RuntimeError(f"presign 失败: {msg}")
    payload = data.get("data") or {}
    if not payload.get("upload_url") or not payload.get("public_url"):
        raise RuntimeError(f"presign 返回缺少 upload_url/public_url: {data}")
    return payload


def _put_file(upload_url: str, content: bytes, content_type: str) -> None:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    resp = requests.put(
        upload_url,
        data=content,
        headers={"Content-Type": content_type},
        timeout=REQUEST_TIMEOUT,
        proxies=REQUEST_PROXIES,
    )
    if not resp.ok:
        raise RuntimeError(f"对象存储上传失败 HTTP {resp.status_code}: {resp.text[:300]}")


def _ref_images_to_urls(base: str, api_key: str, reference_images) -> list:
    """压缩参考图后经预签名上传，返回 public_url 列表。"""
    refs = [img for img in (reference_images or []) if img is not None]
    urls = []
    for i, img in enumerate(refs):
        img_bytes, mime = tensor_to_compressed_image_bytes(
            img, max_bytes=8 * 1024 * 1024, target_ratio=0.7, min_dim=512, max_long_edge=2048
        )
        ext = "jpg" if mime == "image/jpeg" else "png"
        signed = _presign_upload(base, api_key, f"ref_{i}.{ext}", mime)
        _put_file(signed["upload_url"], img_bytes, mime)
        urls.append(signed["public_url"])
        print(f"[Geeknow] 已上传参考图 {i + 1}/{len(refs)} -> {signed['public_url']}")
    return urls


class GeeknowGPTImage2T2INode:
    """Geeknow GPT-IMAGE-2 文生图（/v1/images/generations）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "line": (GEEKNOW_LINES, {"default": GEEKNOW_LINE_DEFAULT}),
                "model": (GEEKNOW_MODELS, {"default": GEEKNOW_MODEL_DEFAULT}),
                "quality": (GEEKNOW_QUALITIES, {"default": "auto"}),
                "resolution": (GEEKNOW_RESOLUTIONS, {"default": "1K"}),
                "aspect_ratio": (GEEKNOW_ASPECT_RATIOS, {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, line, model, quality, resolution, aspect_ratio, seed):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (key_seed, prompt or "", line, model, quality, resolution, aspect_ratio)

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(self, api_key, prompt, line, model, quality, resolution, aspect_ratio, seed):
        _ = seed
        line = _resolve_line(line)
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = get_api_key(api_key)
        if not key:
            return blank_image(), "Error: 请在节点中填写 Geeknow API 密钥。"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        try:
            size = map_geeknow_size(model, resolution, aspect_ratio)
            payload = {
                "model": model,
                "prompt": str(prompt).strip(),
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }
            q = str(quality).strip()
            if q and q != "auto":
                payload["quality"] = q
            print(f"[Geeknow] T2I | line={line} | model={model} | size={size} | quality={quality}")
            data = _post_generations(line, key, payload)
            image = _extract_image(data)
            return image, status_json(
                mode="文生图",
                protocol="images/generations",
                model=model,
                quality=quality,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        except Exception as e:
            print(f"[Geeknow GPT-IMAGE-2 T2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


class GeeknowGPTImage2I2INode:
    """Geeknow GPT-IMAGE-2 图生图（/v1/images/generations，附带 image 参考图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "line": (GEEKNOW_LINES, {"default": GEEKNOW_LINE_DEFAULT}),
                "model": (GEEKNOW_MODELS, {"default": GEEKNOW_MODEL_DEFAULT}),
                "quality": (GEEKNOW_QUALITIES, {"default": "auto"}),
                "resolution": (GEEKNOW_RESOLUTIONS, {"default": "1K"}),
                "aspect_ratio": (GEEKNOW_ASPECT_RATIOS, {"default": "16:9"}),
                "reference_mode": (GEEKNOW_REF_MODES, {"default": "base64 内嵌"}),
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
        line,
        model,
        quality,
        resolution,
        aspect_ratio,
        reference_mode,
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
            line,
            model,
            quality,
            resolution,
            aspect_ratio,
            reference_mode,
            int(inputcount),
            tuple(x is not None for x in refs),
        )

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(
        self,
        api_key,
        prompt,
        line,
        model,
        quality,
        resolution,
        aspect_ratio,
        reference_mode,
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
        line = _resolve_line(line)
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = get_api_key(api_key)
        if not key:
            return blank_image(), "Error: 请在节点中填写 Geeknow API 密钥。"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        inputcount = max(1, min(int(inputcount), 10))
        ref_images = [
            image, image2, image3, image4, image5,
            image6, image7, image8, image9, image10,
        ][:inputcount]
        ref_images = [img for img in ref_images if img is not None]

        if not ref_images:
            return blank_image(), "Error: 图生图至少需要一张参考图 (image)。"

        try:
            size = map_geeknow_size(model, resolution, aspect_ratio)
            if str(reference_mode).strip() == "上传获取URL":
                images = _ref_images_to_urls(line, key, ref_images)
            else:
                images = _ref_images_to_base64(ref_images)
            payload = {
                "model": model,
                "prompt": str(prompt).strip(),
                "n": 1,
                "size": size,
                "response_format": "b64_json",
                "image": images if len(images) > 1 else images[0],
            }
            q = str(quality).strip()
            if q and q != "auto":
                payload["quality"] = q
            print(
                f"[Geeknow] I2I | line={line} | model={model} | refs={len(images)} | "
                f"mode={reference_mode} | size={size} | quality={quality}"
            )
            data = _post_generations(line, key, payload)
            image_out = _extract_image(data)
            return image_out, status_json(
                mode="图生图",
                protocol="images/generations",
                model=model,
                quality=quality,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                size=size,
                inputcount=len(images),
                reference_mode=reference_mode,
            )
        except Exception as e:
            print(f"[Geeknow GPT-IMAGE-2 I2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


NODE_CLASS_MAPPINGS = {
    "GeeknowGPTImage2T2INode": GeeknowGPTImage2T2INode,
    "GeeknowGPTImage2I2INode": GeeknowGPTImage2I2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeeknowGPTImage2T2INode": "🍆Geeknow GPT-IMAGE-2 文生图",
    "GeeknowGPTImage2I2INode": "🍆Geeknow GPT-IMAGE-2 图生图",
}

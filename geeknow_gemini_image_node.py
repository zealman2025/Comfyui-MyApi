"""
Geeknow Gemini 图像（banana）— generateContent 格式

文档: https://docs.geeknow.top/api-reference/images/gemini-image/generation

端点: POST https://www.geeknow.top/v1beta/models/{model}:generateContent
认证: Authorization: Bearer YOUR_API_KEY

- 文生图 / 图生图共用 contents[].parts[] 结构
- 参考图通过 parts[].inlineData(base64) 传入
- generationConfig.responseModalities 建议 ["IMAGE", "TEXT"]
- imageConfig: aspectRatio + imageSize（1K/2K，flash 模型实际回落 1K）
- 返回 candidates[0].content.parts[].inlineData.data（可能是 base64 或 URL）
"""

import asyncio
import base64
import json
import random
import re
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
        status_json,
        tensor_to_compressed_image_bytes,
    )
    from .myapi_keys import (
        PROVIDER_GEEKNOW,
        missing_api_key_message,
        resolve_api_key,
        USE_NODE_API_KEY_INPUT,
    )
except ImportError:
    from autodl_common import (
        REQUEST_PROXIES,
        REQUEST_TIMEOUT,
        blank_image,
        bytes_to_image_tensor,
        check_image_deps,
        status_json,
        tensor_to_compressed_image_bytes,
    )
    from myapi_keys import (
        PROVIDER_GEEKNOW,
        missing_api_key_message,
        resolve_api_key,
        USE_NODE_API_KEY_INPUT,
    )

# 可选 API 线路（显示标签 -> 实际 URL，以 /v1 结尾，Gemini 实际走根域名 + /v1beta）
GEEKNOW_LINE_OPTIONS = {
    "https://geeknow.ai/v1 (cn2线路)": "https://geeknow.ai/v1",
    "https://api.geeknow.ai/v1 (cdn线路推荐国内用户)": "https://api.geeknow.ai/v1",
}
GEEKNOW_LINES = list(GEEKNOW_LINE_OPTIONS.keys())
GEEKNOW_LINE_DEFAULT = GEEKNOW_LINES[0]


def _resolve_line(line: str) -> str:
    return GEEKNOW_LINE_OPTIONS.get(line, line)


def _gemini_url(base: str, model: str) -> str:
    root = str(base).strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    root = root.rstrip("/")
    return f"{root}/v1beta/models/{model}:generateContent"


GEEKNOW_GEMINI_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
    "gemini-3.1-flash-image-preview",
]
GEEKNOW_GEMINI_MODEL_DEFAULT = "gemini-3-pro-image-preview"

GEEKNOW_GEMINI_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
GEEKNOW_GEMINI_SIZES = ["1K", "2K"]


def _resolve_image_size(model: str, image_size: str) -> str:
    size = str(image_size).strip() or "1K"
    if size == "2K" and "flash" in str(model):
        print(f"[Geeknow Gemini] {model} 对 2K 会实际回落为 1K（仅 pro 真正支持 2K）")
    return size


def _build_gemini_payload(prompt: str, *, aspect_ratio: str, image_size: str, reference_images=None) -> dict:
    parts = [{"text": str(prompt).strip()}]
    for img in reference_images or []:
        if img is None:
            continue
        img_bytes, mime = tensor_to_compressed_image_bytes(
            img, max_bytes=4 * 1024 * 1024, target_ratio=0.6, min_dim=256, max_long_edge=1536
        )
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(img_bytes).decode("utf-8"),
                }
            }
        )
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "imageConfig": {
                "aspectRatio": str(aspect_ratio).strip(),
                "imageSize": str(image_size).strip(),
            },
        },
    }


def _format_gemini_error(response, data) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = err.get("message", str(err))
        code = err.get("code") or err.get("status")
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


def _decode_image_payload(data_str: str) -> bytes:
    s = str(data_str).strip()
    if s.startswith("http://") or s.startswith("https://"):
        return _download_image_bytes(s)
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    return base64.b64decode(s)


def _find_url_or_data_uri(text: str):
    if not text:
        return None
    s = str(text)
    m = re.search(r"data:image/[^\s\"')]+;base64,[A-Za-z0-9+/=]+", s)
    if m:
        return m.group(0)
    m = re.search(r"https?://[^\s\"')\]]+\.(?:png|jpg|jpeg|webp|gif)", s, re.IGNORECASE)
    if m:
        return m.group(0)
    m = re.search(r"https?://[^\s\"')\]]+", s)
    if m:
        return m.group(0)
    return None


def _extract_gemini_image(data: dict):
    # 1) 标准 generateContent：candidates[].content.parts[].inlineData.data
    candidates = data.get("candidates") if isinstance(data, dict) else None
    text_blobs = []
    for cand in candidates or []:
        content = (cand or {}).get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return bytes_to_image_tensor(_decode_image_payload(inline["data"]))
            # 部分渠道把图片放进 fileData / file_data.fileUri
            file_data = part.get("fileData") or part.get("file_data")
            if isinstance(file_data, dict):
                uri = file_data.get("fileUri") or file_data.get("file_uri")
                if uri:
                    return bytes_to_image_tensor(_decode_image_payload(uri))
            if part.get("text"):
                text_blobs.append(part.get("text"))

    # 2) 文本片段里可能嵌了图片 URL / data URI（markdown 等）
    for blob in text_blobs:
        found = _find_url_or_data_uri(blob)
        if found:
            return bytes_to_image_tensor(_decode_image_payload(found))

    # 3) 兜底：OpenAI 风格 data[].b64_json / url
    items = data.get("data") if isinstance(data, dict) else None
    if items:
        first = items[0] or {}
        if first.get("b64_json"):
            return bytes_to_image_tensor(base64.b64decode(first["b64_json"]))
        if first.get("url"):
            return bytes_to_image_tensor(_decode_image_payload(first["url"]))

    snippet = json.dumps(data, ensure_ascii=False)[:1500]
    print(f"[Geeknow Gemini] 未识别的响应结构: {snippet}")
    raise ValueError("Gemini 响应中未找到图像数据（已打印响应结构到控制台）")


def _post_gemini(base: str, api_key: str, model: str, payload: dict) -> dict:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    url = _gemini_url(base, model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(
        url, headers=headers, json=payload,
        timeout=REQUEST_TIMEOUT, proxies=REQUEST_PROXIES,
    )
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP {response.status_code} 非 JSON: {response.text[:500]}") from exc
    if response.status_code == 401:
        raise RuntimeError("401 密钥无效或未授权。请检查 Geeknow API Key。")
    if not response.ok:
        detail = _format_gemini_error(response, data)
        print(f"[Geeknow Gemini] HTTP {response.status_code} 响应: {response.text[:1500]}")
        raise RuntimeError(f"HTTP {response.status_code} - {detail}")
    return data


class GeeknowGeminiImageT2INode:
    """Geeknow Gemini 图像 文生图（generateContent）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "line": (GEEKNOW_LINES, {"default": GEEKNOW_LINE_DEFAULT}),
                "model": (GEEKNOW_GEMINI_MODELS, {"default": GEEKNOW_GEMINI_MODEL_DEFAULT}),
                "aspect_ratio": (GEEKNOW_GEMINI_ASPECT_RATIOS, {"default": "16:9"}),
                "image_size": (GEEKNOW_GEMINI_SIZES, {"default": "1K"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(cls, api_key, use_node_api_key, prompt, line, model, aspect_ratio, image_size, seed):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (key_seed, prompt or "", line, model, aspect_ratio, image_size)

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(self, api_key, use_node_api_key, prompt, line, model, aspect_ratio, image_size, seed):
        _ = seed
        line = _resolve_line(line)
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = resolve_api_key(PROVIDER_GEEKNOW, api_key, use_node_api_key)
        if not key:
            return blank_image(), f"Error: {missing_api_key_message(PROVIDER_GEEKNOW)}"

        if not (prompt and str(prompt).strip()):
            return blank_image(), "Error: 请填写 prompt。"

        try:
            size = _resolve_image_size(model, image_size)
            payload = _build_gemini_payload(prompt, aspect_ratio=aspect_ratio, image_size=size)
            print(f"[Geeknow Gemini] T2I | line={line} | model={model} | ratio={aspect_ratio} | size={size}")
            data = _post_gemini(line, key, model, payload)
            image = _extract_gemini_image(data)
            return image, status_json(
                mode="文生图",
                protocol="generateContent",
                model=model,
                aspect_ratio=aspect_ratio,
                image_size=size,
            )
        except Exception as e:
            print(f"[Geeknow Gemini T2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


class GeeknowGeminiImageI2INode:
    """Geeknow Gemini 图像 图生图（generateContent，多参考图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "use_node_api_key": USE_NODE_API_KEY_INPUT,
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "line": (GEEKNOW_LINES, {"default": GEEKNOW_LINE_DEFAULT}),
                "model": (GEEKNOW_GEMINI_MODELS, {"default": GEEKNOW_GEMINI_MODEL_DEFAULT}),
                "aspect_ratio": (GEEKNOW_GEMINI_ASPECT_RATIOS, {"default": "16:9"}),
                "image_size": (GEEKNOW_GEMINI_SIZES, {"default": "1K"}),
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
        use_node_api_key,
        prompt,
        line,
        model,
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
    ):
        key_seed = random.random() if int(seed) == 0 else int(seed)
        refs = [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        return (
            key_seed,
            prompt or "",
            line,
            model,
            aspect_ratio,
            image_size,
            int(inputcount),
            tuple(x is not None for x in refs),
        )

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, **kwargs)

    def _generate_sync(
        self,
        api_key,
        use_node_api_key,
        prompt,
        line,
        model,
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
    ):
        _ = seed
        line = _resolve_line(line)
        missing = check_image_deps(require_torch=True)
        if missing:
            return blank_image(), f"Error: 缺少依赖: {', '.join(missing)}"

        key = resolve_api_key(PROVIDER_GEEKNOW, api_key, use_node_api_key)
        if not key:
            return blank_image(), f"Error: {missing_api_key_message(PROVIDER_GEEKNOW)}"

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
            size = _resolve_image_size(model, image_size)
            payload = _build_gemini_payload(
                prompt, aspect_ratio=aspect_ratio, image_size=size, reference_images=ref_images
            )
            print(
                f"[Geeknow Gemini] I2I | line={line} | model={model} | refs={len(ref_images)} | "
                f"ratio={aspect_ratio} | size={size}"
            )
            data = _post_gemini(line, key, model, payload)
            image_out = _extract_gemini_image(data)
            return image_out, status_json(
                mode="图生图",
                protocol="generateContent",
                model=model,
                aspect_ratio=aspect_ratio,
                image_size=size,
                inputcount=len(ref_images),
            )
        except Exception as e:
            print(f"[Geeknow Gemini I2I] {e}")
            print(traceback.format_exc())
            return blank_image(), f"Error: {e}"


NODE_CLASS_MAPPINGS = {
    "GeeknowGeminiImageT2INode": GeeknowGeminiImageT2INode,
    "GeeknowGeminiImageI2INode": GeeknowGeminiImageI2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeeknowGeminiImageT2INode": "🍆Geeknow Gemini 图像 文生图",
    "GeeknowGeminiImageI2INode": "🍆Geeknow Gemini 图像 图生图",
}

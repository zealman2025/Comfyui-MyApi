import base64
import io
import json

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

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    genai_types = None

AUTODL_OPENAI_BASE = "https://www.autodl.art/api/v1"
AUTODL_GEMINI_BASE = "https://www.autodl.art/api/v1/gemini"
AUTODL_GEMINI_GENERATE_URL = AUTODL_GEMINI_BASE.rstrip("/") + "/v1beta/models/{model}:generateContent"
AUTODL_RESPONSES_URL = f"{AUTODL_OPENAI_BASE.rstrip('/')}/responses"

REQUEST_PROXIES = {"http": None, "https": None}
REQUEST_TIMEOUT = 600

INVALID_API_KEY_PLACEHOLDERS = {
    "YOUR_API_KEY",
    "你的apikey",
    "your_api_key_here",
    "请输入API密钥",
    "请输入你的API密钥",
    "",
}


def normalize_api_key(key: str) -> str:
    if not key:
        return ""
    k = key.strip()
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        k = k[1:-1].strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k


def get_api_key(input_api_key: str) -> str:
    if (
        input_api_key
        and input_api_key.strip()
        and input_api_key.strip() not in INVALID_API_KEY_PLACEHOLDERS
    ):
        return normalize_api_key(input_api_key)
    return ""


def check_image_deps(require_torch: bool = False):
    missing = []
    if not HAS_PIL or not HAS_NUMPY:
        missing.append("Pillow 与 numpy")
    if require_torch and not HAS_TORCH:
        missing.append("torch")
    if not HAS_REQUESTS:
        missing.append("requests")
    return missing


def tensor_to_png_bytes(image) -> bytes:
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


def bytes_to_image_tensor(raw: bytes):
    if not HAS_PIL or not HAS_NUMPY:
        raise RuntimeError("缺少 Pillow/numpy")
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    tensor = arr[np.newaxis, ...]
    if HAS_TORCH:
        return torch.from_numpy(tensor)
    return tensor


def blank_image():
    if HAS_TORCH:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return np.zeros((1, 64, 64, 3), dtype=np.float32)


def png_data_url(image) -> str:
    png_bytes = tensor_to_png_bytes(image)
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_responses_image_b64(response_data: dict) -> str:
    outputs = response_data.get("output") or []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image_generation_call":
            continue
        if item.get("status") not in (None, "completed", "generating", "in_progress"):
            continue
        result = item.get("result")
        if result:
            return result
    raise ValueError("Responses 响应中未找到 image_generation_call 结果")


def post_gemini_json(url: str, api_key: str, payload: dict) -> dict:
    """AutoDL Gemini 中转：使用 x-goog-api-key（与 google-genai SDK 一致）。"""
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(
        url,
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
        raise RuntimeError(
            "401 密钥无效或未授权。请到 https://www.autodl.art/large-model/tokens 检查令牌。"
        )
    if not response.ok:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            msg = err.get("message", str(err))
        else:
            msg = response.text[:500]
        raise RuntimeError(f"HTTP {response.status_code} - {msg}")
    return data


def post_json(url: str, api_key: str, payload: dict) -> dict:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(
        url,
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
        raise RuntimeError(
            "401 密钥无效或未授权。请到 https://www.autodl.art/large-model/tokens 检查令牌。"
        )
    if not response.ok:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            msg = err.get("message", str(err))
        else:
            msg = response.text[:500]
        raise RuntimeError(f"HTTP {response.status_code} - {msg}")
    return data


def build_responses_image_payload(
    prompt: str,
    *,
    orchestrator_model: str,
    image_model: str,
    quality: str,
    size: str,
    action: str,
    reference_images=None,
):
    tool = {
        "type": "image_generation",
        "model": image_model,
        "quality": quality,
        "size": size,
        "action": action,
    }

    if reference_images:
        content = [{"type": "input_text", "text": prompt}]
        for image in reference_images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": png_data_url(image),
                }
            )
        input_payload = [{"role": "user", "content": content}]
    else:
        input_payload = prompt

    return {
        "model": orchestrator_model,
        "input": input_payload,
        "tools": [tool],
    }


def call_responses_image(
    api_key: str,
    prompt: str,
    *,
    orchestrator_model: str,
    image_model: str,
    quality: str,
    size: str,
    action: str,
    reference_images=None,
):
    payload = build_responses_image_payload(
        prompt,
        orchestrator_model=orchestrator_model,
        image_model=image_model,
        quality=quality,
        size=size,
        action=action,
        reference_images=reference_images,
    )
    data = post_json(AUTODL_RESPONSES_URL, api_key, payload)
    image_b64 = extract_responses_image_b64(data)
    return bytes_to_image_tensor(base64.b64decode(image_b64)), data


def map_gemini_image_size(resolution: str) -> str:
    mapping = {
        "0.5K": "512",
        "1K": "1K",
        "2K": "2K",
        "4K": "4K",
    }
    return mapping.get(str(resolution).strip(), "1K")


GPT_IMAGE2_ASPECT_RATIOS = [
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
]

GPT_IMAGE2_RESOLUTIONS = ["1K", "2K", "4K", "auto"]

# resolution + aspect_ratio -> 官方 size（WxH），均满足 gpt-image-2 约束：
# 边长 ≤3840、16 倍数、长宽比 ≤3:1、总像素 655360~8294400
GPT_IMAGE2_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "2:3": "1024x1536",
        "3:2": "1536x1024",
        "3:4": "1024x1360",
        "4:3": "1360x1024",
        "4:5": "1024x1280",
        "5:4": "1280x1024",
        "9:16": "1024x1824",
        "16:9": "1824x1024",
        "21:9": "2384x1024",
    },
    "2K": {
        "1:1": "2048x2048",
        "2:3": "2048x3072",
        "3:2": "3072x2048",
        "3:4": "2048x2736",
        "4:3": "2736x2048",
        "4:5": "2048x2560",
        "5:4": "2560x2048",
        "9:16": "1152x2048",
        "16:9": "2048x1152",
        "21:9": "2048x880",
    },
    "4K": {
        "1:1": "2880x2880",
        "2:3": "2352x3520",
        "3:2": "3520x2352",
        "3:4": "2448x3264",
        "4:3": "3264x2448",
        "4:5": "2576x3216",
        "5:4": "3216x2576",
        "9:16": "2160x3840",
        "16:9": "3840x2160",
        "21:9": "3840x1648",
    },
}


def map_gpt_image2_size(resolution: str, aspect_ratio: str) -> str:
    res = str(resolution).strip()
    if res == "auto":
        return "auto"
    ratio = str(aspect_ratio).strip()
    tier = GPT_IMAGE2_SIZE_MAP.get(res)
    if not tier:
        raise ValueError(f"不支持的 resolution: {resolution}")
    size = tier.get(ratio)
    if not size:
        raise ValueError(f"不支持的 aspect_ratio: {aspect_ratio}")
    return size


def extract_gemini_response_image_bytes(response_data: dict) -> bytes:
    candidates = response_data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline:
                continue
            data = inline.get("data")
            if data:
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data
    raise ValueError("Gemini 响应中未找到图像数据")


def extract_genai_image_bytes(response) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                data = inline_data.data
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data
            if hasattr(part, "as_image"):
                try:
                    img = part.as_image()
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return buf.getvalue()
                except Exception:
                    pass
    raise ValueError("Gemini 响应中未找到图像数据")


def call_gemini_image(
    api_key: str,
    model: str,
    prompt: str,
    *,
    aspect_ratio: str,
    image_size: str,
    reference_images=None,
):
    """经 AutoDL Gemini 中转：v1beta generateContent，支持 aspectRatio + imageSize。"""
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")

    parts = []
    for image in reference_images or []:
        if image is None:
            continue
        png_bytes = tensor_to_png_bytes(image)
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(png_bytes).decode("utf-8"),
                }
            }
        )

    if prompt and str(prompt).strip():
        parts.append({"text": str(prompt).strip()})

    if not parts:
        raise ValueError("至少需要 prompt 或参考图")

    url = AUTODL_GEMINI_GENERATE_URL.format(model=model)
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": map_gemini_image_size(image_size),
            },
        },
    }

    data = post_gemini_json(url, api_key, payload)
    image_bytes = extract_gemini_response_image_bytes(data)
    info = {
        "model": model,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "reference_count": len([x for x in (reference_images or []) if x is not None]),
        "protocol": "gemini v1beta generateContent",
        "endpoint": url,
    }
    return bytes_to_image_tensor(image_bytes), info


def status_json(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)

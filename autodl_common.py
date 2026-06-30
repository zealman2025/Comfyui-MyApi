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

import importlib.util as _importlib_util

HAS_GOOGLE_GENAI = _importlib_util.find_spec("google.genai") is not None
genai = None
genai_types = None

AUTODL_OPENAI_BASE = "https://www.autodl.art/api/v1"
AUTODL_GEMINI_BASE = "https://www.autodl.art/api/v1/gemini"
AUTODL_GEMINI_GENERATE_URL = AUTODL_GEMINI_BASE.rstrip("/") + "/v1beta/models/{model}:generateContent"
AUTODL_RESPONSES_URL = f"{AUTODL_OPENAI_BASE.rstrip('/')}/responses"
AUTODL_IMAGES_GENERATIONS_URL = f"{AUTODL_OPENAI_BASE.rstrip('/')}/images/generations"
AUTODL_IMAGES_EDITS_URL = f"{AUTODL_OPENAI_BASE.rstrip('/')}/images/edits"

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


def _tensor_to_pil_rgb(image) -> "Image.Image":
    if not HAS_PIL or not HAS_NUMPY:
        raise RuntimeError("缺少 Pillow/numpy")
    if HAS_TORCH and isinstance(image, torch.Tensor):
        arr = image.cpu().numpy()
    else:
        arr = np.asarray(image)
    if len(arr.shape) == 4:
        arr = arr[0]
    if len(arr.shape) == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def tensor_to_compressed_image_bytes(
    image,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    target_ratio: float = 0.7,
    min_dim: int = 512,
    max_long_edge: int | None = None,
) -> tuple[bytes, str]:
    """压缩 ComfyUI 图像张量，供 API 内嵌 base64 使用（与 AutodL API 节点策略一致）。"""
    pil_image = _tensor_to_pil_rgb(image)
    if max_long_edge and max(pil_image.width, pil_image.height) > max_long_edge:
        scale = max_long_edge / max(pil_image.width, pil_image.height)
        new_width = max(int(pil_image.width * scale), min_dim)
        new_height = max(int(pil_image.height * scale), min_dim)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    target_raw_bytes = int(max_bytes * target_ratio)

    def save_to_buffer(img, fmt="JPEG", **save_kwargs):
        buf = io.BytesIO()
        img.save(buf, format=fmt, **save_kwargs)
        return buf, buf.tell()

    buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=95, optimize=True)
    mime = "image/jpeg"
    resize_attempts = 0
    while (
        raw_size > target_raw_bytes
        and (pil_image.width > min_dim or pil_image.height > min_dim)
        and resize_attempts < 5
    ):
        scale_factor = max((target_raw_bytes / raw_size) ** 0.5, 0.3)
        new_width = max(int(pil_image.width * scale_factor), min_dim)
        new_height = max(int(pil_image.height * scale_factor), min_dim)
        if new_width == pil_image.width and new_height == pil_image.height:
            new_width = max(int(pil_image.width * 0.75), min_dim)
            new_height = max(int(pil_image.height * 0.75), min_dim)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resize_attempts += 1
        buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=90, optimize=True)

    quality = 90
    while raw_size > target_raw_bytes and quality >= 40:
        buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=quality, optimize=True)
        quality -= 5

    if raw_size > target_raw_bytes:
        raise ValueError(
            f"参考图压缩后仍过大 ({raw_size / 1024 / 1024:.2f}MB)，请换更小分辨率或手动缩小。"
        )
    return buffer.getvalue(), mime


def tensor_to_png_bytes(image) -> bytes:
    pil = _tensor_to_pil_rgb(image)
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


# Responses API 参考图内嵌 JSON：多图时按张数分摊体积预算（仍走 /responses，不用 Image API）
RESPONSES_REF_TOTAL_BUDGET_BYTES = 5 * 1024 * 1024


def image_data_url(image, ref_total: int = 1) -> str:
    ref_total = max(1, min(int(ref_total), 10))
    per_image_target = max(256 * 1024, RESPONSES_REF_TOTAL_BUDGET_BYTES // ref_total)
    if ref_total >= 8:
        max_long_edge = 768
    elif ref_total >= 4:
        max_long_edge = 1024
    elif ref_total >= 2:
        max_long_edge = 1280
    else:
        max_long_edge = 1536
    img_bytes, mime = tensor_to_compressed_image_bytes(
        image,
        max_bytes=per_image_target * 2,
        target_ratio=0.5,
        min_dim=256,
        max_long_edge=max_long_edge,
    )
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


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
        detail = _format_api_error(response, data)
        print(f"[AutodL Responses] HTTP {response.status_code} 响应: {response.text[:1500]}")
        raise RuntimeError(f"HTTP {response.status_code} - {detail}")
    return data


def _format_api_error(response, data) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = err.get("message", str(err))
        code = err.get("code")
        err_type = err.get("type")
        details = []
        if code:
            details.append(f"code={code}")
        if err_type:
            details.append(f"type={err_type}")
        if details:
            msg = f"{msg} ({', '.join(details)})"
        return msg
    return response.text[:500]


# OpenAI Responses image_generation 工具支持的编排模型（gpt-5.4 全量不在官方列表内）
GPT_IMAGE2_ORCHESTRATORS = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5",
    "gpt-5-nano",
]
GPT_IMAGE2_ORCHESTRATOR_DEFAULT = "gpt-5.5"


def _build_image_generation_tool(
    *,
    image_model: str,
    quality: str,
    size: str,
    action: str,
) -> dict:
    tool: dict = {"type": "image_generation", "model": image_model}
    if action:
        tool["action"] = action
    q = str(quality).strip() if quality is not None else ""
    if q and q != "auto":
        tool["quality"] = q
    s = str(size).strip() if size is not None else ""
    if s and s != "auto":
        tool["size"] = s
    return tool


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
    tool = _build_image_generation_tool(
        image_model=image_model,
        quality=quality,
        size=size,
        action=action,
    )

    if reference_images:
        ref_total = len(reference_images)
        content = [{"type": "input_text", "text": prompt}]
        for image in reference_images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_data_url(image, ref_total=ref_total),
                }
            )
        input_payload = [{"role": "user", "content": content}]
    else:
        input_payload = prompt

    payload = {
        "model": orchestrator_model,
        "input": input_payload,
        "tools": [tool],
    }
    if not reference_images:
        payload["tool_choice"] = {"type": "image_generation"}
    return payload


def _log_responses_request_summary(
    *,
    action: str,
    size: str,
    quality: str,
    reference_images,
    payload: dict,
) -> None:
    ref_count = len(reference_images or [])
    mode = "I2I" if ref_count else "T2I"
    try:
        body = json.dumps(payload, ensure_ascii=False)
        body_kb = len(body.encode("utf-8")) // 1024
    except Exception:
        body_kb = -1
    print(
        f"[AutodL Responses] {mode} | refs={ref_count} | action={action} | "
        f"size={size} | quality={quality} | payload≈{body_kb}KB"
    )


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
    _log_responses_request_summary(
        action=action,
        size=size,
        quality=quality,
        reference_images=reference_images,
        payload=payload,
    )
    print(f"[AutodL Responses] orchestrator={orchestrator_model}")
    data = post_json(AUTODL_RESPONSES_URL, api_key, payload)
    image_b64 = extract_responses_image_b64(data)
    return bytes_to_image_tensor(base64.b64decode(image_b64)), data


# ---------------------------------------------------------------------------
# OpenAI Image API（与 ComfyUI 内置节点、OpenAI 官方一致，无需 orchestrator）
#   文生图: POST /images/generations  (application/json)
#   图生图: POST /images/edits        (multipart/form-data)
# ---------------------------------------------------------------------------

def post_multipart(url: str, api_key: str, data: dict, files: list) -> dict:
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    response = requests.post(
        url,
        headers=headers,
        data=data,
        files=files,
        timeout=REQUEST_TIMEOUT,
        proxies=REQUEST_PROXIES,
    )
    try:
        parsed = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP {response.status_code} 非 JSON: {response.text[:500]}") from exc
    if response.status_code == 401:
        raise RuntimeError(
            "401 密钥无效或未授权。请到 https://www.autodl.art/large-model/tokens 检查令牌。"
        )
    if not response.ok:
        detail = _format_api_error(response, parsed)
        print(f"[AutodL Images] HTTP {response.status_code} 响应: {response.text[:1500]}")
        raise RuntimeError(f"HTTP {response.status_code} - {detail}")
    return parsed


def extract_image_api_b64(response_data: dict):
    """Image API 返回 data[0].b64_json（gpt-image 系列），兜底支持 url。"""
    data = response_data.get("data") if isinstance(response_data, dict) else None
    if not data:
        raise ValueError("Image API 响应中未找到 data")
    first = data[0] or {}
    b64 = first.get("b64_json")
    if b64:
        return bytes_to_image_tensor(base64.b64decode(b64))
    url = first.get("url")
    if url:
        if not HAS_REQUESTS:
            raise RuntimeError("缺少 requests")
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, proxies=REQUEST_PROXIES)
        resp.raise_for_status()
        return bytes_to_image_tensor(resp.content)
    raise ValueError("Image API 响应中既无 b64_json 也无 url")


def _build_image_api_params(*, image_model: str, prompt: str, quality: str, size: str, n: int) -> dict:
    params = {"model": image_model, "prompt": prompt, "n": int(n)}
    q = str(quality).strip() if quality is not None else ""
    if q:
        params["quality"] = q
    s = str(size).strip() if size is not None else ""
    if s:
        params["size"] = s
    return params


def call_images_generation(
    api_key: str,
    prompt: str,
    *,
    image_model: str,
    quality: str,
    size: str,
    n: int = 1,
):
    """OpenAI Image API 文生图（/images/generations）。"""
    payload = _build_image_api_params(
        image_model=image_model, prompt=prompt, quality=quality, size=size, n=n
    )
    print(
        f"[AutodL Images] T2I | model={image_model} | size={size} | quality={quality} | n={n}"
    )
    data = post_json(AUTODL_IMAGES_GENERATIONS_URL, api_key, payload)
    return extract_image_api_b64(data), data


def call_images_edit(
    api_key: str,
    prompt: str,
    *,
    image_model: str,
    quality: str,
    size: str,
    reference_images,
    n: int = 1,
):
    """OpenAI Image API 图生图（/images/edits，multipart 上传参考图）。"""
    refs = [img for img in (reference_images or []) if img is not None]
    if not refs:
        raise ValueError("图生图至少需要一张参考图")

    params = _build_image_api_params(
        image_model=image_model, prompt=prompt, quality=quality, size=size, n=n
    )
    # multipart 的 data 字段值必须为字符串
    data = {k: str(v) for k, v in params.items()}

    files = []
    multi = len(refs) > 1
    field = "image[]" if multi else "image"
    for i, img in enumerate(refs):
        img_bytes, mime = tensor_to_compressed_image_bytes(
            img, max_bytes=8 * 1024 * 1024, target_ratio=0.7, min_dim=512, max_long_edge=2048
        )
        ext = "jpg" if mime == "image/jpeg" else "png"
        files.append((field, (f"image_{i}.{ext}", img_bytes, mime)))

    print(
        f"[AutodL Images] I2I | model={image_model} | refs={len(refs)} | "
        f"size={size} | quality={quality} | n={n}"
    )
    resp = post_multipart(AUTODL_IMAGES_EDITS_URL, api_key, data, files)
    return extract_image_api_b64(resp), resp


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
    "1:2",
    "2:1",
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
        "1:2": "1024x2048",
        "2:1": "2048x1024",
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
        "1:2": "1024x2048",
        "2:1": "2048x1024",
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
        "1:2": "1920x3840",
        "2:1": "3840x1920",
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

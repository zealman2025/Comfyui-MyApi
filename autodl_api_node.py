import json
import io
import base64
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

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    genai_types = None

AUTODL_OPENAI_BASE = "https://www.autodl.art/api/v1"
AUTODL_GEMINI_BASE = "https://www.autodl.art/api/v1/gemini"

AUTODL_MODELS = (
    "qwen3.6-plus",
    "Qwen3.5-397B-A17B",
    "Kimi-K2.5",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
)

GEMINI_MODEL_ID = "gemini-3.1-pro-preview"


class AutodlApiNode:
    def _normalize_api_key(self, key: str) -> str:
        """
        去掉首尾空白；若用户粘贴了「Bearer xxx」，与请求头里的 Bearer 重复会导致 401，这里剥掉一层前缀。
        同时去掉成对引号。
        """
        if not key:
            return ""
        k = key.strip()
        if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
            k = k[1:-1].strip()
        low = k.lower()
        if low.startswith("bearer "):
            k = k[7:].strip()
        return k

    def _get_api_key(self, input_api_key):
        """仅从节点输入读取 API 密钥。"""
        invalid_placeholders = [
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥",
            "",
        ]
        if input_api_key and input_api_key.strip() and input_api_key.strip() not in invalid_placeholders:
            return self._normalize_api_key(input_api_key)
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(AUTODL_MODELS), {"default": AUTODL_MODELS[0]}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                # 不参与 API；用于 ComfyUI 缓存键。0=每次运行视为变化（可配合界面「随机化」）；非 0 为固定值可复现缓存
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(
        cls,
        api_key,
        model,
        system_prompt,
        user_prompt,
        seed,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
    ):
        """避免相同输入下二次运行被误判为未变化而无输出；seed=0 时每次随机键强制重跑。"""
        key_seed = random.random() if int(seed) == 0 else int(seed)
        return (
            key_seed,
            model,
            system_prompt or "",
            user_prompt or "",
            image is not None,
            image_2 is not None,
            image_3 is not None,
            image_4 is not None,
            image_5 is not None,
        )

    def _check_deps(self):
        missing = []
        if not HAS_NUMPY and not HAS_TORCH:
            missing.append("numpy 或 torch")
        if not HAS_PIL:
            missing.append("Pillow")
        if not HAS_REQUESTS:
            missing.append("requests")
        return missing

    def _encode_image_to_base64(self, image):
        if not HAS_PIL:
            raise ImportError("缺少 Pillow")
        if not HAS_NUMPY and not HAS_TORCH:
            raise ImportError("缺少 numpy 或 torch")
        if image is None:
            raise ValueError("Image is None")

        if HAS_TORCH and isinstance(image, torch.Tensor):
            if image.is_cuda:
                image = image.cpu()
            image = image.numpy()

        if HAS_NUMPY and isinstance(image, np.ndarray):
            if len(image.shape) == 4:
                image = image[0] if image.shape[0] >= 1 else image
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = image[:, :, :3]
                elif image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            pil_image = Image.fromarray(image.astype(np.uint8), "RGB")
        elif HAS_PIL and isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        max_bytes = 10 * 1024 * 1024
        target_raw_bytes = int(max_bytes * 0.7)
        min_dim = 512

        def save_to_buffer(img, fmt="JPEG", **save_kwargs):
            buf = io.BytesIO()
            img.save(buf, format=fmt, **save_kwargs)
            return buf, buf.tell()

        buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=95, optimize=True)
        resize_attempts = 0
        while raw_size > target_raw_bytes and (pil_image.width > min_dim or pil_image.height > min_dim) and resize_attempts < 5:
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
            raise ValueError("图片压缩后仍过大，请换小图或手动缩小。")

        buffer.seek(0)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8"), img_bytes

    def _call_openai_compatible(self, api_key, model_id, system_prompt, user_prompt, image_slots):
        """使用 requests 直连 /chat/completions，与 AutodL 文档中的 curl 一致（避免 OpenAI SDK 附带额外请求头导致 401）。"""
        if not HAS_REQUESTS:
            return "Error: 未安装 requests，请执行: pip install requests"

        messages = []
        if system_prompt and str(system_prompt).strip():
            messages.append({"role": "system", "content": str(system_prompt).strip()})

        user_content = []
        for im in image_slots:
            if im is None:
                continue
            try:
                b64, _ = self._encode_image_to_base64(im)
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            except Exception as e:
                return f"Error: 处理图片失败: {e}"

        user_content.append(
            {"type": "text", "text": str(user_prompt) if user_prompt is not None else ""}
        )
        messages.append({"role": "user", "content": user_content})

        url = f"{AUTODL_OPENAI_BASE.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"model": model_id, "messages": messages}

        try:
            r = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=600,
                proxies={"http": None, "https": None},
            )
        except requests.RequestException as e:
            return f"Error: 请求失败: {e}"

        if r.status_code == 401:
            return (
                "Error: OpenAI 兼容接口返回 401（密钥无效或未授权该接口）。"
                "请到 https://www.autodl.art/large-model/tokens 创建/检查令牌，并确认对「OpenAI 兼容 / chat completions」可用；"
                "节点里只填裸密钥，不要带「Bearer 」（本节点会自动去掉误贴的前缀）。"
                f" 响应片段: {r.text[:400]}"
            )

        try:
            data = r.json()
        except Exception:
            return f"Error: HTTP {r.status_code}，且响应非 JSON: {r.text[:500]}"

        if not r.ok:
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                msg = err.get("message", str(err))
            else:
                msg = r.text[:500]
            return f"Error: HTTP {r.status_code} - {msg}"

        if isinstance(data, dict) and data.get("choices"):
            msg0 = (data["choices"][0] or {}).get("message") or {}
            content = msg0.get("content")
            return content if content is not None else ""

        if isinstance(data, dict) and data.get("error"):
            return f"Error: {data.get('error')}"

        return json.dumps(data, ensure_ascii=False)

    def _call_gemini(self, api_key, system_prompt, user_prompt, image_slots):
        if not HAS_GOOGLE_GENAI:
            return "Error: 未安装 google-genai，请执行: pip install google-genai"

        client = genai.Client(
            api_key=api_key,
            http_options={"base_url": AUTODL_GEMINI_BASE},
        )

        parts = []
        for im in image_slots:
            if im is None:
                continue
            try:
                _, jpeg_bytes = self._encode_image_to_base64(im)
                parts.append(genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
            except Exception as e:
                return f"Error: 处理图片失败: {e}"

        text = str(user_prompt) if user_prompt is not None else ""
        if text.strip():
            parts.append(text)

        if not parts:
            parts.append(" ")

        config = None
        if system_prompt and str(system_prompt).strip():
            config = genai_types.GenerateContentConfig(system_instruction=str(system_prompt).strip())

        kwargs = {"model": GEMINI_MODEL_ID, "contents": parts}
        if config is not None:
            kwargs["config"] = config

        response = client.models.generate_content(**kwargs)
        if response is None:
            return ""
        t = getattr(response, "text", None)
        if t is not None:
            return t
        return str(response)

    def process(
        self,
        api_key,
        model,
        system_prompt,
        user_prompt,
        seed,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
    ):
        _ = seed  # 仅用于 IS_CHANGED / 界面随机化，不传给远端 API
        missing = self._check_deps()
        if missing:
            return (f"Error: 缺少依赖: {', '.join(missing)}",)

        key = self._get_api_key(api_key)
        if not key:
            return ("Error: 请在节点中填写 AutodL API 密钥。",)

        slots = [image, image_2, image_3, image_4, image_5]

        try:
            if model == GEMINI_MODEL_ID:
                out = self._call_gemini(key, system_prompt, user_prompt, slots)
            else:
                out = self._call_openai_compatible(key, model, system_prompt, user_prompt, slots)
            if isinstance(out, str) and out.startswith("Error:"):
                return (out,)
            return (out,)
        except Exception as e:
            print(f"[AutodL] {e}")
            print(traceback.format_exc())
            return (f"Error: {e}",)


NODE_CLASS_MAPPINGS = {
    "AutodlApiNode": AutodlApiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutodlApiNode": "🍎AutodL API",
}

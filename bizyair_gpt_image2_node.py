import json
import io
import traceback

try:
    from .bizyair_upload import upload_image_to_bizyair
except ImportError:
    from bizyair_upload import upload_image_to_bizyair

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not found. Please install it with: pip install requests")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL library not found. Please install it with: pip install Pillow")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy library not found. Please install it with: pip install numpy")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch library not found. Some features may not work properly.")


GPT_IMAGE_2_API_URL = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"

GPT_IMAGE_2_T2I_WEB_APP_ID = 52330
GPT_IMAGE_2_I2I_WEB_APP_ID = 52304

GPT_IMAGE_2_T2I_NODE_PREFIX = "56:BizyAir_GPT_IMAGE_2_T2I_API"
GPT_IMAGE_2_I2I_NODE_PREFIX = "55:BizyAir_GPT_IMAGE_2_I2I_API"

GPT_IMAGE_2_I2I_REF_NODE_KEYS = [
    "40:LoadImage.image",
    "37:LoadImage.image",
    "39:LoadImage.image",
    "46:LoadImage.image",
]

GPT_IMAGE_2_ASPECT_RATIOS = [
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


def _get_api_key(input_api_key, log_prefix):
    invalid_placeholders = [
        "YOUR_API_KEY",
        "你的apikey",
        "your_api_key_here",
        "请输入API密钥",
        "请输入你的API密钥",
    ]
    if (
        input_api_key
        and input_api_key.strip()
        and input_api_key.strip() not in invalid_placeholders
    ):
        print(f"[{log_prefix}] 使用节点中的 API 密钥")
        return input_api_key.strip()
    return ""


def _check_dependencies():
    missing_deps = []
    if not HAS_PIL:
        missing_deps.append("Pillow")
    if not HAS_NUMPY:
        missing_deps.append("numpy")
    if not HAS_REQUESTS:
        missing_deps.append("requests")
    return missing_deps


def _image_to_bytes(image):
    """将图像转换为字节流（用于 OSS 上传），返回 (bytes, file_ext)。"""
    try:
        if not HAS_PIL or not HAS_NUMPY:
            return None, None

        if HAS_TORCH and hasattr(image, "cpu"):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        if len(image_np.shape) == 4:
            image_np = image_np[0]

        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            pil_image = Image.fromarray(image_np, "RGB")
        else:
            raise ValueError(f"Unsupported image shape: {image_np.shape}")

        max_bytes = 10 * 1024 * 1024
        target_raw_bytes = int(max_bytes * 0.7)
        min_dim = 512

        def save_to_buffer(img, fmt="PNG", **save_kwargs):
            buf = io.BytesIO()
            img.save(buf, format=fmt, **save_kwargs)
            return buf, buf.tell()

        buffer, raw_size = save_to_buffer(pil_image, "PNG", optimize=True)
        image_format = "PNG"

        if raw_size > target_raw_bytes:
            print(
                f"Warning: Image raw size ({raw_size / 1024 / 1024:.2f}MB) exceeds target. Compressing..."
            )

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
            buffer, raw_size = save_to_buffer(pil_image, "PNG", optimize=True)

        if raw_size > target_raw_bytes:
            quality = 90
            while raw_size > target_raw_bytes and quality >= 40:
                buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=quality, optimize=True)
                image_format = "JPEG"
                quality -= 5

        if raw_size > target_raw_bytes:
            raise ValueError(
                f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). "
                "Please use a smaller image or resize manually."
            )

        buffer.seek(0)
        img_bytes = buffer.getvalue()
        ext = "jpg" if image_format == "JPEG" else "png"
        print(f"Image prepared for OSS: {raw_size / 1024 / 1024:.2f}MB, format: {image_format}")
        return img_bytes, ext

    except Exception as e:
        print(f"Error converting image to bytes: {str(e)}")
        print(traceback.format_exc())
        return None, None


def _decode_image_from_url(image_url):
    if not HAS_REQUESTS or not HAS_PIL or not HAS_NUMPY:
        raise Exception("Missing required dependencies")

    print(f"Downloading image from URL: {image_url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(image_url, headers=headers, timeout=30)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = image_np[np.newaxis, ...]

    if HAS_TORCH:
        image_tensor = torch.from_numpy(image_tensor)

    print(f"Successfully converted image to ComfyUI format: {image_tensor.shape}")
    return image_tensor


def _extract_api_error(result):
    error_message = result.get("status", "Unknown error")
    outputs = result.get("outputs", [])
    if outputs and len(outputs) > 0:
        error_output = outputs[0]
        error_msg = error_output.get("error_msg", "")
        error_type = error_output.get("error_type", "")
        if error_msg:
            error_message = f"{error_message}: {error_msg}"
        if error_type:
            error_message = f"{error_message} (类型: {error_type})"

    if error_message == result.get("status", "Unknown error"):
        error_detail = result.get("error", {})
        if isinstance(error_detail, dict):
            error_msg = error_detail.get("message", error_detail.get("msg", ""))
            if error_msg:
                error_message = f"{error_message}: {error_msg}"
    return error_message


def _debug_payload(data, input_values):
    debug_data = data.copy()
    debug_input_values = {}
    for key, value in input_values.items():
        if isinstance(value, str) and (
            "storage.bizyair.cn" in value or "aliyuncs.com" in value
        ):
            debug_input_values[key] = f"[OSS URL: {value[:60]}...]"
        else:
            debug_input_values[key] = value
    debug_data["input_values"] = debug_input_values
    return debug_data


class BizyAirGPTImage2T2INode:
    """BizyAir GPT-IMAGE-2 文生图节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "9:16"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    def generate(self, api_key, prompt, aspect_ratio):
        log_prefix = "BizyAirGPTImage2T2I"
        actual_api_key = _get_api_key(api_key, log_prefix)
        if not actual_api_key:
            raise Exception("请在节点中填写 BizyAir API 密钥。请访问 https://bizyair.cn 获取。")

        missing_deps = _check_dependencies()
        if missing_deps:
            raise Exception(
                f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。"
            )

        if not prompt or not prompt.strip():
            raise Exception("请输入文生图提示词")

        try:
            print(
                f"[{log_prefix}] BizyAir GPT-IMAGE-2 T2I request to: "
                f"{GPT_IMAGE_2_API_URL} (web_app_id: {GPT_IMAGE_2_T2I_WEB_APP_ID})"
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}",
            }

            input_values = {
                f"{GPT_IMAGE_2_T2I_NODE_PREFIX}.prompt": prompt,
                f"{GPT_IMAGE_2_T2I_NODE_PREFIX}.aspect_ratio": aspect_ratio,
            }

            data = {
                "web_app_id": GPT_IMAGE_2_T2I_WEB_APP_ID,
                "suppress_preview_output": False,
                "input_values": input_values,
            }

            print(
                f"[{log_prefix}] Prompt: {prompt[:100]}..., Aspect: {aspect_ratio}"
            )
            print(
                f"[{log_prefix}] Request payload: "
                f"{json.dumps(_debug_payload(data, input_values), indent=2, ensure_ascii=False)}"
            )

            response = requests.post(GPT_IMAGE_2_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()

            result = response.json()
            print(f"[{log_prefix}] API response received")
            print(
                f"[{log_prefix}] API response: "
                f"{json.dumps(result, indent=2, ensure_ascii=False)}"
            )

            if result.get("status") != "Success":
                error_message = _extract_api_error(result)
                print(
                    f"[{log_prefix}] API错误详情: "
                    f"{json.dumps(result, indent=2, ensure_ascii=False)}"
                )
                raise Exception(f"API请求失败: {error_message}")

            outputs = result.get("outputs", [])
            if not outputs:
                raise Exception("API响应中没有找到输出数据")

            image_url = outputs[0].get("object_url")
            if not image_url:
                raise Exception("API响应中没有找到图像URL")

            print(f"[{log_prefix}] Generated image URL: {image_url}")

            output_image = _decode_image_from_url(image_url)

            cost_time = result.get("cost_times", {}).get("total_cost_time", 0)
            request_id = result.get("request_id", "")

            status_text = (
                f"✅ GPT-IMAGE-2 T2I 生成成功\n"
                f"提示词: {prompt[:50]}...\n"
                f"宽高比: {aspect_ratio}\n"
                f"耗时: {cost_time}ms\n"
                f"请求ID: {request_id}"
            )

            return (output_image, status_text)

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求错误: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise Exception(error_msg)


class BizyAirGPTImage2I2INode:
    """BizyAir GPT-IMAGE-2 图生图节点（最多 4 张参考图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (GPT_IMAGE_2_ASPECT_RATIOS, {"default": "9:16"}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    def generate(
        self,
        api_key,
        prompt,
        aspect_ratio,
        inputcount,
        image=None,
        image2=None,
        image3=None,
        image4=None,
    ):
        log_prefix = "BizyAirGPTImage2I2I"
        actual_api_key = _get_api_key(api_key, log_prefix)
        if not actual_api_key:
            raise Exception("请在节点中填写 BizyAir API 密钥。请访问 https://bizyair.cn 获取。")

        missing_deps = _check_dependencies()
        if missing_deps:
            raise Exception(
                f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。"
            )

        if not prompt or not prompt.strip():
            raise Exception("请输入图生图提示词")

        if image is None:
            raise Exception("GPT-IMAGE-2 图生图至少需要一张参考图 (image)")

        inputcount = max(1, min(int(inputcount), 4))

        try:
            print(
                f"[{log_prefix}] BizyAir GPT-IMAGE-2 I2I request to: "
                f"{GPT_IMAGE_2_API_URL} (web_app_id: {GPT_IMAGE_2_I2I_WEB_APP_ID})"
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}",
            }

            add_log = lambda t, m: print(f"[{log_prefix}][{t}] {m}")

            ref_images = [image, image2, image3, image4]
            ref_images = ref_images[:inputcount]

            input_values = {
                f"{GPT_IMAGE_2_I2I_NODE_PREFIX}.prompt": prompt,
                f"{GPT_IMAGE_2_I2I_NODE_PREFIX}.aspect_ratio": aspect_ratio,
            }

            uploaded_count = 0
            for idx, current_image in enumerate(ref_images):
                if current_image is None:
                    continue

                img_bytes, ext = _image_to_bytes(current_image)
                if not img_bytes or not ext:
                    raise Exception(
                        f"参考图 {idx + 1} 转换失败，请检查图像格式"
                    )

                ref_url = upload_image_to_bizyair(
                    img_bytes,
                    actual_api_key,
                    add_log_fn=add_log,
                    file_name=f"gpt_image2_i2i_ref_{idx + 1}.{ext}",
                )
                input_values[GPT_IMAGE_2_I2I_REF_NODE_KEYS[idx]] = ref_url
                uploaded_count += 1
                print(
                    f"[{log_prefix}] Added ref image {idx + 1} "
                    f"(key: {GPT_IMAGE_2_I2I_REF_NODE_KEYS[idx]})"
                )

            if uploaded_count == 0:
                raise Exception("未上传到任何参考图，请至少连接一张图像")

            input_values[f"{GPT_IMAGE_2_I2I_NODE_PREFIX}.inputcount"] = uploaded_count

            data = {
                "web_app_id": GPT_IMAGE_2_I2I_WEB_APP_ID,
                "suppress_preview_output": False,
                "input_values": input_values,
            }

            print(
                f"[{log_prefix}] Prompt: {prompt[:100]}..., Aspect: {aspect_ratio}, "
                f"inputcount: {uploaded_count}"
            )
            print(
                f"[{log_prefix}] Request payload: "
                f"{json.dumps(_debug_payload(data, input_values), indent=2, ensure_ascii=False)}"
            )

            response = requests.post(GPT_IMAGE_2_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()

            result = response.json()
            print(f"[{log_prefix}] API response received")
            print(
                f"[{log_prefix}] API response: "
                f"{json.dumps(result, indent=2, ensure_ascii=False)}"
            )

            if result.get("status") != "Success":
                error_message = _extract_api_error(result)
                print(
                    f"[{log_prefix}] API错误详情: "
                    f"{json.dumps(result, indent=2, ensure_ascii=False)}"
                )
                raise Exception(f"API请求失败: {error_message}")

            outputs = result.get("outputs", [])
            if not outputs:
                raise Exception("API响应中没有找到输出数据")

            image_url = outputs[0].get("object_url")
            if not image_url:
                raise Exception("API响应中没有找到图像URL")

            print(f"[{log_prefix}] Generated image URL: {image_url}")

            output_image = _decode_image_from_url(image_url)

            cost_time = result.get("cost_times", {}).get("total_cost_time", 0)
            request_id = result.get("request_id", "")

            status_text = (
                f"✅ GPT-IMAGE-2 I2I 生成成功\n"
                f"提示词: {prompt[:50]}...\n"
                f"宽高比: {aspect_ratio}, 参考图: {uploaded_count}\n"
                f"耗时: {cost_time}ms\n"
                f"请求ID: {request_id}"
            )

            return (output_image, status_text)

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求错误: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise Exception(error_msg)


NODE_CLASS_MAPPINGS = {
    "BizyAirGPTImage2T2INode": BizyAirGPTImage2T2INode,
    "BizyAirGPTImage2I2INode": BizyAirGPTImage2I2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirGPTImage2T2INode": "🌐BizyAir GPT-IMAGE-2 文生图",
    "BizyAirGPTImage2I2INode": "🌐BizyAir GPT-IMAGE-2 图生图",
}

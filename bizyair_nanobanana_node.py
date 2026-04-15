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

class BizyAirNanoBananaProNode:
    """
    BizyAir NanoBananaPro专用节点
    专门用于调用BizyAir的NanoBananaPro模型API
    """

    def _get_api_key(self, input_api_key):
        """仅从节点输入读取 API 密钥。"""
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
            print("[BizyAirNanoBananaPro] 使用节点中的 API 密钥")
            return input_api_key.strip()
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "9:16"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "inputcount": ("INT", {"default": 6, "min": 1, "max": 6, "step": 1}),
                "mode": (["third-party", "official"], {"default": "third-party"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    def _check_dependencies(self):
        """检查必要的依赖是否已安装"""
        missing_deps = []
        
        if not HAS_PIL:
            missing_deps.append("Pillow")
            
        if not HAS_NUMPY:
            missing_deps.append("numpy")
            
        if not HAS_REQUESTS:
            missing_deps.append("requests")
            
        return missing_deps

    def _image_to_bytes(self, image):
        """将图像转换为字节流（用于 OSS 上传），返回 (bytes, file_ext)"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                return None, None
            
            if HAS_TORCH and hasattr(image, 'cpu'):
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
                pil_image = Image.fromarray(image_np, 'RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image_np.shape}")
            
            max_bytes = 10 * 1024 * 1024
            target_raw_bytes = int(max_bytes * 0.7)
            min_dim = 512

            def save_to_buffer(img, fmt='PNG', **save_kwargs):
                buf = io.BytesIO()
                img.save(buf, format=fmt, **save_kwargs)
                return buf, buf.tell()

            buffer, raw_size = save_to_buffer(pil_image, 'PNG', optimize=True)
            image_format = 'PNG'

            if raw_size > target_raw_bytes:
                print(f"Warning: Image raw size ({raw_size / 1024 / 1024:.2f}MB) exceeds target. Compressing...")
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
                buffer, raw_size = save_to_buffer(pil_image, 'PNG', optimize=True)

            if raw_size > target_raw_bytes:
                quality = 90
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_to_buffer(pil_image, 'JPEG', quality=quality, optimize=True)
                    image_format = 'JPEG'
                    quality -= 5

            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")

            buffer.seek(0)
            img_bytes = buffer.getvalue()
            ext = 'jpg' if image_format == 'JPEG' else 'png'
            print(f"Image prepared for OSS: {raw_size / 1024 / 1024:.2f}MB, format: {image_format}")
            return img_bytes, ext
            
        except Exception as e:
            print(f"Error converting image to bytes: {str(e)}")
            print(traceback.format_exc())
            return None, None

    def _decode_image_from_url(self, image_url):
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            if not HAS_REQUESTS or not HAS_PIL or not HAS_NUMPY:
                raise Exception("Missing required dependencies")
            
            print(f"Downloading image from URL: {image_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 将响应内容转换为PIL图像
            image = Image.open(io.BytesIO(response.content))
            
            # 确保是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 添加批次维度 (1, H, W, 3)
            image_tensor = image_np[np.newaxis, ...]
            
            # 如果有torch，转换为torch张量
            if HAS_TORCH:
                image_tensor = torch.from_numpy(image_tensor)
            
            print(f"Successfully converted image to ComfyUI format: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            print(f"Error downloading/converting image: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate(self, api_key, prompt, aspect_ratio, resolution, inputcount, mode,
                 image=None, image2=None, image3=None, image4=None, image5=None, image6=None):
        """生成图像（与主插件 ZealmanAIforPS NanoBanana2 6图 API 一致）"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            raise Exception("请在节点中填写 BizyAir API 密钥。请访问 https://bizyair.cn 获取。")

        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
        
        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir NanoBanana2 API request to: {api_url} (web_app_id: 47114)")
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 主图必须存在
            if image is None:
                raise Exception("NanoBanana2 需要至少一张主图 (image)")
            
            add_log = lambda t, m: print(f"[BizyAirNanoBanana2][{t}] {m}")
            
            # 上传主图
            primary_bytes, primary_ext = self._image_to_bytes(image)
            if not primary_bytes or not primary_ext:
                raise Exception("主图转换失败，请检查图像格式")
            primary_url = upload_image_to_bizyair(
                primary_bytes, actual_api_key,
                add_log_fn=add_log,
                file_name=f"nanobanana_primary.{primary_ext}"
            )
            
            # 参考图节点（与 API 一致：37,39,46,48,52）
            ref_node_keys = ["37:LoadImage.image", "39:LoadImage.image", "46:LoadImage.image", "48:LoadImage.image", "52:LoadImage.image"]
            input_images = [image2, image3, image4, image5, image6]
            
            # 根据 inputcount 只使用前 N 张图（主图 + inputcount-1 张参考图）
            max_refs = min(inputcount - 1, 5)
            
            # 构建 input_values（35:BizyAir_NanoBanana2.*，与 API 示例一致）
            input_values = {
                "40:LoadImage.image": primary_url,
                "35:BizyAir_NanoBanana2.prompt": prompt,
                "35:BizyAir_NanoBanana2.aspect_ratio": aspect_ratio or "auto",
                "35:BizyAir_NanoBanana2.resolution": resolution,
                "35:BizyAir_NanoBanana2.inputcount": inputcount,
                "35:BizyAir_NanoBanana2.mode": "official" if mode == "official" else "third-party"
            }
            
            # 上传参考图（最多 max_refs 张）
            ref_count = 0
            for idx in range(max_refs):
                if idx >= len(input_images):
                    break
                img = input_images[idx]
                if img is not None and idx < len(ref_node_keys):
                    img_bytes, ext = self._image_to_bytes(img)
                    if img_bytes and ext:
                        try:
                            ref_url = upload_image_to_bizyair(
                                img_bytes, actual_api_key,
                                add_log_fn=add_log,
                                file_name=f"nanobanana_ref_{idx + 1}.{ext}"
                            )
                            input_values[ref_node_keys[idx]] = ref_url
                            ref_count += 1
                            print(f"Added ref image {idx + 1} (key: {ref_node_keys[idx]})")
                        except Exception as up_err:
                            print(f"Warning: Failed to upload ref image {idx + 1}: {up_err}")
            
            # 实际发送的图片数需与 inputcount 一致，若参考图不足则使用实际数量
            actual_count = 1 + ref_count
            if actual_count < inputcount:
                print(f"Warning: inputcount={inputcount} but only {actual_count} images provided. Using actual count.")
                input_values["35:BizyAir_NanoBanana2.inputcount"] = actual_count
            
            # 构建请求数据
            data = {
                "web_app_id": 47114,  # NanoBanana2 6图，与主插件一致
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}, inputcount={input_values['35:BizyAir_NanoBanana2.inputcount']}, refs={ref_count}")
            print(f"Prompt: {prompt[:100]}..., Aspect: {aspect_ratio}, Resolution: {resolution}, Mode: {mode}")
            print(f"Input values keys: {list(input_values.keys())}")
            # 打印请求数据（隐藏 OSS URL 详情）
            debug_data = data.copy()
            debug_input_values = {}
            for key, value in input_values.items():
                if isinstance(value, str) and ('storage.bizyair.cn' in value or 'aliyuncs.com' in value):
                    debug_input_values[key] = f"[OSS URL: {value[:60]}...]"
                else:
                    debug_input_values[key] = value
            debug_data['input_values'] = debug_input_values
            print(f"Request payload: {json.dumps(debug_data, indent=2, ensure_ascii=False)}")
            
            # 发送请求（增加超时时间到120秒）
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            print("API response received")
            print(f"API response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查响应状态
            if result.get("status") != "Success":
                # 尝试获取详细的错误信息
                error_message = result.get("status", "Unknown error")
                
                # 从outputs中提取错误信息
                outputs = result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    error_output = outputs[0]
                    error_msg = error_output.get("error_msg", "")
                    error_type = error_output.get("error_type", "")
                    if error_msg:
                        error_message = f"{error_message}: {error_msg}"
                    if error_type:
                        error_message = f"{error_message} (类型: {error_type})"
                
                # 如果没有从outputs获取到，尝试从error字段获取
                if error_message == result.get("status", "Unknown error"):
                    error_detail = result.get("error", {})
                    if isinstance(error_detail, dict):
                        error_msg = error_detail.get("message", error_detail.get("msg", ""))
                        if error_msg:
                            error_message = f"{error_message}: {error_msg}"
                
                # 打印完整的错误信息用于调试
                print(f"API错误详情: {json.dumps(result, indent=2, ensure_ascii=False)}")
                raise Exception(f"API请求失败: {error_message}")
            
            # 提取图像URL
            outputs = result.get("outputs", [])
            if not outputs:
                raise Exception("API响应中没有找到输出数据")
            
            image_url = outputs[0].get("object_url")
            if not image_url:
                raise Exception("API响应中没有找到图像URL")
            
            print(f"Generated image URL: {image_url}")
            
            # 下载并转换图像
            output_image = self._decode_image_from_url(image_url)
            
            # 构建状态信息
            status_info = {
                "status": "success",
                "web_app_id": 47114,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "inputcount": input_values["35:BizyAir_NanoBanana2.inputcount"],
                "mode": mode,
                "image_url": image_url,
                "cost_time": result.get("cost_times", {}).get("total_cost_time", 0),
                "request_id": result.get("request_id", "")
            }
            
            status_text = f"✅ NanoBanana2 生成成功\n"
            status_text += f"提示词: {prompt[:50]}...\n"
            status_text += f"宽高比: {aspect_ratio}, 分辨率: {resolution}, inputcount: {status_info['inputcount']}, 模式: {mode}\n"
            status_text += f"耗时: {status_info['cost_time']}ms\n"
            status_text += f"请求ID: {status_info['request_id']}"
            
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

# 节点映射
NODE_CLASS_MAPPINGS = {
    "BizyAirNanoBananaProNode": BizyAirNanoBananaProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirNanoBananaProNode": "🌐BizyAir NanoBanana2 (与主插件一致)"
}

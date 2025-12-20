import os
import json
import io
import base64
import traceback
import random

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

    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

    def _get_api_key(self, input_api_key):
        """获取API密钥，优先使用输入的密钥，否则从config.json读取"""
        # 定义无效的占位符文本
        invalid_placeholders = [
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥"
        ]

        # 如果输入了有效的API密钥，优先使用
        if (input_api_key and
            input_api_key.strip() and
            input_api_key.strip() not in invalid_placeholders):
            print(f"[BizyAirNanoBananaPro] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('bizyair_api_key', '').strip()
                if config_api_key:
                    print(f"[BizyAirNanoBananaPro] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[BizyAirNanoBananaPro] config.json中未找到bizyair_api_key")
                    return ''
        except Exception as e:
            print(f"[BizyAirNanoBananaPro] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {"default": "generate"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "auto"}),
                "resolution": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "quality": (["standard", "high"], {"default": "high"}),
                "character_consistency": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
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

    def _image_to_base64(self, image):
        """将图像转换为base64编码"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                return None
            
            # 确保图像是numpy数组
            if HAS_TORCH and hasattr(image, 'cpu'):
                # 如果是torch张量，转换为numpy
                image_np = image.cpu().numpy()
            else:
                image_np = image
            
            # 确保数据类型和范围正确
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            
            # 处理批次维度
            if len(image_np.shape) == 4:
                image_np = image_np[0]  # 取第一张图像
            
            # 确保是RGB格式
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                pil_image = Image.fromarray(image_np, 'RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image_np.shape}")
            
            # 控制图像体积，避免Base64超过服务端限制
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
                print(f"Warning: Image raw size ({raw_size / 1024 / 1024:.2f}MB) exceeds target {target_raw_bytes / 1024 / 1024:.2f}MB. Compressing...")

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
                print(f"Resized image attempt {resize_attempts}: {new_width}x{new_height}")
                buffer, raw_size = save_to_buffer(pil_image, 'PNG', optimize=True)

            if raw_size > target_raw_bytes:
                print("PNG still too large, switching to JPEG compression...")
                quality = 90
                jpeg_attempts = 0
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_to_buffer(pil_image, 'JPEG', quality=quality, optimize=True)
                    image_format = 'JPEG'
                    jpeg_attempts += 1
                    print(f"JPEG compression attempt {jpeg_attempts}: quality={quality}, size={raw_size / 1024 / 1024:.2f}MB")
                    quality -= 5

            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")

            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_size_mb = len(image_base64) / 1024 / 1024
            print(f"Final raw size: {raw_size / 1024 / 1024:.2f}MB, base64 size: {base64_size_mb:.2f}MB, format: {image_format}")

            mime_type = 'image/jpeg' if image_format == 'JPEG' else 'image/png'
            return f"data:{mime_type};base64,{image_base64}"
            
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            print(traceback.format_exc())
            return None

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

    def generate(self, api_key, prompt, operation, seed, temperature, top_p, max_tokens, 
                 aspect_ratio, resolution, quality, character_consistency,
                 image=None, image2=None, image3=None, image4=None, image5=None):
        """生成图像"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            raise Exception("请输入API密钥或在config.json中配置bizyair_api_key。请访问 https://bizyair.cn 获取API密钥。")

        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")

        # 生成随机种子（如果需要）
        if seed == 0:
            seed = random.randint(1, 2147483647)  # API要求seed最大值为2147483647
        
        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir NanoBananaPro API request to: {api_url}")
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 构建input_values
            input_values = {
                "17:BizyAir_NanoBananaPro.prompt": prompt,
                "17:BizyAir_NanoBananaPro.operation": operation,
                "17:BizyAir_NanoBananaPro.temperature": temperature,
                "17:BizyAir_NanoBananaPro.top_p": top_p,
                "17:BizyAir_NanoBananaPro.seed": seed,
                "17:BizyAir_NanoBananaPro.max_tokens": max_tokens,
                "17:BizyAir_NanoBananaPro.aspect_ratio": aspect_ratio,
                "17:BizyAir_NanoBananaPro.resolution": resolution,
                "17:BizyAir_NanoBananaPro.quality": quality,
                "17:BizyAir_NanoBananaPro.character_consistency": character_consistency
            }
            
            # 图片输入键名映射（按顺序）
            image_key_mapping = [
                "18:LoadImage.image",  # image
                "20:LoadImage.image",  # image2
                "21:LoadImage.image",  # image3
                "22:LoadImage.image",  # image4
                "23:LoadImage.image",  # image5
            ]
            
            # 收集所有输入的图片
            input_images = [image, image2, image3, image4, image5]
            image_count = 0
            
            # 处理每个图片输入
            for idx, img in enumerate(input_images):
                if img is not None:
                    image_base64 = self._image_to_base64(img)
                    if image_base64:
                        if idx < len(image_key_mapping):
                            input_values[image_key_mapping[idx]] = image_base64
                            image_count += 1
                            print(f"Added input image {idx + 1} to request (key: {image_key_mapping[idx]})")
                        else:
                            print(f"Warning: Too many images, maximum {len(image_key_mapping)} images supported")
                    else:
                        print(f"Warning: Failed to convert input image {idx + 1} to base64")
            
            # 根据实际图片数量设置 inputcount
            input_count = image_count if image_count > 0 else 2  # 如果没有图片，默认2
            input_values["17:BizyAir_NanoBananaPro.inputcount"] = input_count
            print(f"Input count set to: {input_count} (images provided: {image_count})")
            
            # 构建请求数据
            data = {
                "web_app_id": 41502,  # NanoBananaPro的固定web_app_id
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}, input_values count={len(input_values)}")
            print(f"Operation: {operation}, Prompt: {prompt[:100]}...")
            print(f"Aspect Ratio: {aspect_ratio}, Resolution: {resolution}, Quality: {quality}")
            print(f"Input values keys: {list(input_values.keys())}")
            # 打印请求数据（隐藏base64图片数据）
            debug_data = data.copy()
            debug_input_values = {}
            for key, value in input_values.items():
                if isinstance(value, str) and value.startswith('data:image'):
                    debug_input_values[key] = f"[Base64 Image Data: {len(value)} chars]"
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
                "web_app_id": 41502,
                "operation": operation,
                "prompt": prompt,
                "seed": seed,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "quality": quality,
                "image_url": image_url,
                "cost_time": result.get("cost_times", {}).get("total_cost_time", 0),
                "request_id": result.get("request_id", "")
            }
            
            status_text = f"✅ NanoBananaPro生成成功\n"
            status_text += f"操作模式: {operation}\n"
            status_text += f"提示词: {prompt[:50]}...\n"
            status_text += f"种子: {seed}\n"
            status_text += f"宽高比: {aspect_ratio}, 分辨率: {resolution}, 质量: {quality}\n"
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
    "BizyAirNanoBananaProNode": "🌐BizyAir NanoBanana Pro (需BizyAir.cn充值金币)"
}

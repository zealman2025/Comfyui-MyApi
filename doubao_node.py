import os
import json
import folder_paths
import io
import base64
import traceback
import time
import random
import string

# 尝试导入依赖，但不强制要求
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
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# 从配置文件加载模型配置
def load_models_from_config():
    """从config.json加载模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            doubao_models = models.get('doubao', {})
            return doubao_models
    except Exception as e:
        print(f"[DoubaoMMM] Error loading models from config: {str(e)}")
        import traceback
        traceback.print_exc()
        # 提供默认模型作为回退
        default_models = {
            "doubao-seed-1-6-vision-250815": "doubao-seed-1-6-vision-250815",
            "doubao-seed-1-6-251015": "doubao-seed-1-6-251015",
            "doubao-seed-1-8-251215": "doubao-seed-1-8-251215"
        }
        print(f"[DoubaoMMM] Using default models: {default_models}")
        return default_models

# 加载模型配置
DOUBAO_MODELS = load_models_from_config()

class DoubaoNode:
    def __init__(self):
        self.current_seed = 0  # 初始化种子值
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

    def _get_api_key(self, input_api_key):
        """获取API密钥，优先使用输入的密钥，否则从config.json读取"""
        # 定义无效的占位符文本
        invalid_placeholders = [
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥",
            ""
        ]

        # 如果输入了有效的API密钥，优先使用
        if (input_api_key and
            input_api_key.strip() and
            input_api_key.strip() not in invalid_placeholders):
            print(f"[DoubaoMMM] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('doubao_api_key', '').strip()
                if config_api_key:
                    print(f"[DoubaoMMM] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[DoubaoMMM] config.json中未找到doubao_api_key")
                    return ''
        except Exception as e:
            print(f"[DoubaoMMM] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(DOUBAO_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "图片主要讲了什么?"}),
                "max_completion_tokens": ("INT", {"default": 65535, "min": 1, "max": 65535}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "reasoning_effort": (["minimal", "low", "medium", "high"], {"default": "minimal"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🍎MYAPI"

    def _check_dependencies(self):
        """检查必要的依赖是否已安装"""
        missing_deps = []
        
        if not HAS_NUMPY:
            missing_deps.append("numpy")
        
        if not HAS_PIL:
            missing_deps.append("Pillow")
            
        if not HAS_REQUESTS:
            missing_deps.append("requests")
            
        return missing_deps

    def _debug_image_info(self, image):
        """打印图像信息用于调试"""
        try:
            if image is None:
                return "Image is None"
            
            if HAS_TORCH and isinstance(image, torch.Tensor):
                return f"PyTorch Tensor: shape={image.shape}, dtype={image.dtype}, device={image.device}, min={image.min().item() if image.numel() > 0 else 'N/A'}, max={image.max().item() if image.numel() > 0 else 'N/A'}"
            elif HAS_NUMPY and isinstance(image, np.ndarray):
                return f"NumPy array: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}"
            elif HAS_PIL and isinstance(image, Image.Image):
                return f"PIL Image: size={image.size}, mode={image.mode}"
            else:
                return f"Unknown type: {type(image)}"
        except Exception as e:
            return f"Error getting image info: {str(e)}"

    def _encode_image_to_base64(self, image):
        """将图像编码为base64格式"""
        try:
            # 检查依赖
            if not HAS_PIL:
                raise ImportError("缺少必要的依赖: Pillow")
                
            if not HAS_NUMPY and not HAS_TORCH:
                raise ImportError("缺少必要的依赖: numpy 或 torch")
                
            print(f"Processing image: {self._debug_image_info(image)}")
            
            if image is None:
                raise ValueError("Image is None")
            
            # 处理PyTorch张量
            if HAS_TORCH and isinstance(image, torch.Tensor):
                print("Converting PyTorch tensor to NumPy array")
                # 确保张量在CPU上并转换为numpy
                if image.is_cuda:
                    image = image.cpu()
                
                # 转换为numpy数组
                image = image.numpy()
                print(f"Converted to NumPy array: shape={image.shape}, dtype={image.dtype}")
                
            # 处理ComfyUI的图像格式（通常是浮点数numpy数组）
            if HAS_NUMPY and isinstance(image, np.ndarray):
                # 处理批处理维度
                if len(image.shape) == 4:
                    if image.shape[0] == 1:  # 单张图片的批处理
                        image = image[0]
                    else:
                        # 多张图片，只使用第一张
                        print(f"Warning: Received batch of {image.shape[0]} images, using only the first one")
                        image = image[0]
                
                # 确保图像是3通道的
                if len(image.shape) == 3:
                    # 检查通道数
                    if image.shape[2] == 3:  # RGB
                        pass  # 不需要转换
                    elif image.shape[2] == 4:  # RGBA
                        # 只保留RGB通道
                        image = image[:, :, :3]
                    elif image.shape[2] == 1:  # 灰度
                        # 转换为3通道
                        image = np.repeat(image, 3, axis=2)
                    else:
                        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")
                
                # 确保值范围在0-255之间
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # 转换为PIL图像
                pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
            
            elif HAS_PIL and isinstance(image, Image.Image):
                pil_image = image
                # 确保是RGB模式
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # 限制原始图像体积，避免Base64后超过10MB
            max_bytes = 10 * 1024 * 1024
            target_raw_bytes = int(max_bytes * 0.7)  # 约7MB
            min_dim = 512

            def save_to_buffer(img, fmt="JPEG", **save_kwargs):
                buf = io.BytesIO()
                img.save(buf, format=fmt, **save_kwargs)
                return buf, buf.tell()

            buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=95, optimize=True)
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
                buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=90, optimize=True)

            quality = 90
            jpeg_attempts = 0
            while raw_size > target_raw_bytes and quality >= 40:
                buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=quality, optimize=True)
                jpeg_attempts += 1
                print(f"JPEG compression attempt {jpeg_attempts}: quality={quality}, size={raw_size / 1024 / 1024:.2f}MB")
                quality -= 5

            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")

            buffer.seek(0)
            img_bytes = buffer.getvalue()
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_size_mb = len(img_str) / 1024 / 1024
            print(f"Final raw size: {raw_size / 1024 / 1024:.2f}MB, base64 size: {base64_size_mb:.2f}MB")
            print(f"Successfully encoded image to base64 (length: {len(img_str)})")
            return img_str
            
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            print(traceback.format_exc())
            raise

    def _generate_request_id(self, seed=None):
        """生成请求ID，可以基于种子值"""
        timestamp = int(time.time() * 1000)
        
        # 如果提供了种子值，使用它来生成随机字符串
        if seed is not None:
            # 使用种子初始化随机生成器
            local_random = random.Random(seed)
            random_str = ''.join(local_random.choices(string.ascii_letters + string.digits, k=8))
        else:
            # 否则使用普通随机
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            
        return f"{timestamp}-{random_str}"

    def process(self, api_key, model, prompt, max_completion_tokens=65535, temperature=1.0, top_p=0.7, seed=0, reasoning_effort="minimal", image=None, image_2=None):
        """主处理函数"""
        # 应用种子值
        if seed == 0:  # 0表示使用当前种子
            seed = self.current_seed
        
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            return (f"Error: 缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。",)
            
        try:
            print(f"[DoubaoMMM] Processing request with model: {model}")
            print(f"[DoubaoMMM] Image 1 provided: {image is not None}")
            print(f"[DoubaoMMM] Image 2 provided: {image_2 is not None}")
            print(f"[DoubaoMMM] Using seed: {seed}")
            print(f"[DoubaoMMM] Reasoning effort: {reasoning_effort}")
            
            # 获取实际使用的API密钥
            actual_api_key = self._get_api_key(api_key)
            if not actual_api_key:
                return ("Error: 请输入API密钥或在config.json中配置doubao_api_key。请访问 https://console.volcengine.com/ark 获取API密钥。",)
            
            # 使用豆包API，针对深度思考模型设置更长的超时时间
            # 1.8版本支持reasoning_effort，可能需要更长的处理时间
            timeout_value = 1800 if "1-8" in model or "1-6" in model else 60

            print(f"[DoubaoMMM] Calling Doubao API with model: {model}")
            
            # 使用标准的 /api/v3/chat/completions 端点（支持所有参数）
            url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 构建消息内容（支持OpenAI兼容格式）
            user_content = []
            
            # 处理图像输入（支持两张图像）
            if image is not None:
                try:
                    print(f"Processing image 1 for API...")
                    image_base64 = self._encode_image_to_base64(image)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                    print("Successfully added image 1 to message")
                except Exception as e:
                    print(f"Error processing image 1: {str(e)}")
                    print(traceback.format_exc())
                    return (f"Error processing image 1: {str(e)}",)
            
            if image_2 is not None:
                try:
                    print(f"Processing image 2 for API...")
                    image2_base64 = self._encode_image_to_base64(image_2)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image2_base64}"
                        }
                    })
                    print("Successfully added image 2 to message")
                except Exception as e:
                    print(f"Error processing image 2: {str(e)}")
                    print(traceback.format_exc())
                    return (f"Error processing image 2: {str(e)}",)
            
            # 添加文本提示
            user_content.append({"type": "text", "text": prompt})
            
            # 构建消息列表
            messages = [{
                "role": "user",
                "content": user_content
            }]
            
            # 构建请求payload
            payload = {
                "model": model,
                "messages": messages
            }
            
            # 添加max_completion_tokens（支持所有版本）
            if max_completion_tokens and max_completion_tokens > 0:
                payload["max_completion_tokens"] = max_completion_tokens
            
            # 1.8版本添加reasoning_effort参数
            is_1_8_model = "1-8" in model or "251228" in model
            if is_1_8_model:
                if reasoning_effort:
                    payload["reasoning_effort"] = reasoning_effort
                    print(f"[DoubaoMMM] Reasoning effort: {reasoning_effort}")
            else:
                # 对于其他版本，添加 temperature 和 top_p 参数
                payload["temperature"] = temperature
                payload["top_p"] = top_p
            
            print(f"[DoubaoMMM] API端点: {url}")
            print(f"[DoubaoMMM] 请求payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            # 禁用代理，因为豆包是国内服务，通常不需要代理
            # 如果系统设置了代理但代理不可用，会导致连接失败
            resp = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=timeout_value,
                proxies={"http": None, "https": None}  # 禁用代理
            )
            print(f"[DoubaoMMM] Response status code: {resp.status_code}")
            if not resp.ok:
                try:
                    err_json = resp.json()
                    err_message = err_json.get('error', {}).get('message', resp.text)
                except Exception:
                    err_message = resp.text
                # 针对常见错误给出提示
                if resp.status_code == 401:
                    return ("Error: 身份验证失败(401)。请确认 config.json 中的 doubao_api_key 正确且未包含多余空格。若仍失败，请直接用该 key 以 cURL 调用验证。",)
                elif resp.status_code == 404:
                    error_detail = f"模型 '{model}' 不存在或您没有访问权限。\n"
                    error_detail += f"请检查：\n"
                    error_detail += f"1. 模型名称是否正确\n"
                    error_detail += f"2. 您的账户是否有权限访问该模型\n"
                    error_detail += f"3. 模型是否已发布或需要特殊申请\n"
                    error_detail += f"4. 可尝试使用其他可用模型：doubao-seed-1-6-vision-250815 或 doubao-seed-1-6-251015\n"
                    error_detail += f"\n原始错误: {err_message}"
                    return (f"Error: {resp.status_code} - {error_detail}",)
                return (f"Error: {resp.status_code} - {err_message}",)
            
            result = resp.json()
            print(f"[DoubaoMMM] Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 解析响应（/api/v3/chat/completions 标准格式）
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
            else:
                # 如果无法解析，返回完整响应供调试
                response_text = json.dumps(result, ensure_ascii=False, indent=2)
                print(f"[DoubaoMMM] Warning: 无法解析响应格式，返回完整JSON供调试")

            # 更新种子
            self.current_seed = seed

            return (response_text,)
            
        except Exception as e:
            print(f"Unexpected error in process: {str(e)}")
            print(traceback.format_exc())
            return (f"Error: {str(e)}",)







NODE_CLASS_MAPPINGS = {
    "DoubaoNode": DoubaoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoNode": "🥟豆包MMM"
}
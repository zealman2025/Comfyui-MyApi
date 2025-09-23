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
        print(f"[DoubaoNode] Loading config from: {config_path}")
        print(f"[DoubaoNode] Config file exists: {os.path.exists(config_path)}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            doubao_models = models.get('doubao', {})
            print(f"[DoubaoNode] Loaded Doubao models: {doubao_models}")
            print(f"[DoubaoNode] Doubao model keys: {list(doubao_models.keys())}")
            return doubao_models
    except Exception as e:
        print(f"[DoubaoNode] Error loading models from config: {str(e)}")
        import traceback
        traceback.print_exc()
        # 提供默认模型作为回退
        default_models = {
            "doubao-1-5-thinking-vision-pro-250428": "Doubao-1.5-thinking-vision-pro",
            "doubao-seed-1-6-250615": "豆包Seed1.6版"
        }
        print(f"[DoubaoNode] Using default models: {default_models}")
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
            print(f"[DoubaoNode] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('doubao_api_key', '').strip()
                if config_api_key:
                    print(f"[DoubaoNode] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[DoubaoNode] config.json中未找到doubao_api_key")
                    return ''
        except Exception as e:
            print(f"[DoubaoNode] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(DOUBAO_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image content in detail, without making comments or suggestions"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "thinking_mode": (["自动", "启用", "禁用"], {"default": "禁用"}),
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
            
            # 将PIL图像转换为JPEG格式的base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
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

    def process(self, api_key, model, prompt, max_tokens=4096, temperature=1.0, top_p=0.7, seed=0, thinking_mode="自动", image=None, image_2=None):
        """主处理函数"""
        # 中文思考模式映射为英文API值
        thinking_mode_map = {
            "自动": "auto",
            "启用": "enabled", 
            "禁用": "disabled"
        }
        api_thinking_mode = thinking_mode_map.get(thinking_mode, "auto")
        
        # 应用种子值
        if seed == 0:  # 0表示使用当前种子
            seed = self.current_seed
        
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            return (f"Error: 缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。",)
            
        try:
            print(f"Processing request with model: {model}")
            print(f"Image 1 provided: {image is not None}")
            print(f"Image 2 provided: {image_2 is not None}")
            print(f"Using seed: {seed}")
            
            # 获取实际使用的API密钥
            actual_api_key = self._get_api_key(api_key)
            if not actual_api_key:
                return ("Error: 请输入API密钥或在config.json中配置doubao_api_key。请访问 https://console.volcengine.com/ark 获取API密钥。",)
            
            # 使用豆包API，针对深度思考模型设置更长的超时时间
            timeout_value = 1800 if model == "doubao-seed-1-6-250615" else 60

            # 创建用户消息内容，按照官方示例格式
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

            # 按照官方示例格式构建消息
            messages = [{
                "role": "user",
                "content": user_content
            }]

            print(f"Calling Doubao API with model: {model}")
            
            # 直接使用 HTTP 请求，避免 SDK 兼容性导致的 401
            url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            # 为 doubao-seed-1-6-250615 模型添加深度思考控制（按照官方 extra_body.thinking）
            if model == "doubao-seed-1-6-250615":
                payload["extra_body"] = {"thinking": {"type": api_thinking_mode}}
                print(f"深度思考模式: {thinking_mode} (API值: {api_thinking_mode})")
            
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_value)
            print(f"Response status code: {resp.status_code}")
            if not resp.ok:
                try:
                    err_json = resp.json()
                    err_message = err_json.get('error', {}).get('message', resp.text)
                except Exception:
                    err_message = resp.text
                # 针对常见错误给出提示
                if resp.status_code == 401:
                    return ("Error: 身份验证失败(401)。请确认 config.json 中的 doubao_api_key 正确且未包含多余空格。若仍失败，请直接用该 key 以 cURL 调用验证。",)
                return (f"Error: {resp.status_code} - {err_message}",)
            
            result = resp.json()
            response_text = result["choices"][0]["message"]["content"]

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
    "DoubaoNode": "🥟豆包 AI"
}
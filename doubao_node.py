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
            doubao_image_edit_models = models.get('doubao_image_edit', {})
            doubao_text_to_image_models = models.get('doubao_text_to_image', {})
            # 移除默认模型回退：仅使用配置文件中的内容
            return doubao_models, doubao_image_edit_models, doubao_text_to_image_models
    except Exception as e:
        print(f"Error loading models from config: {str(e)}")
        # 不再提供默认模型，返回空集合
        return {}, {}, {}

# 加载模型配置
DOUBAO_MODELS, DOUBAO_IMAGE_EDIT_MODELS, DOUBAO_TEXT_TO_IMAGE_MODELS = load_models_from_config()

class DoubaoNode:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.api_key = self._load_api_key()
        self.current_seed = 0  # 初始化种子值
        
    def _load_api_key(self):
        try:
            key = ""
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    key = (config.get('doubao_api_key', '') or '').strip()
            if not key:
                key = (os.environ.get('ARK_API_KEY') or os.environ.get('DOUBAO_API_KEY') or '').strip()
            if not key:
                print(f"[DoubaoNode] 未找到 API Key。config 路径: {self.config_path}, exists={os.path.exists(self.config_path)}；环境变量 ARK_API_KEY/DOUABAO_API_KEY 未设置")
            else:
                print(f"[DoubaoNode] 已加载 API Key，长度={len(key)}")
            return key
        except Exception as e:
            print(f"Error loading Doubao API key: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(DOUBAO_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image content in detail, without making comments or suggestions"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "thinking_mode": (["自动", "启用", "禁用"], {"default": "禁用"}),
            },
            "optional": {
                "image": ("IMAGE",),
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

    def process(self, model, prompt, max_tokens=1024, temperature=1.0, top_p=0.7, seed=0, thinking_mode="自动", image=None):
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
            print(f"Image provided: {image is not None}")
            print(f"Using seed: {seed}")
            
            # 校验 API Key 是否存在
            if not self.api_key:
                return ("Error: 请先在 config.json 中配置 doubao_api_key",)
            
            # 使用豆包API，针对深度思考模型设置更长的超时时间
            timeout_value = 1800 if model == "doubao-seed-1-6-250615" else 60

            # 创建用户消息内容，按照官方示例格式
            user_content = []
            
            # 处理图像输入
            if image is not None:
                try:
                    print(f"Processing image for API...")
                    image_base64 = self._encode_image_to_base64(image)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                    print("Successfully added image to message")
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    print(traceback.format_exc())
                    return (f"Error processing image: {str(e)}",)

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
                "Authorization": f"Bearer {self.api_key}"
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

class DoubaoImageEditNode:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.api_key = self._load_api_key()
        self.current_seed = 21  # 初始化种子值

    def _load_api_key(self):
        try:
            key = ""
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    key = (config.get('doubao_api_key', '') or '').strip()
            if not key:
                key = (os.environ.get('ARK_API_KEY') or os.environ.get('DOUBAO_API_KEY') or '').strip()
            if not key:
                print(f"[DoubaoImageEditNode] 未找到 API Key。config 路径: {self.config_path}, exists={os.path.exists(self.config_path)}；环境变量 ARK_API_KEY/DOUABAO_API_KEY 未设置")
            else:
                print(f"[DoubaoImageEditNode] 已加载 API Key，长度={len(key)}")
            return key
        except Exception as e:
            print(f"Error loading Doubao API key: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(DOUBAO_IMAGE_EDIT_MODELS.keys()),),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "改成爱心形状的泡泡"}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 21, "min": 1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
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

    def _encode_image_to_base64(self, image):
        """将图像编码为base64格式"""
        try:
            # 检查依赖
            if not HAS_PIL:
                raise ImportError("缺少必要的依赖: Pillow")
                
            if not HAS_NUMPY and not HAS_TORCH:
                raise ImportError("缺少必要的依赖: numpy 或 torch")
                
            print(f"Processing image for edit API")
            
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

    def _decode_image_from_url(self, image_url):
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            if not HAS_REQUESTS:
                raise ImportError("缺少必要的依赖: requests")
                
            if not HAS_PIL:
                raise ImportError("缺少必要的依赖: Pillow")
                
            if not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: numpy")
            
            print(f"Downloading image from URL: {image_url}")
            
            # 下载图像
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 从字节流创建PIL图像
            image = Image.open(io.BytesIO(response.content))
            
            # 确保是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # 添加批处理维度 [1, H, W, C]
            image_tensor = image_array[np.newaxis, ...]
            
            print(f"Successfully converted image to tensor: shape={image_tensor.shape}")
            
            # 如果有torch，转换为torch张量
            if HAS_TORCH:
                image_tensor = torch.from_numpy(image_tensor)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error downloading/converting image: {str(e)}")
            print(traceback.format_exc())
            raise

    def edit_image(self, model, image, prompt, guidance_scale=5.5, seed=21):
        """主处理函数"""
        # 应用种子值
        if seed <= 0:  # 小于等于0表示使用当前种子
            seed = self.current_seed if self.current_seed > 0 else 21
        
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
            
        try:
            print(f"Processing image edit request")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"Size: adaptive")  # 固定为adaptive
            print(f"Seed: {seed}")
            print(f"Guidance scale: {guidance_scale}")
            
            if not self.api_key:
                raise Exception("请先在config.json文件中配置doubao_api_key")
            
            # 编码输入图像
            image_base64 = self._encode_image_to_base64(image)
            
            # 构建API请求 - 根据官方文档格式
            url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 使用用户选择的模型和固定的adaptive尺寸
            payload = {
                "model": model,  # 使用用户选择的模型
                "prompt": prompt,
                "image": f"data:image/jpeg;base64,{image_base64}",
                "response_format": "url", 
                "size": "adaptive",  # 固定为adaptive
                "seed": seed,
                "guidance_scale": guidance_scale,
                "watermark": False  # 默认关闭水印
            }
            
            print("Calling Doubao image edit API...")
            print(f"Payload keys: {list(payload.keys())}")
            
            # 发起API请求
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # 详细记录响应信息
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if not response.ok:
                error_text = response.text
                print(f"API error response: {error_text}")
                try:
                    error_json = response.json()
                    error_message = error_json.get('error', {}).get('message', error_text)
                except:
                    error_message = error_text
                
                # 针对敏感内容提供友好的错误提示
                if "sensitive information" in error_message.lower() or "敏感" in error_message:
                    user_friendly_message = "提示词可能包含敏感内容，请修改后重试。建议使用更加温和、积极的描述词汇。"
                elif "违禁" in error_message or "forbidden" in error_message.lower():
                    user_friendly_message = "提示词包含违禁内容，请使用符合规范的描述词汇。"
                elif response.status_code == 400:
                    user_friendly_message = f"请求参数有误，请检查输入内容。详细信息：{error_message}"
                elif response.status_code == 401:
                    user_friendly_message = "API密钥无效或已过期，请检查config.json中的doubao_api_key配置。"
                elif response.status_code == 429:
                    user_friendly_message = "请求过于频繁，请稍后再试。"
                elif response.status_code == 500:
                    user_friendly_message = "服务器内部错误，请稍后重试。"
                else:
                    user_friendly_message = f"API调用失败 (状态码: {response.status_code}): {error_message}"
                
                raise Exception(user_friendly_message)
            
            result = response.json()
            print(f"API response received: {result}")
            
            # 解析响应
            if 'data' not in result or len(result['data']) == 0:
                raise Exception("API响应中没有图像数据")
            
            # 获取生成的图像URL
            image_url = result['data'][0]['url']
            print(f"Generated image URL: {image_url}")
            
            # 下载并转换图像
            output_image = self._decode_image_from_url(image_url)
            
            # 更新种子（ComfyUI会自动处理control_after_generate）
            self.current_seed = seed
            
            return (output_image,)
            
        except Exception as e:
            error_str = str(e)
            print(f"Error in edit_image: {error_str}")
            
            # 针对敏感内容等错误，提供友好的错误信息
            if any(keyword in error_str for keyword in ["提示词可能包含敏感内容", "提示词包含违禁内容", "请求参数有误", "API密钥无效", "请求过于频繁", "服务器内部错误"]):
                print(f"用户友好提示: {error_str}")
                raise Exception(error_str)
            
            # 其他错误打印详细信息
            print(traceback.format_exc())
            raise Exception(f"图像编辑失败: {error_str}")

class DoubaoTextToImageNode:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.api_key = self._load_api_key()
        self.current_seed = 21  # 初始化种子值

    def _load_api_key(self):
        try:
            key = ""
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    key = (config.get('doubao_api_key', '') or '').strip()
            if not key:
                key = (os.environ.get('ARK_API_KEY') or os.environ.get('DOUBAO_API_KEY') or '').strip()
            if not key:
                print(f"[DoubaoTextToImageNode] 未找到 API Key。config 路径: {self.config_path}, exists={os.path.exists(self.config_path)}；环境变量 ARK_API_KEY/DOUABAO_API_KEY 未设置")
            else:
                print(f"[DoubaoTextToImageNode] 已加载 API Key，长度={len(key)}")
            return key
        except Exception as e:
            print(f"Error loading Doubao API key: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(DOUBAO_TEXT_TO_IMAGE_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "鱼眼镜头，一只猫咪的头部，画面呈现出猫咪的五官因为拍摄方式扭曲的效果。"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 21, "min": 1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
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



    def _decode_image_from_url(self, image_url):
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            if not HAS_REQUESTS:
                raise ImportError("缺少必要的依赖: requests")
                
            if not HAS_PIL:
                raise ImportError("缺少必要的依赖: Pillow")
                
            if not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: numpy")
            
            print(f"Downloading image from URL: {image_url}")
            
            # 下载图像
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 从字节流创建PIL图像
            image = Image.open(io.BytesIO(response.content))
            
            # 确保是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # 添加批处理维度 [1, H, W, C]
            image_tensor = image_array[np.newaxis, ...]
            
            print(f"Successfully converted image to tensor: shape={image_tensor.shape}")
            
            # 如果有torch，转换为torch张量
            if HAS_TORCH:
                image_tensor = torch.from_numpy(image_tensor)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error downloading/converting image: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate_image(self, model, prompt, width=768, height=1024, guidance_scale=3.0, seed=21):
        """主处理函数"""
        # 应用种子值
        if seed <= 0:  # 小于等于0表示使用当前种子
            seed = self.current_seed if self.current_seed > 0 else 21
        
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
            
        try:
            # 构建尺寸字符串
            size_str = f"{width}x{height}"
            
            print(f"Processing text to image request")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"Size: {size_str} (width={width}, height={height})")
            print(f"Seed: {seed}")
            print(f"Guidance scale: {guidance_scale}")
            
            if not self.api_key:
                raise Exception("请先在config.json文件中配置doubao_api_key")
            
            # 构建API请求
            url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建请求参数
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": "url",
                "size": size_str,
                "guidance_scale": guidance_scale,
                "watermark": False  # 默认关闭水印
            }
            
            # 只有当模型支持seed时才添加seed参数
            if "seedream" in model:
                payload["seed"] = seed
            
            print("Calling Doubao text-to-image API...")
            print(f"Payload keys: {list(payload.keys())}")
            
            # 发起API请求
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # 详细记录响应信息
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if not response.ok:
                error_text = response.text
                print(f"API error response: {error_text}")
                try:
                    error_json = response.json()
                    error_message = error_json.get('error', {}).get('message', error_text)
                except:
                    error_message = error_text
                
                # 针对敏感内容提供友好的错误提示
                if "sensitive information" in error_message.lower() or "敏感" in error_message:
                    user_friendly_message = "提示词可能包含敏感内容，请修改后重试。建议使用更加温和、积极的描述词汇。"
                elif "违禁" in error_message or "forbidden" in error_message.lower():
                    user_friendly_message = "提示词包含违禁内容，请使用符合规范的描述词汇。"
                elif response.status_code == 400:
                    user_friendly_message = f"请求参数有误，请检查输入内容。详细信息：{error_message}"
                elif response.status_code == 401:
                    user_friendly_message = "API密钥无效或已过期，请检查config.json中的doubao_api_key配置。"
                elif response.status_code == 429:
                    user_friendly_message = "请求过于频繁，请稍后再试。"
                elif response.status_code == 500:
                    user_friendly_message = "服务器内部错误，请稍后重试。"
                else:
                    user_friendly_message = f"API调用失败 (状态码: {response.status_code}): {error_message}"
                
                raise Exception(user_friendly_message)
            
            result = response.json()
            print(f"API response received: {result}")
            
            # 解析响应
            if 'data' not in result or len(result['data']) == 0:
                raise Exception("API响应中没有图像数据")
            
            # 获取生成的图像URL
            image_url = result['data'][0]['url']
            print(f"Generated image URL: {image_url}")
            
            # 下载并转换图像
            output_image = self._decode_image_from_url(image_url)
            
            # 更新种子（ComfyUI会自动处理control_after_generate）
            self.current_seed = seed
            
            return (output_image,)
            
        except Exception as e:
            error_str = str(e)
            print(f"Error in generate_image: {error_str}")
            
            # 针对敏感内容等错误，提供友好的错误信息
            if any(keyword in error_str for keyword in ["提示词可能包含敏感内容", "提示词包含违禁内容", "请求参数有误", "API密钥无效", "请求过于频繁", "服务器内部错误"]):
                print(f"用户友好提示: {error_str}")
                raise Exception(error_str)
            
            # 其他错误打印详细信息
            print(traceback.format_exc())
            raise Exception(f"图像生成失败: {error_str}")

NODE_CLASS_MAPPINGS = {
    "DoubaoNode": DoubaoNode,
    "DoubaoImageEditNode": DoubaoImageEditNode,
    "DoubaoTextToImageNode": DoubaoTextToImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoNode": "🥟豆包 AI",
    "DoubaoImageEditNode": "🎨豆包图像编辑",
    "DoubaoTextToImageNode": "🖼️豆包文生图"
} 
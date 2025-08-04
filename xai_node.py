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

# 从配置文件加载模型配置
def load_xai_models_from_config():
    """从config.json加载XAI模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            xai_models = models.get('xai', {})
            
            # 如果配置文件中没有模型配置，使用默认值
            if not xai_models:
                xai_models = {
                    "grok-2-vision-1212": "Grok 2 Vision 1212",
                }
                
            return xai_models
    except Exception as e:
        print(f"Error loading XAI models from config: {str(e)}")
        # 返回默认模型配置
        return {
            "grok-2-vision-1212": "Grok 2 Vision 1212",
        }

# 加载模型配置
XAI_MODELS = load_xai_models_from_config()

class XAINode:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.api_key = self._load_api_key()
        
    def _load_api_key(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('xai_api_key', '')
        except Exception as e:
            print(f"Error loading XAI API key: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(XAI_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image content in detail, without making comments or suggestions"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
            
        if not HAS_OPENAI:
            missing_deps.append("openai")
            
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

    def process(self, model, prompt, max_tokens=1024, temperature=1.0, top_p=0.7, seed=0, image=None):
        """主处理函数"""
        
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            return (f"Error: 缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。",)

        try:
            print(f"Processing request with XAI model: {model}")
            print(f"Image provided: {image is not None}")
            print(f"Using seed: {seed}")
            
            if not HAS_OPENAI:
                return ("Error: openai 软件包未安装。请使用以下命令安装它 'pip install openai'",)
                
            # 使用OpenAI兼容API调用XAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )

            # 使用固定的system message
            system_prompt = "You are a helpful assistant that accurately describes images and answers questions."
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            ]

            # 创建用户消息
            user_content = []
            
            # 处理图像输入
            if image is not None:
                try:
                    print(f"Processing image for XAI API...")
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
            # 根据种子生成请求ID，确保相同种子产生相同结果
            request_id = self._generate_request_id(seed)
            actual_prompt = f"{prompt}\n\n[Request ID: {request_id}]"
            user_content.append({"type": "text", "text": actual_prompt})

            messages.append({
                "role": "user",
                "content": user_content
            })

            print(f"Calling XAI API with model: {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            response_text = completion.choices[0].message.content

            return (response_text,)
            
        except Exception as e:
            print(f"Unexpected error in XAI process: {str(e)}")
            print(traceback.format_exc())
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "XAINode": XAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XAINode": "🚀XAI Grok"
}

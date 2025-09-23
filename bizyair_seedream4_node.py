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

class BizyAirSeedream4Node:
    """
    BizyAir Seedream4专用节点
    专门用于调用BizyAir的Seedream4模型API
    支持图像输入、提示词、尺寸选择和自定义宽高
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
            print(f"[BizyAirSeedream4] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('bizyair_api_key', '').strip()
                if config_api_key:
                    print(f"[BizyAirSeedream4] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[BizyAirSeedream4] config.json中未找到bizyair_api_key")
                    return ''
        except Exception as e:
            print(f"[BizyAirSeedream4] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "将兔子改为小猫"}),
                "size": ([
                    "1K Square (1024x1024)",
                    "2K Square (2048x2048)", 
                    "4K Square (4096x4096)",
                    "HD 16:9 (1920x1080)",
                    "2K 16:9 (2560x1440)",
                    "4K 16:9 (3840x2160)",
                    "Portrait 9:16 (1080x1920)",
                    "Portrait 3:4 (1536x2048)",
                    "Landscape 4:3 (2048x1536)",
                    "Ultra-wide 21:9 (3440x1440)",
                    "Custom"
                ], {"default": "1K Square (1024x1024)"}),
                "custom_width": ("INT", {"default": 1920, "min": 1024, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 1080, "min": 1024, "max": 8192, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
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

    def _parse_size_option(self, size_option, custom_width, custom_height):
        """解析size选项，返回实际的width和height，确保最小值为1024"""
        if size_option == "Custom":
            # 确保自定义尺寸不低于1024
            width = max(custom_width, 1024)
            height = max(custom_height, 1024)
            if width != custom_width or height != custom_height:
                print(f"Warning: Custom dimensions adjusted to minimum 1024. Original: {custom_width}x{custom_height}, Adjusted: {width}x{height}")
            return width, height

        # 从size选项中提取尺寸信息
        size_mappings = {
            "1K Square (1024x1024)": (1024, 1024),
            "2K Square (2048x2048)": (2048, 2048),
            "4K Square (4096x4096)": (4096, 4096),
            "HD 16:9 (1920x1080)": (1920, 1080),
            "2K 16:9 (2560x1440)": (2560, 1440),
            "4K 16:9 (3840x2160)": (3840, 2160),
            "Portrait 9:16 (1080x1920)": (1080, 1920),
            "Portrait 3:4 (1536x2048)": (1536, 2048),
            "Landscape 4:3 (2048x1536)": (2048, 1536),
            "Ultra-wide 21:9 (3440x1440)": (3440, 1440),
        }

        if size_option in size_mappings:
            width, height = size_mappings[size_option]
            # 确保预设尺寸也不低于1024（虽然预设都已经>=1024）
            return max(width, 1024), max(height, 1024)

        # 如果没有找到匹配的预设，尝试从字符串中解析
        import re
        match = re.search(r'\((\d+)x(\d+)\)', size_option)
        if match:
            width = max(int(match.group(1)), 1024)
            height = max(int(match.group(2)), 1024)
            return width, height

        # 默认返回自定义尺寸（确保最小值）
        return max(custom_width, 1024), max(custom_height, 1024)

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
            
            # 转换为base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_base64}"
            
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

    def generate(self, api_key, prompt, size, custom_width, custom_height, seed, image=None):
        """生成图像"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            raise Exception("请输入API密钥或在config.json中配置bizyair_api_key。请访问 https://bizyair.cn 获取API密钥。")

        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
        
        # 处理size选项，获取实际的width和height
        actual_width, actual_height = self._parse_size_option(size, custom_width, custom_height)
        print(f"Using size: {size}, actual dimensions: {actual_width}x{actual_height}")
        
        # 生成随机种子（如果需要）
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir Seedream4 API request to: {api_url}")
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 构建input_values
            input_values = {
                "17:BizyAir_Seedream4.prompt": prompt,
                "17:BizyAir_Seedream4.size": size,
                "17:BizyAir_Seedream4.custom_width": str(actual_width),
                "17:BizyAir_Seedream4.custom_height": str(actual_height)
            }
            
            # 如果有图像输入，添加图像
            if image is not None:
                image_base64 = self._image_to_base64(image)
                if image_base64:
                    input_values["18:LoadImage.image"] = image_base64
                    print("Added input image to request")
                else:
                    print("Warning: Failed to convert input image to base64")
            
            # 构建请求数据
            data = {
                "web_app_id": 36598,  # Seedream4的固定web_app_id
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}, input_values count={len(input_values)}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Size: {size} ({actual_width}x{actual_height})")
            
            # 发送请求（增加超时时间到120秒）
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            print("API response received")
            
            # 检查响应状态
            if result.get("status") != "Success":
                raise Exception(f"API请求失败: {result.get('status', 'Unknown error')}")
            
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
                "web_app_id": 36598,
                "prompt": prompt,
                "size": size,
                "dimensions": f"{actual_width}x{actual_height}",
                "seed": seed,
                "image_url": image_url,
                "cost_time": result.get("cost_times", {}).get("total_cost_time", 0),
                "request_id": result.get("request_id", "")
            }
            
            status_text = f"✅ Seedream4生成成功\n"
            status_text += f"提示词: {prompt[:50]}...\n"
            status_text += f"尺寸: {size} ({actual_width}x{actual_height})\n"
            status_text += f"种子: {seed}\n"
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
    "BizyAirSeedream4Node": BizyAirSeedream4Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirSeedream4Node": "🌐BizyAir Seedream4 (需BizyAir.cn充值金币)"
}

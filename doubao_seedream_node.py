import os
import json
import io
import base64
import traceback
import time
import random

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
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# 从配置文件加载模型配置
def load_doubao_seedream_models_from_config():
    """从config.json加载豆包SEEDREAM模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            seedream_models = models.get('doubao_seedream', {})
            if seedream_models:
                return seedream_models
            # 如果没有配置，返回默认模型
            return {
                "doubao-seedream-4-0-250828": "豆包SEEDREAM 4.0"
            }
    except Exception as e:
        print(f"[DoubaoSeedreamNode] 加载配置失败: {str(e)}")
        traceback.print_exc()
        return {
            "doubao-seedream-4-0-250828": "豆包SEEDREAM 4.0"
        }

# 加载模型配置
DOUBAO_SEEDREAM_MODELS = load_doubao_seedream_models_from_config()

class DoubaoSeedreamNode:
    """豆包 SEEDREAM 4.0 图像生成节点"""
    
    def __init__(self):
        self.current_seed = 21
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
            print(f"[DoubaoSeedream] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('doubao_api_key', '').strip()
                if config_api_key:
                    print(f"[DoubaoSeedream] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[DoubaoSeedream] config.json中未找到doubao_api_key")
                    return ''
        except Exception as e:
            print(f"[DoubaoSeedream] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        # 预设尺寸选项 - 根据参考图更新
        size_options = [
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
        ]

        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫咪，坐在花园里，阳光明媚"}),
                "size": (size_options, {"default": "Custom"}),
                "custom_width": ("INT", {"default": 1920, "min": 1024, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1080, "min": 1024, "max": 4096, "step": 16}),
                "model": (list(DOUBAO_SEEDREAM_MODELS.keys()), {"default": list(DOUBAO_SEEDREAM_MODELS.keys())[0] if DOUBAO_SEEDREAM_MODELS else "doubao-seedream-4-0-250828"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "generate"
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

    def _parse_size_option(self, size, custom_width, custom_height):
        """解析尺寸选项"""
        if size == "Custom":
            # 确保自定义尺寸不小于1024
            width = max(custom_width, 1024)
            height = max(custom_height, 1024)
            return width, height

        # 解析新的尺寸格式 (格式: "名称 (宽度x高度)")
        size_mapping = {
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

        if size in size_mapping:
            return size_mapping[size]

        # 兜底：尝试从括号中提取尺寸
        try:
            if '(' in size and ')' in size:
                size_part = size.split('(')[1].split(')')[0]
                width_str, height_str = size_part.split('x')
                return int(width_str), int(height_str)
        except:
            pass

        # 默认尺寸
        return 1920, 1080

    def _encode_image_to_base64(self, image):
        """将图像编码为base64格式"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: Pillow 或 numpy")
                
            print(f"Processing image for SEEDREAM API")
            
            if image is None:
                raise ValueError("Image is None")
            
            # 处理PyTorch张量
            if HAS_TORCH and isinstance(image, torch.Tensor):
                print("Converting PyTorch tensor to NumPy array")
                if image.is_cuda:
                    image = image.cpu()
                image = image.numpy()
                print(f"Converted to NumPy array: shape={image.shape}, dtype={image.dtype}")
                
            # 处理ComfyUI的图像格式
            if HAS_NUMPY and isinstance(image, np.ndarray):
                # 处理批处理维度
                if len(image.shape) == 4:
                    if image.shape[0] == 1:
                        image = image[0]
                    else:
                        print(f"Warning: Received batch of {image.shape[0]} images, using only the first one")
                        image = image[0]
                
                # 确保图像是3通道的
                if len(image.shape) == 3:
                    if image.shape[2] == 3:  # RGB
                        pass
                    elif image.shape[2] == 4:  # RGBA
                        image = image[:, :, :3]
                    elif image.shape[2] == 1:  # 灰度
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
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # 豆包接口限制原始图像最大10MB。考虑到Base64会膨胀约33%，我们将原始数据控制在7MB左右。
            max_bytes = 10 * 1024 * 1024
            target_raw_bytes = int(max_bytes * 0.7)  # 约7MB
            min_dim = 512

            def save_to_buffer(img, fmt, **save_kwargs):
                buf = io.BytesIO()
                img.save(buf, format=fmt, **save_kwargs)
                return buf, buf.tell()

            buffer, raw_size = save_to_buffer(pil_image, "PNG", optimize=True)
            image_format = "PNG"

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
                buffer, raw_size = save_to_buffer(pil_image, "PNG", optimize=True)
                image_format = "PNG"

            if raw_size > target_raw_bytes:
                print("PNG still too large, switching to JPEG compression...")
                quality = 90
                jpeg_attempts = 0
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_to_buffer(pil_image, "JPEG", quality=quality, optimize=True)
                    image_format = "JPEG"
                    jpeg_attempts += 1
                    print(f"JPEG compression attempt {jpeg_attempts}: quality={quality}, size={raw_size / 1024 / 1024:.2f}MB")
                    quality -= 5

            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")

            buffer.seek(0)
            img_b64_bytes = base64.b64encode(buffer.getvalue())
            img_b64_len = len(img_b64_bytes)
            print(f"Final raw size: {raw_size / 1024 / 1024:.2f}MB, base64 size: {img_b64_len / 1024 / 1024:.2f}MB, format: {image_format}")
            img_str = img_b64_bytes.decode('utf-8')

            mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
            print(f"Successfully encoded image to base64 (length: {len(img_str)})")
            return f"data:{mime_type};base64,{img_str}"
            
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            print(traceback.format_exc())
            raise

    def _decode_image_from_url(self, image_url):
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            if not HAS_REQUESTS or not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: requests, Pillow 或 numpy")
            
            print(f"Downloading image from URL: {image_url}")
            
            # 下载图像
            response = requests.get(image_url, timeout=60)
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

    def _process_stream_response(self, response):
        """处理流式响应"""
        try:
            images = []
            line_count = 0

            print("开始处理流式响应...")

            # 读取流式响应
            for line in response.iter_lines(decode_unicode=True):
                line_count += 1

                if line.strip() == '':
                    continue

                print(f"收到响应行 {line_count}: {line[:100]}...")  # 只显示前100个字符

                # 处理SSE格式数据
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀

                    if data == '[DONE]':
                        print('流式响应完成')
                        break

                    try:
                        parsed = json.loads(data)
                        print(f"解析的JSON数据: {parsed}")

                        # 检查错误
                        if 'error' in parsed:
                            raise Exception(f"API错误: {parsed['error'].get('message', '未知错误')}")

                        # 检查响应类型和提取图片URL
                        response_type = parsed.get('type', '')

                        if response_type == 'image_generation.partial_succeeded':
                            # 部分成功，包含图像URL
                            if 'url' in parsed:
                                images.append(parsed['url'])
                                print(f"收到图片URL (partial_succeeded): {parsed['url']}")
                            else:
                                print("partial_succeeded响应中没有找到url字段")

                        elif response_type == 'image_generation.completed':
                            # 生成完成，通常只包含统计信息
                            print(f"图像生成完成，统计信息: {parsed.get('usage', {})}")

                        elif 'data' in parsed and len(parsed['data']) > 0:
                            # 兼容旧格式：data数组格式
                            print(f"找到data字段，包含 {len(parsed['data'])} 个项目")
                            for i, item in enumerate(parsed['data']):
                                print(f"处理项目 {i}: {item}")
                                if 'url' in item:
                                    images.append(item['url'])
                                    print(f"收到图片URL (data格式): {item['url']}")
                                else:
                                    print(f"项目 {i} 中没有找到url字段")
                        else:
                            print(f"未识别的响应类型或格式: {response_type}")

                    except json.JSONDecodeError as e:
                        print(f"解析响应数据失败: {e}, 原始数据: {data}")
                        continue
                else:
                    print(f"非data行: {line}")

            print(f"流式响应处理完成，共收到 {len(images)} 张图片")
            return images

        except Exception as e:
            print(f"处理流式响应失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def generate(self, api_key, prompt, size, custom_width, custom_height, model, seed, watermark, stream,
                 image=None, image2=None, image3=None, image4=None, image5=None,
                 image6=None, image7=None, image8=None, image9=None, image10=None):
        """主生成函数"""
        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
        
        try:
            # 处理种子
            if seed == 0:
                seed = random.randint(1, 0xffffffffffffffff)
            
            # 解析尺寸
            width, height = self._parse_size_option(size, custom_width, custom_height)
            
            # 收集所有输入的图像
            input_images = []
            for img in [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
                if img is not None:
                    input_images.append(img)

            print(f"豆包 SEEDREAM 4.0 图像生成")
            print(f"提示词: {prompt}")
            print(f"尺寸: {width}x{height}")
            print(f"种子: {seed}")
            print(f"水印: {watermark}")
            print(f"流式: {stream}")
            print(f"模型: {model}")
            print(f"模式: {'图生图' if len(input_images) > 0 else '文生图'}")
            if len(input_images) > 0:
                print(f"参考图数量: {len(input_images)}")

            # 获取实际使用的API密钥
            actual_api_key = self._get_api_key(api_key)
            if not actual_api_key:
                raise Exception("请输入API密钥或在config.json中配置doubao_api_key。请访问 https://console.volcengine.com/ark 获取API密钥。")

            # 构建API请求
            url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {actual_api_key}',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            }

            # 构建请求体
            request_body = {
                "model": model,
                "prompt": prompt,
                "response_format": "url",
                "size": f"{width}x{height}",
                "stream": stream,
                "watermark": watermark
            }

            # 如果有输入图像，添加图像数据（图生图模式）
            if len(input_images) > 0:
                # 根据官方API文档，image字段应该是数组格式
                image_urls = []
                for i, img in enumerate(input_images[:5]):  # 最多支持5张图像
                    try:
                        image_base64 = self._encode_image_to_base64(img)
                        image_urls.append(image_base64)
                        print(f"已编码参考图像 {i+1}")
                    except Exception as e:
                        print(f"编码参考图像 {i+1} 失败: {str(e)}")
                        continue

                if image_urls:
                    request_body["image"] = image_urls
                    print(f"已添加 {len(image_urls)} 张参考图像到请求中")
            
            print("调用豆包 SEEDREAM 4.0 API...")
            print(f"请求参数: {list(request_body.keys())}")
            
            # 发送请求
            response = requests.post(url, headers=headers, json=request_body, timeout=120, stream=stream)
            
            print(f"响应状态码: {response.status_code}")
            
            if not response.ok:
                error_text = response.text
                print(f"API错误响应: {error_text}")
                try:
                    error_json = response.json()
                    error_message = error_json.get('error', {}).get('message', error_text)
                except:
                    error_message = error_text
                
                # 友好的错误提示
                if "sensitive information" in error_message.lower() or "敏感" in error_message:
                    user_friendly_message = "提示词可能包含敏感内容，请修改后重试。"
                elif "违禁" in error_message or "forbidden" in error_message.lower():
                    user_friendly_message = "提示词包含违禁内容，请使用符合规范的描述词汇。"
                elif response.status_code == 400:
                    user_friendly_message = f"请求参数有误: {error_message}"
                elif response.status_code == 401:
                    user_friendly_message = "API密钥无效，请检查config.json中的doubao_api_key配置。"
                elif response.status_code == 429:
                    user_friendly_message = "请求过于频繁，请稍后再试。"
                elif response.status_code == 500:
                    user_friendly_message = "服务器内部错误，请稍后重试。"
                else:
                    user_friendly_message = f"API调用失败 (状态码: {response.status_code}): {error_message}"
                
                raise Exception(user_friendly_message)
            
            # 处理响应
            if stream:
                # 处理流式响应
                images = self._process_stream_response(response)
            else:
                # 处理普通响应
                result = response.json()
                print(f"API响应: {result}")
                
                if 'data' not in result or len(result['data']) == 0:
                    raise Exception("API响应中没有图像数据")
                
                images = [item['url'] for item in result['data'] if 'url' in item]
            
            if not images:
                error_msg = "没有生成任何图像。可能的原因：\n"
                error_msg += "1. 提示词包含敏感内容被过滤\n"
                error_msg += "2. 参考图像格式不支持\n"
                error_msg += "3. API服务暂时不可用\n"
                error_msg += "4. 账户余额不足\n"
                error_msg += "建议：检查提示词内容，确认账户状态，或稍后重试"
                raise Exception(error_msg)
            
            # 下载第一张图像
            output_image = self._decode_image_from_url(images[0])
            
            # 更新种子
            self.current_seed = seed
            
            status = f"成功生成图像，种子: {seed}，尺寸: {width}x{height}"
            
            return (output_image, status)
            
        except Exception as e:
            error_str = str(e)
            print(f"生成失败: {error_str}")
            print(traceback.format_exc())
            
            # 返回错误状态
            # 创建一个空白图像作为错误占位符
            if HAS_NUMPY and HAS_TORCH:
                error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
            
            return (error_image, f"生成失败: {error_str}")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "DoubaoSeedreamNode": DoubaoSeedreamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedreamNode": "🥟豆包 SEEDREAM 4.0"
}

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

DOUBAO_SEEDREAM5_MODELS = {
    "doubao-seedream-4-5-251128": "豆包SEEDREAM 5",
}

class DoubaoSeedream5Node:
    """豆包 SEEDREAM 5 图像生成节点"""
    
    def __init__(self):
        self.current_seed = 21

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

        if (
            input_api_key
            and input_api_key.strip()
            and input_api_key.strip() not in invalid_placeholders
        ):
            print("[DoubaoSeedream5] 使用节点中的 API 密钥")
            return input_api_key.strip()
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        # 尺寸选项：2K, 4K, Custom
        size_options = [
            "2K",
            "4K",
            "Custom"
        ]

        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫咪，坐在花园里，阳光明媚"}),
                "size": (size_options, {"default": "2K"}),
                "custom_width": ("INT", {"default": 2560, "min": 1024, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1440, "min": 1024, "max": 16384, "step": 16}),
                "model": (list(DOUBAO_SEEDREAM5_MODELS.keys()), {"default": list(DOUBAO_SEEDREAM5_MODELS.keys())[0] if DOUBAO_SEEDREAM5_MODELS else "doubao-seedream-4-5-251128"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": True}),
                "sequential_image_generation": (["enabled", "disabled"], {"default": "disabled"}),
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

    def _calculate_height_range(self, width):
        """根据宽度计算高度的有效范围
        
        方式2的限制：
        - 总像素范围：[2560x1440=3686400, 4096x4096=16777216]
        - 宽高比范围：[1/16, 16]
        
        返回：(min_height, max_height)
        """
        min_pixels = 2560 * 1440  # 3686400
        max_pixels = 4096 * 4096   # 16777216
        min_aspect_ratio = 1 / 16  # 0.0625
        max_aspect_ratio = 16
        
        if width <= 0:
            return (1, 16384)
        
        # 根据总像素限制计算高度范围
        min_height_by_pixels = max(1, int(min_pixels / width))
        max_height_by_pixels = min(16384, int(max_pixels / width))
        
        # 根据宽高比限制计算高度范围
        min_height_by_aspect = max(1, int(width / max_aspect_ratio))
        max_height_by_aspect = min(16384, int(width / min_aspect_ratio))
        
        # 取交集
        min_height = max(min_height_by_pixels, min_height_by_aspect)
        max_height = min(max_height_by_pixels, max_height_by_aspect)
        
        # 确保最小值不超过最大值
        if min_height > max_height:
            # 如果无法满足要求，返回一个合理的范围
            # 例如：如果宽度太小，可能需要更大的高度
            min_height = max(1, int(min_pixels / width))
            max_height = min(16384, int(max_pixels / width))
        
        return (min_height, max_height)

    def _validate_custom_size(self, width, height):
        """验证自定义尺寸是否符合API要求
        
        方式2的限制：
        - 总像素范围：[2560x1440=3686400, 4096x4096=16777216]
        - 宽高比范围：[1/16, 16]
        """
        total_pixels = width * height
        min_pixels = 2560 * 1440  # 3686400
        max_pixels = 4096 * 4096   # 16777216
        
        aspect_ratio = width / height if height > 0 else 0
        min_aspect_ratio = 1 / 16  # 0.0625
        max_aspect_ratio = 16
        
        errors = []
        
        if total_pixels < min_pixels:
            errors.append(f"总像素值 {total_pixels} 小于最小值 {min_pixels}")
        if total_pixels > max_pixels:
            errors.append(f"总像素值 {total_pixels} 大于最大值 {max_pixels}")
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            errors.append(f"宽高比 {aspect_ratio:.2f} 不在允许范围 [{min_aspect_ratio}, {max_aspect_ratio}]")
        
        return len(errors) == 0, errors

    def _auto_adjust_height(self, width, height):
        """根据宽度自动调整高度，使其符合API要求"""
        min_height, max_height = self._calculate_height_range(width)
        
        # 如果当前高度在有效范围内，保持不变
        if min_height <= height <= max_height:
            return height
        
        # 如果高度太小，调整为最小值
        if height < min_height:
            print(f"高度 {height} 小于最小值 {min_height}，自动调整为 {min_height}")
            return min_height
        
        # 如果高度太大，调整为最大值
        if height > max_height:
            print(f"高度 {height} 大于最大值 {max_height}，自动调整为 {max_height}")
            return max_height
        
        return height

    def _parse_size_option(self, size, custom_width, custom_height):
        """解析尺寸选项
        
        方式1 (2K/4K): 返回 ("2K" 或 "4K", None, None)
        方式2 (Custom): 返回 ("宽度x高度", 调整后的宽度, 调整后的高度)
        """
        if size == "Custom":
            # 方式2：自动调整高度并验证自定义尺寸
            # 记录原始输入值用于调试
            original_width = custom_width
            original_height = custom_height
            
            # 调试信息：显示实际接收到的值
            if custom_width % 16 != 0:
                print(f"⚠️  警告：宽度 {custom_width} 不是16的倍数（step=16），ComfyUI可能会自动对齐")
            
            min_height, max_height = self._calculate_height_range(custom_width)
            adjusted_height = self._auto_adjust_height(custom_width, custom_height)
            
            # 如果高度被调整，给出明确提示
            if adjusted_height != custom_height:
                print(f"\n{'='*60}")
                print(f"⚠️  尺寸自动调整提示")
                print(f"{'='*60}")
                print(f"输入的宽度: {original_width} (实际使用: {custom_width})")
                print(f"输入的高度: {original_height}")
                print(f"调整后的高度: {adjusted_height}")
                print(f"原因: 高度 {original_height} 不在有效范围 [{min_height}, {max_height}] 内")
                print(f"总像素值: {custom_width * adjusted_height} (要求: 3686400 - 16777216)")
                print(f"宽高比: {custom_width / adjusted_height:.2f} (要求: 0.0625 - 16)")
                print(f"{'='*60}\n")
            
            is_valid, errors = self._validate_custom_size(custom_width, adjusted_height)
            if not is_valid:
                error_msg = "自定义尺寸不符合API要求：\n" + "\n".join(errors)
                error_msg += f"\n提示：对于宽度 {custom_width}，高度应在 {min_height} 到 {max_height} 之间"
                raise ValueError(error_msg)
            
            return (f"{custom_width}x{adjusted_height}", custom_width, adjusted_height)
        else:
            # 方式1：直接返回预设值 "2K" 或 "4K"
            if size in ["2K", "4K"]:
                return (size, None, None)
            else:
                return ("2K", None, None)  # 默认值

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
            
            # 下载图像 - 禁用代理
            response = requests.get(
                image_url, 
                timeout=60,
                proxies={"http": None, "https": None}  # 禁用代理
            )
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

    def generate(self, api_key, prompt, size, custom_width, custom_height, model, seed, watermark, stream, sequential_image_generation,
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
            size_result = self._parse_size_option(size, custom_width, custom_height)
            if isinstance(size_result, tuple):
                size_value, actual_width, actual_height = size_result
            else:
                # 兼容旧格式
                size_value = size_result
                actual_width, actual_height = None, None
            
            # 收集所有输入的图像
            input_images = []
            for img in [image, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
                if img is not None:
                    input_images.append(img)

            print(f"豆包 SEEDREAM 4.5 图像生成")
            print(f"提示词: {prompt}")
            print(f"尺寸: {size_value}")
            if actual_width is not None and actual_height is not None:
                print(f"实际使用的尺寸: {actual_width}x{actual_height}")
            print(f"种子: {seed}")
            print(f"水印: {watermark}")
            print(f"流式: {stream}")
            print(f"顺序生成: {sequential_image_generation}")
            print(f"模型: {model}")
            print(f"模式: {'图生图' if len(input_images) > 0 else '文生图'}")
            if len(input_images) > 0:
                print(f"参考图数量: {len(input_images)}")

            # 获取实际使用的API密钥
            actual_api_key = self._get_api_key(api_key)
            if not actual_api_key:
                raise Exception("请在节点中填写豆包 API 密钥。请访问 https://console.volcengine.com/ark 获取。")

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
                "size": size_value,
                "stream": stream,
                "watermark": watermark,
                "sequential_image_generation": sequential_image_generation
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
            
            print("调用豆包 SEEDREAM 4.5 API...")
            print(f"请求参数: {list(request_body.keys())}")
            
            # 发送请求 - 禁用代理，因为豆包是国内服务
            response = requests.post(
                url, 
                headers=headers, 
                json=request_body, 
                timeout=120, 
                stream=stream,
                proxies={"http": None, "https": None}  # 禁用代理
            )
            
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
                    user_friendly_message = "API 密钥无效，请在节点中检查豆包 API 密钥。"
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
            
            # 构建状态信息
            if actual_width is not None and actual_height is not None:
                status = f"成功生成图像，种子: {seed}，尺寸: {actual_width}x{actual_height}"
                if actual_height != custom_height:
                    status += f" (高度已从 {custom_height} 自动调整为 {actual_height})"
            else:
                status = f"成功生成图像，种子: {seed}，尺寸: {size_value}"
            
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
    "DoubaoSeedream5Node": DoubaoSeedream5Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedream5Node": "🥟豆包 SEEDREAM 5"
}

import os
import json
import io
import base64
import traceback
import requests

# 检查依赖
HAS_TORCH = True
HAS_PIL = True
HAS_NUMPY = True

try:
    import torch
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
except ImportError:
    HAS_NUMPY = False

class GeminiImagePreviewNode:
    """
    Gemini 2.5 Flash Image Preview 独立节点 (通过OpenRouter API)

    重要说明：
    - 使用OpenRouter API访问Google Gemini 2.5 Flash Image Preview模型
    - OpenRouter的Gemini模型主要用于文本生成和图像理解，不直接生成图像
    - 如果需要图像生成功能，可能需要使用其他专门的图像生成模型
    - 本节点会显示详细的API交互日志，帮助调试和理解API响应
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
            "请输入你的API密钥",
            ""
        ]

        # 如果输入了有效的API密钥，优先使用
        if (input_api_key and
            input_api_key.strip() and
            input_api_key.strip() not in invalid_placeholders):
            print(f"[GeminiImagePreview] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('openrouter_api_key', '').strip()
                if config_api_key:
                    print(f"[GeminiImagePreview] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[GeminiImagePreview] config.json中未找到openrouter_api_key")
                    return ''
        except Exception as e:
            print(f"[GeminiImagePreview] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "Create a picture of a cat eating a nano-banana in a fancy restaurant under the Gemini constellation"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("string", "image")
    FUNCTION = "generate"
    CATEGORY = "🍎MYAPI"

    def _get_placeholder_image(self, height=512, width=512):
        """创建占位符图像，使用ComfyUI标准格式"""
        try:
            import torch
            # 创建黑色占位符图像，格式: (batch, height, width, channels)
            placeholder = torch.zeros((1, height, width, 3), dtype=torch.float32)
            print(f"Created placeholder image: {placeholder.shape}")
            return placeholder
        except Exception as e:
            print(f"Error creating placeholder image: {str(e)}")
            # 如果torch不可用，返回None
            return None

    def _image_to_base64(self, image):
        """将ComfyUI图像转换为base64格式"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: Pillow 或 numpy")
            
            # 转换为numpy数组
            if HAS_TORCH and hasattr(image, 'cpu'):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)
            
            # 确保是正确的格式 (batch, height, width, channels)
            if len(image_np.shape) == 4 and image_np.shape[0] == 1:
                image_np = image_np[0]  # 移除批次维度
            
            # 转换到0-255范围
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(image_np)
            
            # 控制图像体积，避免传输超过接口限制
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

    def _bytes_to_image_tensor(self, image_bytes):
        """将字节数据转换为ComfyUI图像张量"""
        try:
            from PIL import Image
            import numpy as np
            import torch
            
            # 打开图像并确保是RGB格式
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为numpy数组，归一化到0-1范围
                np_img = np.array(img).astype(np.float32) / 255.0
            
            # 添加批次维度 (1, H, W, 3) - ComfyUI标准格式
            image_tensor = np_img[np.newaxis, ...]
            
            # 转换为torch张量
            tensor = torch.from_numpy(image_tensor)
            
            print(f"Successfully converted image bytes to tensor: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"Error converting bytes to image tensor: {str(e)}")
            raise Exception(f"Error converting bytes to image tensor: {str(e)}")

    def _base64_to_image_tensor(self, base64_url):
        """将base64图像URL转换为ComfyUI图像张量"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: Pillow 或 numpy")

            # 解析base64数据
            if base64_url.startswith('data:image/'):
                # 移除data:image/png;base64,前缀
                base64_data = base64_url.split(',', 1)[1]
            else:
                base64_data = base64_url

            # 解码base64
            import base64
            image_bytes = base64.b64decode(base64_data)

            # 使用PIL打开图像
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))

            # 确保是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0

            # 转换为ComfyUI格式 (batch, height, width, channels)
            image_tensor = np.expand_dims(image_np, axis=0)

            if HAS_TORCH:
                return torch.from_numpy(image_tensor)
            else:
                return image_tensor

        except Exception as e:
            raise Exception(f"Error converting base64 to image tensor: {str(e)}")

    def generate(self, api_key, prompt, max_tokens=4096, temperature=1.0, top_p=0.95, seed=0,
                 image=None, image2=None, image3=None, image4=None, image5=None):
        """主生成函数"""
        
        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            return ("Error: 请输入API密钥或在config.json中配置openrouter_api_key。请访问 https://openrouter.ai/ 获取API密钥。", self._get_placeholder_image())
        
        try:
            print(f"🌐 Gemini 2.5 Flash Image Preview (OpenRouter) - Processing...")

            # 收集输入图像
            input_images = []
            for img in [image, image2, image3, image4, image5]:
                if img is not None:
                    input_images.append(img)

            if len(input_images) > 0:
                print(f"📸 Input: {len(input_images)} image(s) + text prompt")

            # 构建消息内容
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # 添加图像
            for idx, img in enumerate(input_images, start=1):
                try:
                    # 转换图像为base64
                    image_base64 = self._image_to_base64(img)
                    if not image_base64:
                        raise Exception("Failed to convert image to base64")

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    })

                except Exception as e:
                    return (f"Error processing image {idx}: {str(e)}", self._get_placeholder_image())

            # 构建请求数据
            request_data = {
                "model": "google/gemini-2.5-flash-image-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

            # 添加种子（如果提供）
            if seed != 0:
                # 确保种子值在INT32范围内 (-2147483648 到 2147483647)
                if seed > 2147483647:
                    # 如果种子值太大，使用模运算将其限制在INT32范围内
                    seed = seed % 2147483647
                    print(f"⚠️  种子值过大，已调整为: {seed}")
                elif seed < -2147483648:
                    seed = -((-seed) % 2147483647)
                    print(f"⚠️  种子值过小，已调整为: {seed}")

                request_data["seed"] = seed
                print(f"🎲 使用种子值: {seed}")

            # 移除可能不被支持的参数，使用标准的OpenAI兼容格式
            # request_data["stream"] = False  # 默认就是非流式

            # 不添加extra_body，因为OpenRouter可能不支持这些自定义参数
            # request_data["extra_body"] = {
            #     "response_modalities": ["TEXT", "IMAGE"],
            #     "image_generation": True
            # }

            # 调用OpenRouter API


            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}",
                "HTTP-Referer": "https://github.com/zealman2025/Comfyui-MyApi",
                "X-Title": "ComfyUI MyAPI Plugin"
            }



            # 修正请求头中的Authorization
            headers["Authorization"] = f"Bearer {actual_api_key}"

            try:
                print(f"🚀 发送API请求...")
                print(f"   请求体大小: {len(json.dumps(request_data))} 字节")

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=request_data,
                    timeout=120
                )

                print(f"📨 收到API响应")
                print(f"   状态码: {response.status_code}")
                print(f"   响应头:")
                for key, value in response.headers.items():
                    if key.lower() in ['content-type', 'content-length', 'x-ratelimit-remaining', 'x-ratelimit-reset']:
                        print(f"     {key}: {value}")

                if response.status_code != 200:
                    print(f"❌ API请求失败")
                    error_msg = f"API请求失败 (状态码: {response.status_code})"
                    try:
                        error_data = response.json()
                        print(f"   错误响应: {json.dumps(error_data, indent=2, ensure_ascii=False)}")

                        if 'error' in error_data:
                            api_error = error_data['error'].get('message', '未知错误')
                            error_msg += f": {api_error}"

                            # 检查是否是种子值相关的错误
                            if 'seed' in api_error.lower() or 'TYPE_INT32' in api_error:
                                error_msg += "\n💡 建议: 种子值超出范围，请使用较小的种子值（0-2147483647）"

                            # 检查是否是模型不支持的功能
                            if 'not supported' in api_error.lower() or 'invalid' in api_error.lower():
                                error_msg += "\n💡 建议: 该模型可能不支持图像生成，主要用于图像理解和文本生成"
                    except Exception as e:
                        print(f"   无法解析错误响应: {str(e)}")
                        print(f"   原始响应: {response.text[:500]}...")
                        error_msg += f": {response.text}"
                    return (f"Error: {error_msg}", self._get_placeholder_image())

                result = response.json()



                # 提取响应内容
                if 'choices' not in result or len(result['choices']) == 0:
                    return ("Error: API响应中没有找到生成内容", self._get_placeholder_image())

                choice = result['choices'][0]
                message = choice.get('message', {})
                content = message.get('content', '')



                # 检查是否有图像数据
                has_image = False
                generated_image = None

                # 首先检查message.images字段（OpenRouter Gemini的标准格式）
                if 'images' in message and message['images']:
                    images = message['images']
                    if isinstance(images, list) and len(images) > 0:
                        first_image = images[0]
                        if isinstance(first_image, dict) and 'image_url' in first_image:
                            image_url_obj = first_image['image_url']
                            if isinstance(image_url_obj, dict) and 'url' in image_url_obj:
                                image_url = image_url_obj['url']
                                if image_url.startswith('data:image/'):
                                    try:
                                        # 解析base64图像
                                        generated_image = self._base64_to_image_tensor(image_url)
                                        has_image = True
                                        print(f"🎨 成功生成图像！")
                                    except Exception as e:
                                        print(f"❌ 图像解析失败: {str(e)}")



                # 处理结果
                if has_image and generated_image is not None:
                    final_content = content if content else "图像生成成功"
                    return (str(final_content), generated_image)
                else:
                    final_content = content if content else "请求处理完成"
                    if not has_image and isinstance(final_content, str):
                        final_content += "\n\n[注意: 未生成新图像，返回占位符图像]"
                    return (str(final_content), self._get_placeholder_image())

            except requests.exceptions.Timeout:
                print(f"⏰ API请求超时")
                return ("Error: API请求超时，请稍后重试", self._get_placeholder_image())
            except requests.exceptions.RequestException as e:
                print(f"🌐 网络请求失败: {str(e)}")
                return (f"Error: 网络请求失败：{str(e)}", self._get_placeholder_image())
            except json.JSONDecodeError as e:
                print(f"📄 JSON解析失败: {str(e)}")
                print(f"   原始响应: {response.text[:500]}...")
                return ("Error: API响应格式错误", self._get_placeholder_image())
            except Exception as e:
                print(f"❌ 未知错误: {str(e)}")
                import traceback
                print(f"   错误详情: {traceback.format_exc()}")
                return (f"Error: API调用失败：{str(e)}", self._get_placeholder_image())
            
        except Exception as e:
            print(f"Unexpected error in Gemini Image Preview: {str(e)}")
            return (f"Error: {str(e)}", self._get_placeholder_image())

NODE_CLASS_MAPPINGS = {
    "GeminiImagePreviewNode": GeminiImagePreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImagePreviewNode": "🌐Gemini 2.5 Flash 图像预览 (OpenRouter)"
}

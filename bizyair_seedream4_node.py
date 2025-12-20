import os
import json
import io
import base64
import traceback
import random
import tempfile
import uuid

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

class BizyAirSeedream45Node:
    """
    BizyAir Seedream4.5专用节点
    专门用于调用BizyAir的Seedream4.5模型API
    支持图像输入、提示词、尺寸选择和自定义宽高
    """

    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        # 尝试获取ComfyUI的根目录和input目录
        self.comfyui_root = None
        self.input_dir = None
        try:
            # ComfyUI通常会在环境变量或配置中设置输入目录
            # 尝试从常见位置获取
            self.comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            potential_input = os.path.join(self.comfyui_root, "input")
            if os.path.exists(potential_input):
                self.input_dir = potential_input
            else:
                # 如果input目录不存在，尝试创建它
                try:
                    os.makedirs(potential_input, exist_ok=True)
                    self.input_dir = potential_input
                except:
                    pass
        except:
            pass
        
        # 如果找不到input目录，使用临时目录
        if self.input_dir is None:
            self.input_dir = tempfile.gettempdir()

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
            print(f"[BizyAirSeedream45] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('bizyair_api_key', '').strip()
                if config_api_key:
                    print(f"[BizyAirSeedream45] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[BizyAirSeedream45] config.json中未找到bizyair_api_key")
                    return ''
        except Exception as e:
            print(f"[BizyAirSeedream45] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "size": (["2K", "4K", "Custom"], {"default": "2K"}),
                "custom_width": ("INT", {"default": 2048, "min": 1024, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 2048, "min": 1024, "max": 4096, "step": 16}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "optimize_prompt": (["enabled", "disabled"], {"default": "disabled"}),
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

    def _validate_custom_size(self, width, height):
        """验证自定义尺寸是否符合API要求
        
        要求：
        - 总像素范围：[3686400, 16777216]
        - 宽高比范围：[1/16, 16]
        """
        total_pixels = width * height
        min_pixels = 3686400  # 2560x1440
        max_pixels = 16777216  # 4096x4096
        
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

    def _parse_size_option(self, size_option, custom_width, custom_height):
        """解析size选项，返回实际的width和height"""
        if size_option == "Custom":
            # 验证自定义尺寸
            is_valid, errors = self._validate_custom_size(custom_width, custom_height)
            if not is_valid:
                error_msg = "自定义尺寸不符合API要求：\n" + "\n".join(errors)
                error_msg += f"\n提示：总像素应在 [{3686400}, {16777216}] 范围内，宽高比应在 [1/16, 16] 范围内"
                raise ValueError(error_msg)
            return custom_width, custom_height
        
        # 预设尺寸映射（根据用户提供的推荐值）
        size_mappings = {
            "2K": (2048, 2048),  # 默认2K
            "4K": (4096, 4096),   # 默认4K
        }
        
        if size_option in size_mappings:
            return size_mappings[size_option]
        
        # 默认返回2K
        return (2048, 2048)

    def _image_to_local_file(self, image):
        """将图像保存为本地文件并返回本地URL"""
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
            
            # 检查并压缩图像大小（Seedream 4.0只支持最大10MB）
            max_size_mb = 10
            max_size_bytes = max_size_mb * 1024 * 1024
            
            # 先尝试保存为PNG检查大小
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_size = buffer.tell()
            image_format = 'PNG'
            
            # 如果图像太大，尝试压缩
            if image_size > max_size_bytes:
                print(f"Warning: Image size ({image_size / 1024 / 1024:.2f}MB) exceeds {max_size_mb}MB limit. Attempting to compress...")
                
                # 计算压缩比例
                scale_factor = (max_size_bytes / image_size) ** 0.5
                new_width = int(pil_image.width * scale_factor)
                new_height = int(pil_image.height * scale_factor)
                
                # 确保最小尺寸
                new_width = max(new_width, 512)
                new_height = max(new_height, 512)
                
                # 调整图像大小
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized image to {new_width}x{new_height}")
                
                # 重新检查大小
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', optimize=True)
                image_size = buffer.tell()
                
                # 如果还是太大，尝试JPEG格式（质量较低）
                if image_size > max_size_bytes:
                    print("PNG still too large, trying JPEG format...")
                    quality = 85
                    while image_size > max_size_bytes and quality > 30:
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
                        image_size = buffer.tell()
                        if image_size > max_size_bytes:
                            quality -= 10
                            print(f"Trying JPEG quality {quality}...")
                    
                    if image_size > max_size_bytes:
                        raise ValueError(f"Image is too large even after compression ({image_size / 1024 / 1024:.2f}MB). Please use a smaller image.")
                    
                    image_format = 'JPEG'
            
            # 生成唯一的文件名
            filename = f"bizyair_seedream4_{uuid.uuid4().hex[:8]}.{image_format.lower()}"
            filepath = os.path.join(self.input_dir, filename)
            
            # 保存图像到文件
            pil_image.save(filepath, format=image_format, optimize=True)
            print(f"Saved image to local file: {filepath} ({os.path.getsize(filepath) / 1024 / 1024:.2f}MB)")
            
            # 返回相对路径（相对于ComfyUI根目录）
            # API服务器可能无法访问绝对路径，使用相对路径可能更合适
            if self.comfyui_root and filepath.startswith(self.comfyui_root):
                # 计算相对于ComfyUI根目录的路径
                relative_path = os.path.relpath(filepath, self.comfyui_root)
                # 统一使用正斜杠（跨平台兼容）
                relative_path = relative_path.replace('\\', '/')
                print(f"Using relative path: {relative_path}")
                return relative_path
            else:
                # 如果无法计算相对路径，返回文件名（API可能只需要文件名）
                filename = os.path.basename(filepath)
                print(f"Using filename only: {filename}")
                return filename
            
        except Exception as e:
            print(f"Error saving image to local file: {str(e)}")
            print(traceback.format_exc())
            return None

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
            
            # Seedream 4.0服务端限制图像最大10MB。
            # Base64编码会膨胀约1/3，因此我们把原始图像压缩到最多约7MB，保证编码后仍低于10MB。
            max_size_mb = 10
            max_size_bytes = max_size_mb * 1024 * 1024
            target_raw_bytes = int(max_size_bytes * 0.7)  # 约7MB
            min_dim = 512
            
            def save_image_to_buffer(img, fmt, **save_kwargs):
                buf = io.BytesIO()
                img.save(buf, format=fmt, **save_kwargs)
                size = buf.tell()
                return buf, size
            
            # 初始保存为PNG
            buffer, raw_size = save_image_to_buffer(pil_image, 'PNG', optimize=True)
            image_format = 'PNG'
            
            # 如果原图太大，循环压缩，先缩放分辨率
            if raw_size > target_raw_bytes:
                print(f"Warning: Image raw size ({raw_size / 1024 / 1024:.2f}MB) exceeds target {target_raw_bytes / 1024 / 1024:.2f}MB. Compressing...")
            
            resize_attempts = 0
            while raw_size > target_raw_bytes and (pil_image.width > min_dim or pil_image.height > min_dim) and resize_attempts < 5:
                scale_factor = max((target_raw_bytes / raw_size) ** 0.5, 0.3)
                new_width = max(int(pil_image.width * scale_factor), min_dim)
                new_height = max(int(pil_image.height * scale_factor), min_dim)
                if new_width == pil_image.width and new_height == pil_image.height:
                    # Scale factor太小导致尺寸不变，强制缩小一截
                    new_width = max(int(pil_image.width * 0.75), min_dim)
                    new_height = max(int(pil_image.height * 0.75), min_dim)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_attempts += 1
                print(f"Resized image attempt {resize_attempts}: {new_width}x{new_height}")
                buffer, raw_size = save_image_to_buffer(pil_image, 'PNG', optimize=True)
                image_format = 'PNG'
            
            # 如果仍超出限制，切换到JPEG并降低质量
            if raw_size > target_raw_bytes:
                print("PNG still too large, switching to JPEG compression...")
                quality = 90
                jpeg_attempts = 0
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_image_to_buffer(pil_image, 'JPEG', quality=quality, optimize=True)
                    image_format = 'JPEG'
                    jpeg_attempts += 1
                    print(f"JPEG compression attempt {jpeg_attempts}: quality={quality}, size={raw_size / 1024 / 1024:.2f}MB")
                    quality -= 5
                
            # 最终检查
            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")
            
            # 转换为base64
            buffer.seek(0)
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_size_mb = len(base64_bytes) / 1024 / 1024
            print(f"Final raw size: {raw_size / 1024 / 1024:.2f}MB, base64 size: {base64_size_mb:.2f}MB, format: {image_format}")
            image_base64 = base64_bytes.decode('utf-8')
            
            # 根据格式返回相应的data URI
            if image_format == 'JPEG':
                return f"data:image/jpeg;base64,{image_base64}"
            else:
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

    def generate(self, api_key, prompt, size, custom_width, custom_height, max_images, optimize_prompt, seed,
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
        
        # 处理size选项，获取实际的width和height
        actual_width, actual_height = self._parse_size_option(size, custom_width, custom_height)
        print(f"Using size: {size}, actual dimensions: {actual_width}x{actual_height}")
        
        # 生成随机种子（如果需要）
        if seed == 0:
            seed = random.randint(1, 2147483647)  # API要求seed最大值为2147483647
        
        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir Seedream4.5 API request to: {api_url}")
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 构建input_values
            input_values = {
                "20:BizyAir_Seedream4_5.prompt": prompt,
                "20:BizyAir_Seedream4_5.model": "doubao-seedream-4-5-251128",  # Seedream4.5模型
                "20:BizyAir_Seedream4_5.size": size,
                "20:BizyAir_Seedream4_5.max_images": max_images,
                "20:BizyAir_Seedream4_5.optimize_prompt": optimize_prompt
            }
            
            # 只有当size是"Custom"时才发送custom_width和custom_height
            if size == "Custom":
                input_values["20:BizyAir_Seedream4_5.custom_width"] = actual_width
                input_values["20:BizyAir_Seedream4_5.custom_height"] = actual_height
            
            # 图片输入键名映射（按顺序）
            image_key_mapping = [
                "18:LoadImage.image",  # image
                "21:LoadImage.image",  # image2
                "23:LoadImage.image",  # image3
                "22:LoadImage.image",  # image4
                "24:LoadImage.image",  # image5
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
            input_count = image_count if image_count > 0 else 1  # 如果没有图片，默认1
            input_values["20:BizyAir_Seedream4_5.inputcount"] = input_count
            print(f"Input count set to: {input_count} (images provided: {image_count})")
            
            # 构建请求数据
            data = {
                "web_app_id": 41504,  # Seedream4.5的固定web_app_id
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}, input_values count={len(input_values)}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Size: {size} ({actual_width}x{actual_height})")
            print(f"Max images: {max_images}, Optimize prompt: {optimize_prompt}")
            
            # 发送请求（增加超时时间到120秒）
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            print("API response received")
            print(f"Full API response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查响应状态
            if result.get("status") != "Success":
                # 尝试提取详细的错误信息
                error_details = []
                status = result.get("status", "Unknown")
                error_details.append(f"状态: {status}")
                
                # 首先检查outputs中的错误信息（这是BizyAir API返回错误的主要位置）
                outputs = result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    if output.get("error_msg"):
                        error_details.append(f"错误消息: {output.get('error_msg').strip()}")
                    if output.get("error_type"):
                        error_details.append(f"错误类型: {output.get('error_type')}")
                
                # 检查响应根级别的错误信息字段
                if result.get("error_message"):
                    error_details.append(f"错误消息: {result.get('error_message')}")
                if result.get("message"):
                    error_details.append(f"消息: {result.get('message')}")
                if result.get("error"):
                    error_details.append(f"错误: {result.get('error')}")
                if result.get("details"):
                    error_details.append(f"详情: {result.get('details')}")
                if result.get("reason"):
                    error_details.append(f"原因: {result.get('reason')}")
                
                error_msg = "API请求失败: " + " | ".join(error_details)
                print(f"Error details: {error_msg}")
                raise Exception(error_msg)
            
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
                "web_app_id": 41504,
                "prompt": prompt,
                "size": size,
                "dimensions": f"{actual_width}x{actual_height}",
                "max_images": max_images,
                "optimize_prompt": optimize_prompt,
                "seed": seed,
                "image_url": image_url,
                "cost_time": result.get("cost_times", {}).get("total_cost_time", 0),
                "request_id": result.get("request_id", "")
            }
            
            status_text = f"✅ Seedream4.5生成成功\n"
            status_text += f"提示词: {prompt[:50]}...\n"
            status_text += f"尺寸: {size} ({actual_width}x{actual_height})\n"
            status_text += f"最大图像数: {max_images}, 优化提示词: {optimize_prompt}\n"
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
    "BizyAirSeedream45Node": BizyAirSeedream45Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirSeedream45Node": "🌐BizyAir Seedream 4.5 (需BizyAir.cn充值金币)"
}

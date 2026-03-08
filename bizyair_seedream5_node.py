import os
import json
import io
import traceback
import tempfile
import uuid

try:
    from .bizyair_upload import upload_image_to_bizyair
except ImportError:
    from bizyair_upload import upload_image_to_bizyair

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

class BizyAirSeedream5Node:
    """
    BizyAir Seedream5 专用节点
    无比例节点，使用 size=Custom + custom_width/custom_height
    支持 inputcount (1-6) 控制输入图像数量，主图 + 最多5张参考图
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
            print(f"[BizyAirSeedream5] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('bizyair_api_key', '').strip()
                if config_api_key:
                    print(f"[BizyAirSeedream5] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[BizyAirSeedream5] config.json中未找到bizyair_api_key")
                    return ''
        except Exception as e:
            print(f"[BizyAirSeedream5] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "输入提示词"}),
                "size": (["Custom"], {"default": "Custom"}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 16}),
                "inputcount": ("INT", {"default": 6, "min": 1, "max": 6, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
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
            
            # 检查并压缩图像大小（Seedream5 只支持最大10MB）
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
            filename = f"bizyair_seedream5_{uuid.uuid4().hex[:8]}.{image_format.lower()}"
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

    def _image_to_bytes(self, image):
        """将图像转换为字节流（用于 OSS 上传），返回 (bytes, file_ext)"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                return None, None
            
            if HAS_TORCH and hasattr(image, 'cpu'):
                image_np = image.cpu().numpy()
            else:
                image_np = image
            
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            
            if len(image_np.shape) == 4:
                image_np = image_np[0]
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                pil_image = Image.fromarray(image_np, 'RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image_np.shape}")
            
            max_size_mb = 10
            max_size_bytes = max_size_mb * 1024 * 1024
            target_raw_bytes = int(max_size_bytes * 0.7)
            min_dim = 512
            
            def save_image_to_buffer(img, fmt, **save_kwargs):
                buf = io.BytesIO()
                img.save(buf, format=fmt, **save_kwargs)
                return buf, buf.tell()
            
            buffer, raw_size = save_image_to_buffer(pil_image, 'PNG', optimize=True)
            image_format = 'PNG'
            
            if raw_size > target_raw_bytes:
                print(f"Warning: Image raw size ({raw_size / 1024 / 1024:.2f}MB) exceeds target. Compressing...")
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
                buffer, raw_size = save_image_to_buffer(pil_image, 'PNG', optimize=True)
            
            if raw_size > target_raw_bytes:
                quality = 90
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_image_to_buffer(pil_image, 'JPEG', quality=quality, optimize=True)
                    image_format = 'JPEG'
                    quality -= 5
            
            if raw_size > target_raw_bytes:
                raise ValueError(f"Image is too large even after compression ({raw_size / 1024 / 1024:.2f}MB). Please use a smaller image or resize manually.")
            
            buffer.seek(0)
            img_bytes = buffer.getvalue()
            ext = 'jpg' if image_format == 'JPEG' else 'png'
            print(f"Image prepared for OSS: {raw_size / 1024 / 1024:.2f}MB, format: {image_format}")
            return img_bytes, ext
            
        except Exception as e:
            print(f"Error converting image to bytes: {str(e)}")
            print(traceback.format_exc())
            return None, None

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

    def generate(self, api_key, prompt, size, custom_width, custom_height, inputcount,
                 image=None, image2=None, image3=None, image4=None, image5=None, image6=None):
        """生成图像（Seedream5 无比例节点，使用 Custom 尺寸 + inputcount）"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            raise Exception("请输入API密钥或在config.json中配置bizyair_api_key。请访问 https://bizyair.cn 获取API密钥。")

        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")
        
        print(f"Seedream5: size={size}, {custom_width}x{custom_height}, inputcount={inputcount}")
        
        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir Seedream5 API request to: {api_url} (web_app_id: 47120)")
            
            # 主图必须存在
            if image is None:
                raise Exception("Seedream5 需要至少一张主图 (image)")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            add_log = lambda t, m: print(f"[BizyAirSeedream5][{t}] {m}")
            
            # 上传主图
            primary_bytes, primary_ext = self._image_to_bytes(image)
            if not primary_bytes or not primary_ext:
                raise Exception("主图转换失败，请检查图像格式")
            primary_url = upload_image_to_bizyair(
                primary_bytes, actual_api_key,
                add_log_fn=add_log,
                file_name=f"seedream5_primary.{primary_ext}"
            )
            
            # 参考图节点（与 API 一致：37,39,46,48,52）
            ref_node_keys = ["37:LoadImage.image", "39:LoadImage.image", "46:LoadImage.image", "48:LoadImage.image", "52:LoadImage.image"]
            input_images = [image2, image3, image4, image5, image6]
            
            # 根据 inputcount 只使用前 N 张图（主图 + inputcount-1 张参考图）
            max_refs = min(inputcount - 1, 5)  # inputcount=1 则无参考图，最多5张参考图
            
            # 构建 input_values（54:BizyAir_Seedream5.*）
            input_values = {
                "40:LoadImage.image": primary_url,
                "54:BizyAir_Seedream5.prompt": prompt,
                "54:BizyAir_Seedream5.size": "Custom",
                "54:BizyAir_Seedream5.custom_width": custom_width,
                "54:BizyAir_Seedream5.custom_height": custom_height,
                "54:BizyAir_Seedream5.inputcount": inputcount,
            }
            
            # 上传参考图（最多 max_refs 张）
            ref_count = 0
            for idx in range(max_refs):
                if idx >= len(input_images):
                    break
                img = input_images[idx]
                if img is not None and idx < len(ref_node_keys):
                    img_bytes, ext = self._image_to_bytes(img)
                    if img_bytes and ext:
                        try:
                            ref_url = upload_image_to_bizyair(
                                img_bytes, actual_api_key,
                                add_log_fn=add_log,
                                file_name=f"seedream5_ref_{idx + 1}.{ext}"
                            )
                            input_values[ref_node_keys[idx]] = ref_url
                            ref_count += 1
                            print(f"Added ref image {idx + 1} (key: {ref_node_keys[idx]})")
                        except Exception as up_err:
                            print(f"Warning: Failed to upload ref image {idx + 1}: {up_err}")
            
            # 实际发送的图片数需与 inputcount 一致，若参考图不足则警告
            actual_count = 1 + ref_count
            if actual_count < inputcount:
                print(f"Warning: inputcount={inputcount} but only {actual_count} images provided. Using actual count.")
                input_values["54:BizyAir_Seedream5.inputcount"] = actual_count
            
            # 构建请求数据
            data = {
                "web_app_id": 47120,  # Seedream5 6图，与主插件一致
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}, inputcount={input_values['54:BizyAir_Seedream5.inputcount']}, refs={ref_count}")
            print(f"Prompt: {prompt[:100]}..., Size: {custom_width}x{custom_height}")
            debug_data = data.copy()
            debug_input_values = {k: (f"[OSS URL: {v[:60]}...]" if isinstance(v, str) and ('storage.bizyair.cn' in v or 'aliyuncs.com' in v) else v) for k, v in input_values.items()}
            debug_data['input_values'] = debug_input_values
            print(f"Request payload: {json.dumps(debug_data, indent=2, ensure_ascii=False)}")
            
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
                "web_app_id": 47120,
                "prompt": prompt,
                "dimensions": f"{custom_width}x{custom_height}",
                "inputcount": input_values["54:BizyAir_Seedream5.inputcount"],
                "image_url": image_url,
                "cost_time": result.get("cost_times", {}).get("total_cost_time", 0),
                "request_id": result.get("request_id", "")
            }
            
            status_text = f"✅ Seedream5 生成成功\n"
            status_text += f"提示词: {prompt[:50]}...\n"
            status_text += f"尺寸: {custom_width}x{custom_height}, inputcount: {status_info['inputcount']}\n"
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
    "BizyAirSeedream5Node": BizyAirSeedream5Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirSeedream5Node": "🌐BizyAir Seedream5 (与主插件一致)"
}

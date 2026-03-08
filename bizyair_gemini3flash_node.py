import os
import json
import io
import traceback

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


class BizyAirGemini3FlashVLMNode:
    """
    BizyAir Gemini3Flash VLM专用节点
    专门用于调用BizyAir的Gemini3Flash视觉语言模型API
    支持图像输入和文本提示词，返回文本描述
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
            print(f"[BizyAirGemini3FlashVLM] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('bizyair_api_key', '').strip()
                if config_api_key:
                    print(f"[BizyAirGemini3FlashVLM] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[BizyAirGemini3FlashVLM] config.json中未找到bizyair_api_key")
                    return ''
        except Exception as e:
            print(f"[BizyAirGemini3FlashVLM] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {"multiline": True, "default": "我是一个图像描述助手"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "描述这个图像内容"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
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
                buffer, raw_size = save_to_buffer(pil_image, 'PNG', optimize=True)

            if raw_size > target_raw_bytes:
                quality = 90
                while raw_size > target_raw_bytes and quality >= 40:
                    buffer, raw_size = save_to_buffer(pil_image, 'JPEG', quality=quality, optimize=True)
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

    def _download_text_from_url(self, url):
        """从URL下载文本文件并返回内容"""
        try:
            if not HAS_REQUESTS:
                raise Exception("缺少requests库")
            
            print(f"Downloading text from URL: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/plain, text/html, */*',
                'Accept-Charset': 'utf-8'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 强制使用UTF-8编码，因为现代API通常使用UTF-8
            # 不要使用response.text，因为它可能使用错误的编码
            try:
                text_content = response.content.decode('utf-8')
                print("Successfully decoded using UTF-8")
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                print("UTF-8 decode failed, trying other encodings...")
                encodings_to_try = ['gbk', 'gb2312', 'gb18030', 'big5']
                text_content = None
                
                for enc in encodings_to_try:
                    try:
                        text_content = response.content.decode(enc)
                        print(f"Successfully decoded using {enc}")
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                # 如果所有编码都失败，使用UTF-8 with errors='replace'
                if text_content is None:
                    print("Warning: All encoding attempts failed, using UTF-8 with errors='replace'")
                    text_content = response.content.decode('utf-8', errors='replace')
            
            # 清理文本内容（移除BOM等）
            if text_content.startswith('\ufeff'):
                text_content = text_content[1:]
            
            print(f"Successfully downloaded text, length: {len(text_content)} characters")
            print(f"First 100 chars: {text_content[:100]}")
            return text_content
            
        except Exception as e:
            print(f"Error downloading text from URL: {str(e)}")
            print(traceback.format_exc())
            raise

    def analyze(self, api_key, image, system_prompt, user_prompt):
        """分析图像并返回文本描述"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            raise Exception("请输入API密钥或在config.json中配置bizyair_api_key。请访问 https://bizyair.cn 获取API密钥。")

        # 检查依赖
        missing_deps = self._check_dependencies()
        if missing_deps:
            raise Exception(f"缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。")

        try:
            api_url = "https://api.bizyair.cn/w/v1/webapp/task/openapi/create"
            print(f"BizyAir Gemini3Flash VLM API request to: {api_url}")
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 将图像上传到 OSS 获取 URL（与主插件一致）
            img_bytes, ext = self._image_to_bytes(image)
            if not img_bytes or not ext:
                raise Exception("图像转换失败，请检查图像格式")
            add_log = lambda t, m: print(f"[BizyAirGemini3FlashVLM][{t}] {m}")
            image_url = upload_image_to_bizyair(img_bytes, actual_api_key, add_log_fn=add_log, file_name=f"vlm_image.{ext}")
            
            # 构建input_values（使用 OSS URL 替代 base64）
            input_values = {
                "19:LoadImage.image": image_url,
                "18:BizyAir_TRD_VLM_API.system_prompt": system_prompt,
                "18:BizyAir_TRD_VLM_API.user_prompt": user_prompt
            }
            
            # 构建请求数据
            data = {
                "web_app_id": 44360,  # Gemini3Flash VLM的固定web_app_id
                "suppress_preview_output": False,
                "input_values": input_values
            }
            
            print(f"Request data: web_app_id={data['web_app_id']}")
            print(f"System Prompt: {system_prompt[:100]}...")
            print(f"User Prompt: {user_prompt[:100]}...")
            
            # 打印请求数据（隐藏 OSS URL 详情）
            debug_data = data.copy()
            debug_input_values = {}
            for key, value in input_values.items():
                if isinstance(value, str) and ('storage.bizyair.cn' in value or 'aliyuncs.com' in value):
                    debug_input_values[key] = f"[OSS URL: {value[:60]}...]"
                else:
                    debug_input_values[key] = value
            debug_data['input_values'] = debug_input_values
            print(f"Request payload: {json.dumps(debug_data, indent=2, ensure_ascii=False)}")
            
            # 发送请求（增加超时时间到120秒）
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            print("API response received")
            print(f"API response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查响应状态
            if result.get("status") != "Success":
                # 尝试获取详细的错误信息
                error_message = result.get("status", "Unknown error")
                
                # 从outputs中提取错误信息
                outputs = result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    error_output = outputs[0]
                    error_msg = error_output.get("error_msg", "")
                    error_type = error_output.get("error_type", "")
                    if error_msg:
                        error_message = f"{error_message}: {error_msg}"
                    if error_type:
                        error_message = f"{error_message} (类型: {error_type})"
                
                # 如果没有从outputs获取到，尝试从error字段获取
                if error_message == result.get("status", "Unknown error"):
                    error_detail = result.get("error", {})
                    if isinstance(error_detail, dict):
                        error_msg = error_detail.get("message", error_detail.get("msg", ""))
                        if error_msg:
                            error_message = f"{error_message}: {error_msg}"
                
                # 打印完整的错误信息用于调试
                print(f"API错误详情: {json.dumps(result, indent=2, ensure_ascii=False)}")
                raise Exception(f"API请求失败: {error_message}")
            
            # 提取文本结果
            outputs = result.get("outputs", [])
            if not outputs:
                raise Exception("API响应中没有找到输出数据")
            
            # VLM节点返回的是文本，不是图像
            # 尝试从outputs中提取文本内容
            text_result = ""
            text_url = None
            
            # 检查outputs中的文本字段或文件URL
            for output in outputs:
                # 优先检查是否有文件URL（object_url或file_url）
                if "object_url" in output:
                    url = output["object_url"]
                    # 检查是否是文本文件URL（.txt结尾或包含_file.txt）
                    if isinstance(url, str) and (url.endswith('.txt') or '_file.txt' in url or 'storage.bizyair.cn' in url):
                        text_url = url
                        break
                elif "file_url" in output:
                    url = output["file_url"]
                    if isinstance(url, str) and (url.endswith('.txt') or '_file.txt' in url or 'storage.bizyair.cn' in url):
                        text_url = url
                        break
                # 检查直接的文本字段
                elif "text" in output:
                    text_result = output["text"]
                    if text_result and text_result.strip():
                        break
                elif "content" in output:
                    text_result = output["content"]
                    if text_result and text_result.strip():
                        break
                elif "message" in output:
                    text_result = output["message"]
                    if text_result and text_result.strip():
                        break
                elif "result" in output:
                    text_result = output["result"]
                    if text_result and text_result.strip():
                        break
            
            # 如果找到了文本文件URL，下载并读取内容
            if text_url:
                print(f"Found text file URL: {text_url}")
                text_result = self._download_text_from_url(text_url)
            # 如果还是没有找到，尝试从整个output中提取字符串
            elif not text_result and len(outputs) > 0:
                first_output = outputs[0]
                # 检查是否有URL字段
                if isinstance(first_output, dict):
                    # 检查所有可能的URL字段
                    for key in ["object_url", "file_url", "url", "text_url"]:
                        if key in first_output:
                            url_value = first_output[key]
                            if isinstance(url_value, str) and ('storage.bizyair.cn' in url_value or url_value.endswith('.txt') or '_file.txt' in url_value):
                                print(f"Found text file URL in {key}: {url_value}")
                                text_result = self._download_text_from_url(url_value)
                                break
                    
                    # 如果还没有找到，查找所有字符串值
                    if not text_result:
                        for key, value in first_output.items():
                            if isinstance(value, str) and value.strip():
                                # 检查是否是URL
                                if ('storage.bizyair.cn' in value or value.endswith('.txt') or '_file.txt' in value):
                                    print(f"Found text file URL in {key}: {value}")
                                    text_result = self._download_text_from_url(value)
                                    break
                                else:
                                    text_result = value
                                    break
                    
                    # 如果还是没有，尝试将整个字典转换为JSON字符串
                    if not text_result:
                        text_result = json.dumps(first_output, ensure_ascii=False, indent=2)
                elif isinstance(first_output, str):
                    # 检查是否是URL
                    if 'storage.bizyair.cn' in first_output or first_output.endswith('.txt') or '_file.txt' in first_output:
                        print(f"Found text file URL: {first_output}")
                        text_result = self._download_text_from_url(first_output)
                    else:
                        text_result = first_output
            
            if not text_result:
                # 如果仍然没有找到文本，返回整个响应的JSON
                text_result = json.dumps(result, ensure_ascii=False, indent=2)
                print("Warning: 无法从API响应中提取文本，返回完整响应JSON")
            
            print(f"Extracted text result: {text_result[:200]}...")
            
            # 打印状态信息到控制台（用于调试）
            cost_time = result.get("cost_times", {}).get("total_cost_time", 0)
            request_id = result.get("request_id", "")
            print(f"✅ Gemini3Flash VLM分析成功")
            print(f"系统提示词: {system_prompt[:50]}...")
            print(f"用户提示词: {user_prompt[:50]}...")
            print(f"文本长度: {len(text_result)} 字符")
            print(f"耗时: {cost_time}ms")
            print(f"请求ID: {request_id}")
            
            # 直接返回纯文本结果，便于ComfyUI解析和显示
            return (text_result,)
            
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
    "BizyAirGemini3FlashVLMNode": BizyAirGemini3FlashVLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirGemini3FlashVLMNode": "🌐BizyAir Gemini3Flash VLM (需BizyAir.cn充值金币)"
}

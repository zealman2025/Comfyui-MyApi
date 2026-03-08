import os
import json
import io
import base64
import traceback
import requests

# 尝试导入依赖
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

# 不再需要 dashscope SDK，直接使用 HTTP 请求
# 保留导入检查以便将来可能需要
try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def load_qwen_image_edit_models_from_config():
    """从config.json加载Qwen图像编辑模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            image_edit_models = models.get('qwen_image_edit', {})
            if image_edit_models:
                return image_edit_models
            # 如果没有配置，返回默认模型（qwen-image-2.0-pro）
            return {
                "qwen-image-2.0-pro": "Qwen Image 2.0 Pro"
            }
    except Exception as e:
        print(f"[QwenImageEditPlusNode] 加载配置失败: {str(e)}")
        traceback.print_exc()
        return {
            "qwen-image-2.0-pro": "Qwen Image 2.0 Pro"
        }


QWEN_IMAGE_EDIT_MODELS = load_qwen_image_edit_models_from_config()


class QwenImageEditPlusNode:
    """Qwen Image Edit Plus 图像编辑节点"""
    
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

    def _get_api_key(self, input_api_key):
        """获取API密钥，优先使用输入的密钥，否则从config.json读取"""
        invalid_placeholders = [
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥",
            ""
        ]

        if (input_api_key and
            input_api_key.strip() and
            input_api_key.strip() not in invalid_placeholders):
            print(f"[QwenImageEditPlusNode] 使用输入的API密钥")
            return input_api_key.strip()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('qwen_api_key', '').strip()
                if config_api_key:
                    print(f"[QwenImageEditPlusNode] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[QwenImageEditPlusNode] config.json中未找到qwen_api_key")
                    return ''
        except Exception as e:
            print(f"[QwenImageEditPlusNode] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(QWEN_IMAGE_EDIT_MODELS.keys()), {"default": list(QWEN_IMAGE_EDIT_MODELS.keys())[0] if QWEN_IMAGE_EDIT_MODELS else "qwen-image-2.0-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": " "}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "edit"
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

    def _image_to_base64_data_url(self, image):
        """将图像转换为base64 data URL格式"""
        try:
            if not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: Pillow 或 numpy")
            
            if image is None:
                raise ValueError("Image is None")
            
            # 处理PyTorch张量
            if HAS_TORCH and isinstance(image, torch.Tensor):
                if image.is_cuda:
                    image = image.cpu()
                image = image.numpy()
            
            # 处理ComfyUI的图像格式
            if HAS_NUMPY and isinstance(image, np.ndarray):
                # 处理批处理维度
                if len(image.shape) == 4:
                    image = image[0]
                
                # 确保图像是3通道的
                if len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = image[:, :, :3]
                    elif image.shape[2] == 1:  # 灰度
                        image = np.repeat(image, 3, axis=2)
                
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
            
            # 转换为base64 - 使用JPEG格式以减小体积（根据API文档，支持JPEG和PNG）
            buffer = io.BytesIO()
            # 优先使用JPEG格式（体积更小），质量设置为95
            pil_image.save(buffer, format='JPEG', quality=95, optimize=True)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # 返回符合API文档要求的格式：data:{mime_type};base64,{base64_data}
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            print(traceback.format_exc())
            raise

    def _download_image_from_url(self, image_url):
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            if not HAS_REQUESTS or not HAS_PIL or not HAS_NUMPY:
                raise ImportError("缺少必要的依赖: requests, Pillow 或 numpy")
            
            print(f"[QwenImageEditPlusNode] 下载图像: {image_url}")
            
            # 使用流式下载，设置300秒超时（根据官方示例）- 禁用代理
            response = requests.get(
                image_url, 
                stream=True, 
                timeout=300,
                proxies={"http": None, "https": None}  # 禁用代理
            )
            response.raise_for_status()
            
            # 使用 iter_content 进行流式下载
            image_bytes = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    image_bytes.write(chunk)
            
            image_bytes.seek(0)
            image = Image.open(image_bytes)
            
            # 确保是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组，归一化到0-1范围
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # 添加批处理维度 [1, H, W, C] - ComfyUI标准格式
            image_tensor = image_array[np.newaxis, ...]
            
            # 转换为torch张量（如果可用）
            if HAS_TORCH:
                image_tensor = torch.from_numpy(image_tensor)
            
            print(f"[QwenImageEditPlusNode] 图像下载成功，尺寸: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            print(f"[QwenImageEditPlusNode] 图像下载失败: {str(e)}")
            print(traceback.format_exc())
            raise

    def edit(self, api_key, model, prompt, negative_prompt=" ", watermark=False, seed=0,
             image1=None, image2=None, image3=None):
        """图像编辑函数"""
        missing_deps = self._check_dependencies()
        if missing_deps:
            error_msg = f"Error: 缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。"
            if HAS_NUMPY and HAS_TORCH:
                error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
            return (error_image, error_msg)

        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            error_msg = "Error: 请输入API密钥或在config.json中配置qwen_api_key。请访问 https://bailian.console.aliyun.com/?tab=api#/api 获取API密钥。"
            if HAS_NUMPY and HAS_TORCH:
                error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
            return (error_image, error_msg)

        if not prompt or not prompt.strip():
            error_msg = "Error: 请输入提示词。"
            if HAS_NUMPY and HAS_TORCH:
                error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
            return (error_image, error_msg)

        try:
            # n 固定为 1
            n = 1
            
            print(f"[QwenImageEditPlusNode] 图像编辑请求:")
            print(f"  模型: {model}")
            print(f"  提示词: {prompt}")
            print(f"  负向提示词: {negative_prompt}")
            print(f"  生成数量: {n} (固定)")
            print(f"  水印: {watermark}")
            print(f"  种子: {seed if seed > 0 else '自动生成'}")

            # 构建消息内容
            content = []
            
            # 添加图像（最多3张）
            input_images = [image1, image2, image3]
            image_count = 0
            for img in input_images:
                if img is not None:
                    try:
                        image_data_url = self._image_to_base64_data_url(img)
                        content.append({"image": image_data_url})
                        image_count += 1
                        print(f"已添加图像 {image_count}")
                    except Exception as e:
                        print(f"处理图像失败: {str(e)}")
                        continue

            if image_count == 0:
                error_msg = "Error: 至少需要提供1张输入图像。"
                if HAS_NUMPY and HAS_TORCH:
                    error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                else:
                    error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
                return (error_image, error_msg)

            # 添加文本提示
            content.append({"text": prompt})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            print(f"[QwenImageEditPlusNode] 调用 DashScope API...")
            
            # 直接使用 HTTP 请求，确保请求格式完全符合 API 文档
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}"
            }
            
            # 构建请求体 - 完全按照 curl 示例格式
            request_body = {
                "model": model,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "n": n,
                    "negative_prompt": negative_prompt if negative_prompt.strip() else " ",
                    "watermark": watermark
                }
            }
            
            # 只有当seed > 0时才添加seed参数（根据API文档，0表示自动生成）
            if seed > 0:
                request_body["parameters"]["seed"] = seed
            
            print(f"[QwenImageEditPlusNode] 请求URL: {url}")
            print(f"[QwenImageEditPlusNode] 请求体结构: model, input.messages, parameters")
            
            # 发送 HTTP 请求 - 禁用代理，因为Qwen是国内服务
            response = requests.post(
                url, 
                headers=headers, 
                json=request_body, 
                timeout=300,
                proxies={"http": None, "https": None}  # 禁用代理
            )

            print(f"[QwenImageEditPlusNode] 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                print(f"[QwenImageEditPlusNode] API调用成功")
                
                # 解析 JSON 响应 - 根据官方响应格式
                try:
                    result = response.json()
                    print(f"[QwenImageEditPlusNode] 响应解析成功")
                    
                    # 根据 curl 示例的响应格式解析
                    # {
                    #   "status_code": 200,
                    #   "output": {
                    #     "choices": [{
                    #       "message": {
                    #         "content": [{"image": "url"}, ...]
                    #       }
                    #     }]
                    #   }
                    # }
                    output_images = []
                    
                    if "output" in result and "choices" in result["output"]:
                        choices = result["output"]["choices"]
                        if choices and len(choices) > 0:
                            message = choices[0].get("message", {})
                            content = message.get("content", [])
                            
                            for i, content_item in enumerate(content):
                                if isinstance(content_item, dict) and "image" in content_item:
                                    image_url = content_item["image"]
                                    output_images.append(image_url)
                                    print(f"[QwenImageEditPlusNode] 输出图像{i+1}的URL: {image_url}")
                    
                    if not output_images:
                        # 打印完整响应用于调试
                        print(f"[QwenImageEditPlusNode] 调试：完整响应 = {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                except json.JSONDecodeError as e:
                    print(f"[QwenImageEditPlusNode] JSON解析失败: {str(e)}")
                    print(f"[QwenImageEditPlusNode] 原始响应: {response.text[:500]}")
                    output_images = []
                except Exception as e:
                    print(f"[QwenImageEditPlusNode] 解析响应时出错: {str(e)}")
                    print(traceback.format_exc())
                    output_images = []

                if not output_images:
                    error_msg = "Error: API响应中没有找到图像数据。请检查响应格式。"
                    print(f"[QwenImageEditPlusNode] 完整响应: {response}")
                    if HAS_NUMPY and HAS_TORCH:
                        error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                    else:
                        error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
                    return (error_image, error_msg)

                # 下载第一张图像
                try:
                    output_image = self._download_image_from_url(output_images[0])
                except Exception as e:
                    error_msg = f"Error: 下载图像失败: {str(e)}"
                    print(f"[QwenImageEditPlusNode] {error_msg}")
                    if HAS_NUMPY and HAS_TORCH:
                        error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                    else:
                        error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
                    return (error_image, error_msg)
                
                status_msg = f"成功生成图像，共{len(output_images)}张，使用第1张"
                if len(output_images) > 1:
                    status_msg += f"\n其他图像URL:\n" + "\n".join([f"  - {url}" for url in output_images[1:]])
                
                return (output_image, status_msg)
            else:
                # 尝试解析错误响应
                try:
                    error_result = response.json()
                    error_msg = f"Error: HTTP返回码：{response.status_code}"
                    if "code" in error_result:
                        error_msg += f"\n错误码：{error_result['code']}"
                    if "message" in error_result:
                        error_msg += f"\n错误信息：{error_result['message']}"
                except:
                    error_msg = f"Error: HTTP返回码：{response.status_code}\n响应内容：{response.text[:200]}"
                
                print(f"[QwenImageEditPlusNode] {error_msg}")
                
                if HAS_NUMPY and HAS_TORCH:
                    error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                else:
                    error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
                return (error_image, error_msg)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[QwenImageEditPlusNode] {error_msg}")
            print(traceback.format_exc())
            
            if HAS_NUMPY and HAS_TORCH:
                error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                error_image = np.zeros((1, 512, 512, 3), dtype=np.float32)
            return (error_image, error_msg)


NODE_CLASS_MAPPINGS = {
    "QwenImageEditPlusNode": QwenImageEditPlusNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditPlusNode": "🍭Qwen 图像编辑增强版"
}


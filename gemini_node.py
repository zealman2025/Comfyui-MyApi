import os
import json
import io

# 从配置文件加载模型配置
def load_gemini_models_from_config():
    """从config.json加载Gemini模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            gemini_models = models.get('gemini', {})
            return gemini_models
    except Exception as e:
        print(f"[GeminiNode] Error loading Gemini models from config: {str(e)}")
        import traceback
        traceback.print_exc()
        # 提供默认模型作为回退
        default_models = {
            "gemini-2.5-pro": "Gemini 2.5 pro",
            "gemini-2.5-flash": "Gemini 2.5 flash"
        }
        print(f"[GeminiNode] Using default models: {default_models}")
        return default_models

# 加载模型配置
GEMINI_MODELS = load_gemini_models_from_config()

class GeminiAINode:
    """
    Gemini AI节点，支持多种Gemini模型
    """

    def __init__(self):
        self.current_seed = 0  # 初始化种子值
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
            print(f"[GeminiNode] 使用输入的API密钥")
            return input_api_key.strip()

        # 否则从config.json读取
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_api_key = config.get('gemini_api_key', '').strip()
                if config_api_key:
                    print(f"[GeminiNode] 使用config.json中的API密钥")
                    return config_api_key
                else:
                    print(f"[GeminiNode] config.json中未找到gemini_api_key")
                    return ''
        except Exception as e:
            print(f"[GeminiNode] 读取config.json失败: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(GEMINI_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "请详细描述这张图片的内容，不要做出评论或建议"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "process"
    CATEGORY = "🍎MYAPI"



    def process(self, api_key, model, prompt, max_tokens=4096, temperature=1.0, top_p=0.95, top_k=40, seed=0, image=None, image_2=None):
        """主处理函数"""

        # 获取实际使用的API密钥
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            return ("Error: 请输入API密钥或在config.json中配置gemini_api_key。请访问 https://aistudio.google.com/ 获取API密钥。",)

        try:
            # 检查google-genai是否可用
            from google import genai
            from google.genai import types
        except ImportError:
            return ("Error: 请安装google-genai: pip install google-genai",)
        
        try:
            print(f"Processing request with Gemini model: {model}")
            print(f"Image 1 provided: {image is not None}")
            print(f"Image 2 provided: {image_2 is not None}")
            print(f"Image 3 provided: {image_3 is not None}")
            print(f"Image 4 provided: {image_4 is not None}")
            print(f"Image 5 provided: {image_5 is not None}")
            print(f"Using seed: {seed}")
            
            # 初始化客户端
            client = genai.Client(api_key=actual_api_key)
            
            # 准备内容（按官方建议：单图时将文本放在图后；多图时文本放前）
            contents = []
            
            # 将图像转换为 Gemini Part（与官方示例一致）
            def image_to_part(img):
                try:
                    from PIL import Image
                    import numpy as np
                    import torch
                    from google.genai import types as _types
                    
                    # 张量 -> numpy
                    if isinstance(img, torch.Tensor):
                        if img.is_cuda:
                            img = img.cpu()
                        img = img.numpy()
                    
                    # 处理批维
                    if len(img.shape) == 4:
                        img = img[0]
                    
                    # 归一化像素
                    if img.dtype in [np.float32, np.float64] and img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    # numpy -> PNG bytes
                    pil_image = Image.fromarray(img.astype(np.uint8), 'RGB')
                    buf = io.BytesIO()
                    pil_image.save(buf, format='PNG')
                    image_bytes = buf.getvalue()
                    
                    # 构建 Part
                    return _types.Part.from_bytes(data=image_bytes, mime_type='image/png')
                except Exception as e:
                    raise Exception(f"Error converting image to part: {str(e)}")

            def bytes_to_image_tensor(image_bytes):
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

            # 处理最多两张图像（作为 Part）
            image_inputs = [image, image_2]
            parts = []
            for idx, img in enumerate(image_inputs, start=1):
                if img is not None:
                    try:
                        part = image_to_part(img)
                        parts.append(part)
                        print(f"Successfully converted image {idx} to Part")
                    except Exception as e:
                        return (f"Error processing image {idx}: {str(e)}",)

            # 依据是否单图/多图构造 contents 顺序
            if len(parts) >= 2:
                contents = [prompt] + parts
            elif len(parts) == 1:
                contents = [parts[0], prompt]
            else:
                contents = [prompt]

            # 生成配置
            is_stream_image_model = (model == "gemini-2.5-flash-image-preview")
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
                seed=seed if seed != 0 and seed <= 0x7fffffff else None,
                response_modalities=["IMAGE", "TEXT"] if is_stream_image_model else ["Text"],
                response_mime_type=None if is_stream_image_model else "text/plain"
            )

            # 根据模型选择流式或非流式调用
            print(f"Calling Gemini API with model: {model}")
            if is_stream_image_model:
                collected_images = []
                collected_texts = []
                try:
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=config,
                    ):
                        if (
                            getattr(chunk, 'candidates', None) is None
                            or chunk.candidates[0].content is None
                            or chunk.candidates[0].content.parts is None
                            or len(chunk.candidates[0].content.parts) == 0
                        ):
                            # 纯文本增量
                            if hasattr(chunk, 'text') and chunk.text:
                                collected_texts.append(chunk.text)
                            continue
                        part0 = chunk.candidates[0].content.parts[0]
                        inline_data = getattr(part0, 'inline_data', None)
                        if inline_data and getattr(inline_data, 'data', None):
                            try:
                                tensor_img = bytes_to_image_tensor(inline_data.data)
                                collected_images.append(tensor_img)
                                if len(collected_images) >= 5:
                                    # 收集最多5张
                                    pass
                            except Exception as e:
                                print(f"Error decoding streamed image: {str(e)}")
                        else:
                            if hasattr(chunk, 'text') and chunk.text:
                                collected_texts.append(chunk.text)
                except Exception as e:
                    print(f"Stream error: {str(e)}")
                    return (f"Error: 流式生成失败：{str(e)}",)

                text_out = "".join(collected_texts).strip()
                return (text_out if text_out else "",)
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
            
            # 检查响应结构并提取文本
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # 检查finish_reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    if finish_reason == 'MAX_TOKENS':
                        return ("Error: 响应因达到最大token限制而截断。请增加max_tokens值或简化提示词。",)
                    elif finish_reason == 'SAFETY':
                        return ("Error: 响应因安全原因被阻止。请修改提示词内容。",)
                    elif finish_reason == 'RECITATION':
                        return ("Error: 响应因重复内容被截断。",)
                
                # 优先从 finish_message 中提取
                if hasattr(candidate, 'finish_message') and candidate.finish_message:
                    fm = candidate.finish_message
                    try:
                        # finish_message 可能含有 content/parts/text
                        if hasattr(fm, 'text') and fm.text:
                            return (fm.text,)
                        if hasattr(fm, 'content') and fm.content:
                            fm_content = fm.content
                            if hasattr(fm_content, 'parts') and fm_content.parts:
                                for p in fm_content.parts:
                                    if hasattr(p, 'text') and p.text:
                                        return (p.text,)
                            if hasattr(fm_content, 'text') and fm_content.text:
                                return (fm_content.text,)
                    except Exception:
                        pass

                if hasattr(candidate, 'content'):
                    content = candidate.content
                    
                    if hasattr(content, 'parts') and content.parts is not None:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                return (part.text,)

                    if hasattr(content, 'text'):
                        return (content.text,)

                if hasattr(candidate, 'text'):
                    return (candidate.text,)
            
            # 尝试直接访问response.text
            if hasattr(response, 'text') and response.text:
                return (response.text,)

            # SDK 辅助方法兜底
            if hasattr(response, '_get_text'):
                try:
                    _t = response._get_text()
                    if _t:
                        return (_t,)
                except Exception:
                    pass
            
            # parsed 兜底
            try:
                if hasattr(response, 'parsed') and response.parsed:
                    parsed_val = response.parsed
                    if isinstance(parsed_val, str) and parsed_val.strip():
                        return (parsed_val,)
            except Exception:
                pass

            # JSON 兜底提取
            try:
                if hasattr(response, 'to_json_dict'):
                    jd = response.to_json_dict()
                    # 广度优先搜索所有'text'键
                    queue = [jd]
                    while queue:
                        cur = queue.pop(0)
                        if isinstance(cur, dict):
                            if 'text' in cur and isinstance(cur['text'], str) and cur['text'].strip():
                                return (cur['text'],)
                            queue.extend(cur.values())
                        elif isinstance(cur, list):
                            queue.extend(cur)
            except Exception:
                pass
            
            # 最终失败
            # 尝试给出 finish_reason 提示
            try:
                fr = None
                if hasattr(response, 'candidates') and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'finish_reason'):
                        fr = str(cand.finish_reason)
                if fr:
                    return (f"Error: 无法提取文本（finish_reason={fr}）。",)
            except Exception:
                pass
            return ("Error: 无法从响应中提取文本内容。",)

        except Exception as e:
            print(f"Unexpected error in Gemini process: {str(e)}")
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "GeminiAINode": GeminiAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAINode": "🌟Gemini AI"
}

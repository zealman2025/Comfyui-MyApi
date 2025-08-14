import os
import json

# 从配置文件加载模型配置
def load_gemini_models_from_config():
    """从config.json加载Gemini模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get('models', {})
            gemini_models = models.get('gemini', {})
            # 移除默认模型回退
            return gemini_models
    except Exception as e:
        print(f"Error loading Gemini models from config: {str(e)}")
        # 不再提供默认模型
        return {}

# 加载模型配置
GEMINI_MODELS = load_gemini_models_from_config()

class GeminiAINode:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.api_key = self._load_api_key()
        
    def _load_api_key(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('gemini_api_key', '')
        except Exception as e:
            print(f"Error loading Gemini API key: {str(e)}")
            return ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(GEMINI_MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": "请详细描述这张图片的内容，不要做出评论或建议"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🍎MYAPI"

    def process(self, model, prompt, max_tokens=1024, temperature=1.0, top_p=0.95, top_k=40, seed=0, image=None):
        """主处理函数"""
        
        if not self.api_key:
            return ("Error: 请在config.json中配置gemini_api_key。请访问 https://aistudio.google.com/ 获取API密钥。",)
        
        try:
            # 检查google-genai是否可用
            from google import genai
            from google.genai import types
        except ImportError:
            return ("Error: 请安装google-genai: pip install google-genai",)
        
        try:
            print(f"Processing request with Gemini model: {model}")
            print(f"Image provided: {image is not None}")
            print(f"Using seed: {seed}")
            
            # 初始化客户端
            client = genai.Client(api_key=self.api_key)
            
            # 准备内容
            contents = [prompt]
            
            # 简单的图像处理
            if image is not None:
                try:
                    from PIL import Image
                    import numpy as np
                    import torch
                    
                    # 简化处理，与minimal版本相同
                    if isinstance(image, torch.Tensor):
                        if image.is_cuda:
                            image = image.cpu()
                        image = image.numpy()
                    
                    if len(image.shape) == 4:
                        image = image[0]
                    
                    if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    
                    pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
                    contents.insert(0, pil_image)
                    print("Successfully added image to content")
                    
                except Exception as e:
                    return (f"Error processing image: {str(e)}",)

            # 生成配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
                seed=seed if seed != 0 and seed <= 0x7fffffff else None
            )

            # 调用API
            print(f"Calling Gemini API with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            return (response.text,)
            
        except Exception as e:
            print(f"Unexpected error in Gemini process: {str(e)}")
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "GeminiAINode": GeminiAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAINode": "🌟Gemini AI"
}
import os
import json
import traceback

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def load_deepseek_models_from_config():
    """从config.json加载DeepSeek模型配置"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        models = config.get("models", {})
        deepseek_models = models.get("deepseek", {})
        return deepseek_models if deepseek_models else {
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-reasoner": "DeepSeek Reasoner",
        }
    except Exception as e:
        print(f"[DeepSeekV32ExpNode] 加载配置失败: {str(e)}")
        traceback.print_exc()
        return {
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-reasoner": "DeepSeek Reasoner",
        }


DEEPSEEK_MODELS = load_deepseek_models_from_config()


class DeepSeekV32ExpNode:
    """DeepSeek V3.2 Experimental 聊天节点"""

    CATEGORY = "🍎MYAPI"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"

    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(DEEPSEEK_MODELS.keys()), {"default": "deepseek-chat"}),
                "prompt": ("STRING", {"multiline": True, "default": "你好，DeepSeek!"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "stream": ("BOOLEAN", {"default": False}),
            },
        }

    def _get_api_key(self, input_api_key: str) -> str:
        invalid_placeholders = {
            "YOUR_API_KEY",
            "你的apikey",
            "your_api_key_here",
            "请输入API密钥",
            "请输入你的API密钥",
            "",
        }

        if input_api_key and input_api_key.strip() and input_api_key.strip() not in invalid_placeholders:
            print("[DeepSeekV32ExpNode] 使用输入的API密钥")
            return input_api_key.strip()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config_key = config.get("deepseek_api_key", "").strip()
            if config_key:
                print("[DeepSeekV32ExpNode] 使用config.json中的API密钥")
            else:
                print("[DeepSeekV32ExpNode] config.json中未找到deepseek_api_key")
            return config_key
        except Exception as e:
            print(f"[DeepSeekV32ExpNode] 读取config.json失败: {str(e)}")
            traceback.print_exc()
            return ""

    def _check_dependencies(self):
        missing = []
        if not HAS_OPENAI:
            missing.append("openai")
        return missing

    def process(self, api_key, model, prompt, system_prompt="You are a helpful assistant.", temperature=1.0,
                max_tokens=2048, top_p=1.0, stream=False):
        missing = self._check_dependencies()
        if missing:
            return (f"Error: 请安装依赖 {', '.join(missing)}，例如运行 pip install openai",)

        actual_key = self._get_api_key(api_key)
        if not actual_key:
            return ("Error: 未找到有效的API密钥，请手动输入或在config.json中配置deepseek_api_key。",)

        try:
            client = OpenAI(api_key=actual_key, base_url="https://api.deepseek.com")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response_text = ""

            if stream:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                )

                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta:
                        delta_content = chunk.choices[0].delta.content or ""
                        response_text += delta_content
                return (response_text,)

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )

            if completion.choices:
                response_text = completion.choices[0].message.content or ""
            return (response_text,)

        except Exception as e:
            print(f"[DeepSeekV32ExpNode] 调用DeepSeek失败: {str(e)}")
            traceback.print_exc()
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "DeepSeekV32ExpNode": DeepSeekV32ExpNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekV32ExpNode": "🍭DeepSeek V3.2 Exp",
}


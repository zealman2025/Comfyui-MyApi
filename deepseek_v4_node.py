import traceback

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


DEEPSEEK_V4_MODELS = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
}


# 思考模式下，DeepSeek V4 仅 `high` / `max` 实际生效
# （low/medium 会被映射为 high，xhigh 会被映射为 max，
#  详见 https://api-docs.deepseek.com/zh-cn/guides/thinking_mode）
REASONING_EFFORT_OPTIONS = {
    "高（high）": "high",
    "最大（max）": "max",
}


class DeepSeekV4Node:
    """DeepSeek V4 聊天节点（支持思考模式）"""

    CATEGORY = "🍎MYAPI"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("string", "reasoning")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    list(DEEPSEEK_V4_MODELS.keys()),
                    {"default": "deepseek-v4-pro"},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "你好，DeepSeek!"}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "reasoning_effort": (
                    list(REASONING_EFFORT_OPTIONS.keys()),
                    {"default": "高（high）"},
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant."},
                ),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 65535}),
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

        if (
            input_api_key
            and input_api_key.strip()
            and input_api_key.strip() not in invalid_placeholders
        ):
            print("[DeepSeekV4Node] 使用节点中的 API 密钥")
            return input_api_key.strip()
        return ""

    def _check_dependencies(self):
        missing = []
        if not HAS_OPENAI:
            missing.append("openai")
        return missing

    def _resolve_reasoning_effort(self, reasoning_effort: str) -> str:
        resolved = REASONING_EFFORT_OPTIONS.get(reasoning_effort, reasoning_effort)
        return resolved

    def _extract_reasoning(self, message) -> str:
        """从 message 中提取 reasoning_content（如有）。"""
        if message is None:
            return ""

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return reasoning

        if isinstance(message, dict):
            return message.get("reasoning_content", "") or ""

        return ""

    def process(
        self,
        api_key,
        model,
        prompt,
        enable_thinking=False,
        reasoning_effort="高（high）",
        system_prompt="You are a helpful assistant.",
        max_tokens=4096,
        stream=False,
    ):
        missing = self._check_dependencies()
        if missing:
            return (
                f"Error: 请安装依赖 {', '.join(missing)}，例如运行 pip install openai",
                "",
            )

        actual_key = self._get_api_key(api_key)
        if not actual_key:
            return ("Error: 请在节点中填写有效的 DeepSeek API 密钥。", "")

        try:
            client = OpenAI(api_key=actual_key, base_url="https://api.deepseek.com")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            # 显式控制思考开关：默认 enabled，关闭时显式发送 disabled
            # 思考模式下 reasoning_effort 才有意义；非思考模式下不发送，
            # 避免与 thinking.disabled 同时设置引起歧义。
            if enable_thinking:
                actual_reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
                request_kwargs["reasoning_effort"] = actual_reasoning_effort
                request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                print(
                    f"[DeepSeekV4Node] thinking=enabled, reasoning_effort={actual_reasoning_effort}"
                )
            else:
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                print("[DeepSeekV4Node] thinking=disabled")

            print(f"[DeepSeekV4Node] Calling DeepSeek API with model: {model}")

            response_text = ""
            reasoning_text = ""

            if stream:
                completion = client.chat.completions.create(stream=True, **request_kwargs)

                for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    delta_reasoning = getattr(delta, "reasoning_content", None) or ""
                    if delta_reasoning:
                        reasoning_text += delta_reasoning
                    delta_content = getattr(delta, "content", None) or ""
                    if delta_content:
                        response_text += delta_content
                return (response_text, reasoning_text)

            completion = client.chat.completions.create(stream=False, **request_kwargs)

            if completion.choices:
                message = completion.choices[0].message
                response_text = (getattr(message, "content", None) or "")
                reasoning_text = self._extract_reasoning(message)
            return (response_text, reasoning_text)

        except Exception as e:
            print(f"[DeepSeekV4Node] 调用 DeepSeek 失败: {str(e)}")
            traceback.print_exc()
            return (f"Error: {str(e)}", "")


NODE_CLASS_MAPPINGS = {
    "DeepSeekV4Node": DeepSeekV4Node,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekV4Node": "🔎DeepSeek V4",
}

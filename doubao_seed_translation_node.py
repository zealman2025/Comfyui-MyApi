import json
import traceback

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not found. Please install it with: pip install requests")


DOUBAO_TRANSLATION_MODELS = {
    "doubao-seed-translation-250915": "豆包Seed翻译模型",
}


class DoubaoSeedTranslationNode:
    """豆包 Seed 翻译模型节点"""
    
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
            print("[DoubaoSeedTranslationNode] 使用节点中的 API 密钥")
            return input_api_key.strip()
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        # 语言代码选项
        language_options = [
            "zh",  # 中文
            "en",  # 英语
            "ja",  # 日语
            "ko",  # 韩语
            "fr",  # 法语
            "de",  # 德语
            "es",  # 西班牙语
            "it",  # 意大利语
            "pt",  # 葡萄牙语
            "ru",  # 俄语
            "ar",  # 阿拉伯语
            "th",  # 泰语
            "vi",  # 越南语
            "id",  # 印尼语
            "hi",  # 印地语
            "tr",  # 土耳其语
            "pl",  # 波兰语
            "nl",  # 荷兰语
            "cs",  # 捷克语
            "sv",  # 瑞典语
            "da",  # 丹麦语
            "fi",  # 芬兰语
            "no",  # 挪威语
            "hu",  # 匈牙利语
            "ro",  # 罗马尼亚语
            "el",  # 希腊语
            "he",  # 希伯来语
            "uk",  # 乌克兰语
            "bg",  # 保加利亚语
            "hr",  # 克罗地亚语
            "sk",  # 斯洛伐克语
            "sl",  # 斯洛文尼亚语
            "et",  # 爱沙尼亚语
            "lv",  # 拉脱维亚语
            "lt",  # 立陶宛语
            "mt",  # 马耳他语
            "ga",  # 爱尔兰语
            "cy",  # 威尔士语
        ]

        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (list(DOUBAO_TRANSLATION_MODELS.keys()), {"default": "doubao-seed-translation-250915"}),
                "text": ("STRING", {"multiline": True, "default": "若夫淫雨霏霏，连月不开，阴风怒号，浊浪排空"}),
                "source_language": (language_options, {"default": "zh"}),
                "target_language": (language_options, {"default": "en"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "translate"
    CATEGORY = "🍎MYAPI"

    def _check_dependencies(self):
        """检查必要的依赖是否已安装"""
        missing_deps = []
        if not HAS_REQUESTS:
            missing_deps.append("requests")
        return missing_deps

    def translate(self, api_key, model, text, source_language, target_language):
        """翻译函数"""
        missing_deps = self._check_dependencies()
        if missing_deps:
            return (f"Error: 缺少必要的依赖: {', '.join(missing_deps)}. 请安装这些依赖后再试。",)

        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            return ("Error: 请在节点中填写豆包 API 密钥。请访问 https://console.volcengine.com/ark 获取。",)

        if not text or not text.strip():
            return ("Error: 请输入要翻译的文本。",)

        if source_language == target_language:
            return ("Error: 源语言和目标语言不能相同。",)

        try:
            print(f"[DoubaoSeedTranslationNode] 翻译请求:")
            print(f"  模型: {model}")
            print(f"  源语言: {source_language}")
            print(f"  目标语言: {target_language}")
            print(f"  文本长度: {len(text)} 字符")

            url = "https://ark.cn-beijing.volces.com/api/v3/responses"
            headers = {
                "Authorization": f"Bearer {actual_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text,
                                "translation_options": {
                                    "source_language": source_language,
                                    "target_language": target_language
                                }
                            }
                        ]
                    }
                ]
            }

            print(f"[DoubaoSeedTranslationNode] 发送请求到: {url}")
            # 禁用代理，因为豆包是国内服务，通常不需要代理
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=60,
                proxies={"http": None, "https": None}  # 禁用代理
            )
            
            print(f"[DoubaoSeedTranslationNode] 响应状态码: {response.status_code}")

            if not response.ok:
                try:
                    err_json = response.json()
                    err_message = err_json.get('error', {}).get('message', response.text)
                except Exception:
                    err_message = response.text
                
                if response.status_code == 401:
                    return ("Error: 身份验证失败(401)。请确认节点中填写的 API 密钥正确且未含多余空格。",)
                return (f"Error: {response.status_code} - {err_message}",)

            result = response.json()
            print(f"[DoubaoSeedTranslationNode] 响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")

            # 解析响应，根据API文档，响应格式可能不同
            translated_text = ""
            if "output" in result:
                output = result["output"]
                if isinstance(output, list) and len(output) > 0:
                    first_output = output[0]
                    if "choices" in first_output:
                        choices = first_output["choices"]
                        if isinstance(choices, list) and len(choices) > 0:
                            translated_text = choices[0].get("message", {}).get("content", "")
                    elif "content" in first_output:
                        # 直接包含content字段
                        content = first_output["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text_item = content[0]
                            if isinstance(text_item, dict) and "text" in text_item:
                                translated_text = text_item["text"]
                            elif isinstance(text_item, str):
                                translated_text = text_item
                    elif "text" in first_output:
                        translated_text = first_output["text"]
            elif "choices" in result:
                choices = result["choices"]
                if isinstance(choices, list) and len(choices) > 0:
                    translated_text = choices[0].get("message", {}).get("content", "")
            elif "text" in result:
                translated_text = result["text"]
            elif "content" in result:
                translated_text = result["content"]

            if not translated_text:
                # 如果无法解析，返回原始响应供调试
                return (f"Error: 无法解析响应。原始响应: {json.dumps(result, ensure_ascii=False)}",)

            print(f"[DoubaoSeedTranslationNode] 翻译完成，结果长度: {len(translated_text)} 字符")
            return (translated_text,)

        except requests.exceptions.Timeout:
            return ("Error: 请求超时，请稍后重试。",)
        except requests.exceptions.RequestException as e:
            return (f"Error: 网络请求失败: {str(e)}",)
        except Exception as e:
            print(f"[DoubaoSeedTranslationNode] 未预期的错误: {str(e)}")
            print(traceback.format_exc())
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "DoubaoSeedTranslationNode": DoubaoSeedTranslationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedTranslationNode": "🥟豆包翻译模型"
}


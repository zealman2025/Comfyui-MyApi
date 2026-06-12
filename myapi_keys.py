"""
MYAPI 全局 API Key 存储与解析（ComfyUI 设置 + 节点开关）。
"""

import json
import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

INVALID_API_KEY_PLACEHOLDERS = frozenset(
    {
        "YOUR_API_KEY",
        "你的apikey",
        "your_api_key_here",
        "请输入API密钥",
        "请输入你的API密钥",
        "",
    }
)

PROVIDER_DOUBAO = "doubao"
PROVIDER_AUTODL = "autodl"
PROVIDER_BIZYAIR = "bizyair"
PROVIDER_GEEKNOW = "geeknow"

PROVIDERS = frozenset({PROVIDER_DOUBAO, PROVIDER_AUTODL, PROVIDER_BIZYAIR, PROVIDER_GEEKNOW})

PROVIDER_LABELS = {
    PROVIDER_DOUBAO: "豆包",
    PROVIDER_AUTODL: "AutoDL",
    PROVIDER_BIZYAIR: "BizyAir",
    PROVIDER_GEEKNOW: "Geeknow",
}

USE_NODE_API_KEY_INPUT = (
    "BOOLEAN",
    {
        "default": False,
        "label_on": "使用输入框中的apikey",
        "label_off": "使用系统设置",
    },
)


def normalize_api_key(key: str) -> str:
    if not key:
        return ""
    k = str(key).strip()
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        k = k[1:-1].strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k


def clean_input_api_key(input_api_key: str) -> str:
    key = normalize_api_key(input_api_key or "")
    if not key or key in INVALID_API_KEY_PLACEHOLDERS:
        return ""
    return key


def _comfy_settings_path():
    if folder_paths is None:
        return None
    try:
        user_dir = folder_paths.get_user_directory()
        if user_dir and os.path.isdir(user_dir):
            return os.path.join(user_dir, "default", "comfy.settings.json")
    except Exception:
        pass
    return None


def _read_comfy_setting(setting_id: str) -> str:
    """兜底：从 ComfyUI 原生设置文件读取（前端 text 设置会写入这里）。"""
    path = _comfy_settings_path()
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return str(data.get(setting_id, "") or "")
    except Exception:
        pass
    return ""


def _default_store_path() -> str:
    if folder_paths is not None:
        try:
            user_dir = folder_paths.get_user_directory()
            if user_dir and os.path.isdir(user_dir):
                base = os.path.join(user_dir, "default", "myapi")
                os.makedirs(base, exist_ok=True)
                return os.path.join(base, "api_keys.json")
        except Exception:
            pass
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    fallback = os.path.join(plugin_dir, "user_default_myapi")
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, "api_keys.json")


class ApiKeyStore:
    def __init__(self, path=None):
        self.path = path or _default_store_path()

    def _load_raw(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _atomic_write(self, data: dict) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception as exc:
            print(f"[Comfyui-MyApi] 保存 API Key 失败: {exc}")
            return False

    @staticmethod
    def mask_api_key(api_key: str) -> str:
        if not api_key:
            return ""
        if len(api_key) < 8:
            return "***"
        return f"{api_key[:4]}***{api_key[-4:]}"

    def get(self, provider: str) -> str:
        if provider not in PROVIDERS:
            return ""
        key = clean_input_api_key(self._load_raw().get(provider, ""))
        if key:
            return key
        # 兜底：从 comfy.settings.json 读取（原生 text 设置 id：MyAPI.ApiKey.<provider>）
        return clean_input_api_key(_read_comfy_setting(f"MyAPI.ApiKey.{provider}"))

    def set(self, provider: str, api_key: str) -> bool:
        if provider not in PROVIDERS:
            return False
        data = self._load_raw()
        key = clean_input_api_key(api_key)
        if not key:
            return False
        data[provider] = key
        return self._atomic_write(data)

    def get_masked_all(self) -> dict:
        raw = self._load_raw()
        result = {}
        for provider in sorted(PROVIDERS):
            key = clean_input_api_key(raw.get(provider, ""))
            result[provider] = {
                "exists": bool(key),
                "masked": self.mask_api_key(key) if key else "",
            }
        return result


api_key_store = ApiKeyStore()


def missing_api_key_message(provider: str) -> str:
    label = PROVIDER_LABELS.get(provider, provider)
    return (
        f"请在 ComfyUI 设置 → 🍎MYAPI → API密钥 中配置 {label} API Key，"
        "或在节点中开启「使用输入框中的apikey」后填写。"
    )


def resolve_api_key(provider: str, input_api_key: str, use_node_api_key: bool) -> str:
    """
    use_node_api_key=True：使用节点输入框中的 Key。
    use_node_api_key=False（默认）：使用系统设置中的 Key；
        若系统设置为空但节点输入框已填（多见于旧工作流升级），
        则自动回退使用节点输入框中的 Key，保证旧工作流无需手动改动即可继续运行。
    """
    node_key = clean_input_api_key(input_api_key)
    if use_node_api_key:
        return node_key
    system_key = api_key_store.get(provider)
    if system_key:
        return system_key
    return node_key

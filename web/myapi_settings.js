/**
 * ComfyUI 设置面板：MYAPI 全局 API Key（豆包 / AutoDL / Geeknow）
 *
 * 使用 ComfyUI 原生 text 设置项：
 *   - 自动套用默认 CSS、4 个条目均正常显示
 *   - 值随 comfy.settings.json 持久化（id 即 MyAPI.ApiKey.<provider>）
 *   - onChange 时同步写入 user/default/myapi/api_keys.json，供 Python 后端读取
 */
import { app } from "../../../scripts/app.js";

const API_KEYS_URL = "/myapi/settings/api_keys";

const PROVIDER_DEFS = [
    { provider: "doubao", label: "豆包 API Key" },
    { provider: "autodl", label: "AutoDL API Key" },
    { provider: "geeknow", label: "Geeknow API Key" },
];

async function saveKey(provider, apiKey) {
    const val = (apiKey || "").trim();
    if (!val) return;
    try {
        await fetch(API_KEYS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider, api_key: val }),
        });
    } catch (_) {
        /* ignore */
    }
}

const settings = PROVIDER_DEFS.map((p) => ({
    id: `MyAPI.ApiKey.${p.provider}`,
    name: p.label,
    category: ["🍎MYAPI", "API密钥", p.label],
    type: "text",
    defaultValue: "",
    tooltip:
        "保存到 user/default/myapi/api_keys.json，供 MYAPI 节点在「使用系统设置」模式下读取",
    onChange: (value) => saveKey(p.provider, value),
}));

app.registerExtension({
    name: "Comfyui-MyApi.Settings",
    settings,
});

/**
 * 旧工作流兼容：节点新增了 use_node_api_key 开关（位于 api_key 之后，index 1）。
 *
 * ComfyUI 的 widgets_values 按位置序列化，旧工作流缺少该开关会导致后续 widget 整体错位。
 * 这里在节点 configure 之前检测：若保存的 widget 值比当前 widget 数恰好少 1，
 * 且 index 1 不是布尔（说明是旧版数据），则在 index 1 补回默认值 false（=使用系统设置）。
 */
import { app } from "../../../scripts/app.js";

const TOGGLE_INDEX = 1; // api_key 始终是第一个 widget，开关紧随其后

app.registerExtension({
    name: "Comfyui-MyApi.LegacyMigrate",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const required = nodeData?.input?.required;
        if (!required) return;
        const keys = Object.keys(required);
        // 仅处理「第一个是 api_key、第二个是 use_node_api_key」的 MYAPI 节点
        if (keys[0] !== "api_key" || keys[1] !== "use_node_api_key") return;

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            try {
                const wv = info && info.widgets_values;
                if (
                    Array.isArray(wv) &&
                    this.widgets &&
                    this.widgets.length &&
                    wv.length === this.widgets.length - 1 &&
                    typeof wv[TOGGLE_INDEX] !== "boolean"
                ) {
                    // 旧工作流：补回开关默认值，避免后续 widget 错位
                    wv.splice(TOGGLE_INDEX, 0, false);
                }
            } catch (_) {
                /* ignore */
            }
            return origConfigure ? origConfigure.apply(this, arguments) : undefined;
        };
    },
});

import { app } from "../../scripts/app.js";

const NODE_CONFIGS = {
    BizyAirGPTImage2I2INode: {
        countWidget: "inputcount",
        maxCount: 4,
        minCount: 1,
    },
    BizyAirNanoBananaProNode: {
        countWidget: "inputcount",
        maxCount: 6,
        minCount: 1,
    },
};

function getInputName(index) {
    // 与 Python 节点的 INPUT_TYPES 命名保持一致：
    //   index 1 -> "image"
    //   index 2 -> "image2"
    //   ...
    return index === 1 ? "image" : `image${index}`;
}

function findInputIndex(node, name) {
    if (!node.inputs) return -1;
    for (let i = 0; i < node.inputs.length; i += 1) {
        if (node.inputs[i].name === name) {
            return i;
        }
    }
    return -1;
}

function clampCount(value, config) {
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return config.minCount;
    return Math.max(config.minCount, Math.min(parsed, config.maxCount));
}

function syncImageInputs(node, config) {
    const widget = node.widgets?.find((w) => w.name === config.countWidget);
    if (!widget) return;

    const desired = clampCount(widget.value, config);

    // 删除多余的 image 输入端口（保留已连线情况下用户主动操作时的安全性，
    // 这里只清理超出 desired 的位置）。
    for (let index = config.maxCount; index > desired; index -= 1) {
        const name = getInputName(index);
        const slot = findInputIndex(node, name);
        if (slot !== -1) {
            node.removeInput(slot);
        }
    }

    // 补齐缺失的 image 输入端口
    for (let index = 1; index <= desired; index += 1) {
        const name = getInputName(index);
        if (findInputIndex(node, name) === -1) {
            node.addInput(name, "IMAGE");
        }
    }

    if (typeof node.computeSize === "function") {
        const size = node.computeSize();
        if (Array.isArray(size) && size.length >= 2) {
            node.size[0] = Math.max(node.size[0] || 0, size[0]);
            node.size[1] = Math.max(node.size[1] || 0, size[1]);
        }
    }

    if (typeof node.setDirtyCanvas === "function") {
        node.setDirtyCanvas(true, true);
    } else {
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

function attachDynamicInputs(node, config) {
    const widget = node.widgets?.find((w) => w.name === config.countWidget);
    if (!widget) return;

    // 让 inputcount 数值变化时自动同步端口
    const originalCallback = widget.callback;
    widget.callback = function patchedCallback(value, ...args) {
        if (typeof originalCallback === "function") {
            originalCallback.call(this, value, ...args);
        }
        syncImageInputs(node, config);
    };

    // 添加"更新图片输入"按钮
    const buttonName = "更新图片输入";
    const alreadyHasButton = node.widgets?.some(
        (w) => w.name === buttonName && w.type === "button"
    );
    if (!alreadyHasButton) {
        node.addWidget("button", buttonName, null, () => {
            syncImageInputs(node, config);
        });
    }
}

app.registerExtension({
    name: "Comfyui-MyApi.DynamicImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData.name];
        if (!config) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated(...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            attachDynamicInputs(this, config);
            // 节点新建时按 inputcount 默认值初始化端口
            setTimeout(() => syncImageInputs(this, config), 0);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure(info, ...args) {
            const result = originalOnConfigure?.apply(this, [info, ...args]);
            // 工作流加载完成后，根据保存的 inputcount 同步一次端口，
            // 防止旧工作流加载后多出空闲端口。
            setTimeout(() => syncImageInputs(this, config), 0);
            return result;
        };
    },
});

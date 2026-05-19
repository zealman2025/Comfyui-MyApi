import { app } from "../../scripts/app.js";

const AUTODL_DETAIL_API = "https://www.autodl.art/api/v1/mllm/model/detail";

// 节点用到的模型；优先从 AutoDL 市场 API 拉取，失败时用本地快照兜底
const AUTODL_MODEL_IDS = {
    "Qwen3.5-397B-A17B": 4,
    "Kimi-K2.5": 6,
    "gpt-5.4": 13,
    "gpt-5.4-nano": 14,
    "gpt-5.4-mini": 15,
    "gemini-3.1-pro-preview": 17,
    "qwen3.6-plus": 19,
    "nano-banana-2": 22,
    "Kimi-K2.6": 39,
    "gpt-image-2": 40,
    "gpt-5.5": 44,
};

// 价目快照（内部单位 ÷1000 = 显示人民币）；始终覆盖 API 返回值
const AUTODL_MODEL_FALLBACK = {
    "nano-banana-2": {
        price_type: "universal_token",
        price_config: {
            universal_token: { input_text_price: 2625, output_image_price: 315000 },
        },
    },
    "gpt-image-2": {
        price_type: "simple_token",
        price_config: { simple_token: { input_price: 28000, output_price: 168000 } },
    },
    "qwen3.6-plus": {
        price_type: "context_tiered_token",
        price_config: {
            context_tiered_token: [{ input_price: 1600, output_price: 9600 }],
        },
    },
    "Qwen3.5-397B-A17B": {
        price_type: "context_tiered_token",
        price_config: {
            context_tiered_token: [{ input_price: 720, output_price: 4320 }],
        },
    },
    "Kimi-K2.5": {
        price_type: "simple_token",
        price_config: { simple_token: { input_price: 2400, output_price: 12600 } },
    },
    "Kimi-K2.6": {
        price_type: "simple_token",
        price_config: { simple_token: { input_price: 3900, output_price: 16200 } },
    },
    "gpt-5.4-nano": {
        price_type: "simple_token",
        price_config: { simple_token: { input_price: 630, output_price: 3938 } },
    },
    "gpt-5.4-mini": {
        price_type: "simple_token",
        price_config: { simple_token: { input_price: 2363, output_price: 14175 } },
    },
    "gpt-5.4": {
        price_type: "context_tiered_token",
        price_config: {
            context_tiered_token: [{ input_price: 7875, output_price: 47250 }],
        },
    },
    "gpt-5.5": {
        price_type: "context_tiered_token",
        price_config: {
            context_tiered_token: [{ input_price: 17500, output_price: 105000 }],
        },
    },
    "gemini-3.1-pro-preview": {
        price_type: "context_tiered_token",
        price_config: {
            context_tiered_token: [{ input_price: 10500, output_price: 63000 }],
        },
    },
};

const autodlModelCatalog = { ...AUTODL_MODEL_FALLBACK };

class SimpleBadge {
    constructor({
        text,
        fgColor = "#FFFFFF",
        bgColor = "#059669",
        fontSize = 10,
        padding = 6,
        height = 18,
        cornerRadius = 9,
    }) {
        this.text = text;
        this.fgColor = fgColor;
        this.bgColor = bgColor;
        this.fontSize = fontSize;
        this.padding = padding;
        this.height = height;
        this.cornerRadius = cornerRadius;
    }

    get visible() {
        return (this.text?.length ?? 0) > 0;
    }

    getWidth(ctx) {
        if (!this.visible) return 0;
        const { font } = ctx;
        ctx.font = `${this.fontSize}px sans-serif`;
        const width = ctx.measureText(this.text).width + this.padding * 2;
        ctx.font = font;
        return width;
    }

    draw(ctx, x, y) {
        if (!this.visible) return;

        const { font, fillStyle, textBaseline, textAlign } = ctx;
        ctx.font = `${this.fontSize}px sans-serif`;
        const width = this.getWidth(ctx);

        ctx.fillStyle = this.bgColor;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, width, this.height, this.cornerRadius);
        } else {
            ctx.rect(x, y, width, this.height);
        }
        ctx.fill();

        ctx.fillStyle = this.fgColor;
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        ctx.fillText(this.text, x + this.padding, y + this.height / 2 + 1);

        ctx.font = font;
        ctx.fillStyle = fillStyle;
        ctx.textBaseline = textBaseline;
        ctx.textAlign = textAlign;
    }
}

function formatAutodlYuan(raw) {
    if (raw == null || Number.isNaN(Number(raw))) return null;
    const value = Number(raw);
    let rounded = parseInt(String(value), 10);
    if ((value * 10) % 10 >= 5) {
        rounded += 1;
    }
    let text = (rounded / 1000).toFixed(3);
    text = text.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
    return text;
}

function getPrimaryTokenPrices(model) {
    if (!model?.price_config) return null;
    const type = model.price_type;
    const cfg = model.price_config;
    if (type === "simple_token") return cfg.simple_token;
    if (type === "context_tiered_token") {
        const tiers = cfg.context_tiered_token;
        return Array.isArray(tiers) && tiers.length ? tiers[0] : null;
    }
    if (type === "universal_token") return cfg.universal_token;
    return null;
}

function formatChatPriceBadge(model) {
    if (!model) return "ADL 价目加载中";
    const prices = getPrimaryTokenPrices(model);
    if (!prices) return "ADL 见市场价";

    const input = formatAutodlYuan(prices.input_price ?? prices.input_text_price);
    const output = formatAutodlYuan(
        prices.output_price ?? prices.output_text_price ?? prices.output_image_price
    );

    if (input && output) return `ADL 入¥${input} 出¥${output}/M`;
    if (output) return `ADL ¥${output}/M`;
    if (input) return `ADL 入¥${input}/M`;
    return "ADL 见市场价";
}

function formatImagePriceBadge(model) {
    if (!model) return "ADL 价目加载中";
    const universal = model.price_config?.universal_token;
    if (universal?.output_image_price != null) {
        const output = formatAutodlYuan(universal.output_image_price);
        if (output) return `ADL 出图¥${output}/M`;
    }
    return formatChatPriceBadge(model);
}

async function fetchAutodlModelDetail(id) {
    const response = await fetch(AUTODL_DETAIL_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (payload?.code !== "Success" || !payload?.data?.model_name) {
        throw new Error(payload?.msg || "Invalid model detail response");
    }
    return payload.data;
}

async function refreshAutodlModelCatalog() {
    const entries = await Promise.all(
        Object.entries(AUTODL_MODEL_IDS).map(async ([modelName, id]) => {
            try {
                const detail = await fetchAutodlModelDetail(id);
                return [modelName, detail];
            } catch (error) {
                console.warn(
                    `[Comfyui-MyApi] AutoDL 价目拉取失败 (${modelName}, id=${id}):`,
                    error
                );
                return [modelName, AUTODL_MODEL_FALLBACK[modelName] ?? null];
            }
        })
    );

    for (const [modelName, detail] of entries) {
        if (detail) {
            autodlModelCatalog[modelName] = detail;
        }
    }

    // 本地价目快照优先，确保与 AutoDL 模型广场展示一致
    for (const [modelName, fallback] of Object.entries(AUTODL_MODEL_FALLBACK)) {
        autodlModelCatalog[modelName] = fallback;
    }

    app.graph?.setDirtyCanvas?.(true, true);
}

function watchWidget(node, widgetName) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (!widget || widget._myapiAutodlPriceBadgeWatch) return;

    widget._myapiAutodlPriceBadgeWatch = true;
    const originalCallback = widget.callback;
    widget.callback = function patchedAutodlPriceBadgeCallback(value, ...args) {
        if (typeof originalCallback === "function") {
            originalCallback.call(this, value, ...args);
        }
        app.graph?.setDirtyCanvas?.(true, true);
    };
}

function attachAutodlPriceBadge(node, config) {
    if (node._myapiAutodlPriceBadgeAttached) return;
    node._myapiAutodlPriceBadgeAttached = true;

    node.badgePosition = "top-right";
    node.badges = node.badges ?? [];

    node.badges.push(() => {
        const modelName =
            typeof config.getModelName === "function"
                ? config.getModelName(node)
                : config.modelName;
        const model = autodlModelCatalog[modelName];
        const text =
            typeof config.formatBadge === "function"
                ? config.formatBadge(model)
                : formatChatPriceBadge(model);
        return new SimpleBadge({ text });
    });

    for (const widgetName of config.watchWidgets ?? []) {
        watchWidget(node, widgetName);
    }
}

const AUTODL_NODE_PRICES = {
    AutodlApiNode: {
        getModelName(node) {
            return node.widgets?.find((w) => w.name === "model")?.value;
        },
        formatBadge: formatChatPriceBadge,
        watchWidgets: ["model"],
    },
    AutodlNanoBanana2T2INode: {
        modelName: "nano-banana-2",
        formatBadge: formatChatPriceBadge,
        watchWidgets: [],
    },
    AutodlNanoBanana2I2INode: {
        modelName: "nano-banana-2",
        formatBadge: formatChatPriceBadge,
        watchWidgets: [],
    },
    AutodlGPTImage2T2INode: {
        modelName: "gpt-image-2",
        formatBadge: formatChatPriceBadge,
        watchWidgets: [],
    },
    AutodlGPTImage2I2INode: {
        modelName: "gpt-image-2",
        formatBadge: formatChatPriceBadge,
        watchWidgets: [],
    },
};

refreshAutodlModelCatalog().catch((error) => {
    console.warn("[Comfyui-MyApi] AutoDL 价目初始化失败，使用本地快照:", error);
});

app.registerExtension({
    name: "Comfyui-MyApi.AutodlPriceBadges",
    nodeCreated(node) {
        const config = AUTODL_NODE_PRICES[node.constructor?.comfyClass];
        if (!config) return;
        attachAutodlPriceBadge(node, config);
    },
});

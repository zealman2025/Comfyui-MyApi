import { app } from "../../scripts/app.js";

// Geeknow 各模型固定单价（元/次）
const GEEKNOW_MODEL_PRICES = {
    "gpt-image-2": 0.04,
    "gpt-image-2-pro": 0.08,
    "gemini-3-pro-image-preview": 0.22,
    "gemini-3.1-flash-image-preview": 0.15,
    "gemini-2.5-flash-image-preview": 0.06,
};

class SimpleBadge {
    constructor({
        text,
        fgColor = "#FFFFFF",
        bgColor = "#7C3AED",
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

function formatGeeknowBadge(node) {
    const model = node.widgets?.find((w) => w.name === "model")?.value;
    const price = GEEKNOW_MODEL_PRICES[model];
    if (price == null) return "GK 见平台价";
    return `GK ¥${price.toFixed(2)}/次`;
}

function watchWidget(node, widgetName) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (!widget || widget._myapiGeeknowPriceBadgeWatch) return;

    widget._myapiGeeknowPriceBadgeWatch = true;
    const originalCallback = widget.callback;
    widget.callback = function patchedGeeknowPriceBadgeCallback(value, ...args) {
        if (typeof originalCallback === "function") {
            originalCallback.call(this, value, ...args);
        }
        app.graph?.setDirtyCanvas?.(true, true);
    };
}

function attachGeeknowPriceBadge(node) {
    if (node._myapiGeeknowPriceBadgeAttached) return;
    node._myapiGeeknowPriceBadgeAttached = true;

    node.badgePosition = "top-right";
    node.badges = node.badges ?? [];

    node.badges.push(() => new SimpleBadge({ text: formatGeeknowBadge(node) }));

    watchWidget(node, "model");
}

const GEEKNOW_NODE_CLASSES = new Set([
    "GeeknowGPTImage2T2INode",
    "GeeknowGPTImage2I2INode",
    "GeeknowGeminiImageT2INode",
    "GeeknowGeminiImageI2INode",
]);

app.registerExtension({
    name: "Comfyui-MyApi.GeeknowPriceBadges",
    nodeCreated(node) {
        if (!GEEKNOW_NODE_CLASSES.has(node.constructor?.comfyClass)) return;
        attachGeeknowPriceBadge(node);
    },
});

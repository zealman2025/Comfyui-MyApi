import { app } from "../../scripts/app.js";

const GPT_IMAGE2_PRICE = 100;

const NANOBANANA2_PRICES = {
    "1K": 200,
    "2K": 200,
    "4K": 250,
};

const NANOBANANA2_OFFICIAL_RATES = {
    "0.5K": 550,
    "1K": 550,
    "2K": 850,
    "4K": 1100,
};

const GPT_IMAGE2_OFFICIAL_RATES = {
    "1K": { low: 161, medium: 378, high: 1120 },
    "2K": { low: 182, medium: 630, high: 2149 },
    "4K": { low: 224, medium: 966, high: 3486 },
};

function nanobananaThirdPartyPrice(node) {
    const resolution =
        node.widgets?.find((w) => w.name === "resolution")?.value ?? "1K";
    return NANOBANANA2_PRICES[resolution] ?? 200;
}

function nanobananaOfficialPrice(node) {
    const resolution =
        node.widgets?.find((w) => w.name === "resolution")?.value ?? "1K";
    return NANOBANANA2_OFFICIAL_RATES[resolution] ?? 550;
}

function gptImage2OfficialPrice(node) {
    const resolution =
        node.widgets?.find((w) => w.name === "resolution")?.value ?? "2K";
    const quality =
        node.widgets?.find((w) => w.name === "quality")?.value ?? "medium";
    return GPT_IMAGE2_OFFICIAL_RATES[resolution]?.[quality] ?? 630;
}

const BIZYAIR_NODE_PRICES = {
    BizyAirNanoBanana2ThirdPartyT2INode: { getPrice: nanobananaThirdPartyPrice },
    BizyAirNanoBanana2ThirdPartyI2INode: { getPrice: nanobananaThirdPartyPrice },
    BizyAirNanoBanana2OfficialT2INode: { getPrice: nanobananaOfficialPrice },
    BizyAirNanoBanana2OfficialI2INode: { getPrice: nanobananaOfficialPrice },
    BizyAirGPTImage2ThirdPartyT2INode: { price: GPT_IMAGE2_PRICE },
    BizyAirGPTImage2ThirdPartyI2INode: { price: GPT_IMAGE2_PRICE },
    BizyAirGPTImage2OfficialT2INode: { getPrice: gptImage2OfficialPrice },
    BizyAirGPTImage2OfficialI2INode: { getPrice: gptImage2OfficialPrice },
};

class SimpleBadge {
    constructor({
        text,
        fgColor = "#FFFFFF",
        bgColor = "#2563EB",
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

function formatPriceBadge(price) {
    if (price == null || price === "") return "";
    return `BZ 付费金币 ${price}/次`;
}

function watchWidget(node, widgetName) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (!widget || widget._myapiPriceBadgeWatch) return;

    widget._myapiPriceBadgeWatch = true;
    const originalCallback = widget.callback;
    widget.callback = function patchedPriceBadgeCallback(value, ...args) {
        if (typeof originalCallback === "function") {
            originalCallback.call(this, value, ...args);
        }
        app.graph?.setDirtyCanvas?.(true, true);
    };
}

function attachPriceBadge(node, config) {
    if (node._myapiPriceBadgeAttached) return;
    node._myapiPriceBadgeAttached = true;

    node.badgePosition = "top-right";
    node.badges = node.badges ?? [];

    node.badges.push(() => {
        const price =
            typeof config.getPrice === "function"
                ? config.getPrice(node)
                : config.price;
        return new SimpleBadge({
            text: formatPriceBadge(price),
        });
    });

    watchWidget(node, "resolution");
    watchWidget(node, "quality");
    watchWidget(node, "inputcount");
}

app.registerExtension({
    name: "Comfyui-MyApi.BizyAirPriceBadges",
    nodeCreated(node) {
        const config = BIZYAIR_NODE_PRICES[node.constructor?.comfyClass];
        if (!config) return;
        attachPriceBadge(node, config);
    },
});

# 🍎 ComfyUI MyAPI - 多模态 AI 节点集合

[![Version](https://img.shields.io/badge/version-2.4.2-blue.svg)](https://github.com/zealman2025/Comfyui-MyApi/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个面向 ComfyUI 的多模态 AI 节点集合，集成豆包、DeepSeek、BizyAir、AutoDL、Geeknow 等服务，覆盖文本生成、视觉理解、图像生成、图像编辑、翻译、文本处理等常见场景。所有节点都遵循统一的密钥与输入输出规范，便于在工作流中混搭使用。

## 主要特性

- 🔑 **节点内填写密钥**：每个节点的 `api_key` 输入框中独立填写，不读取本地配置文件
- 🖼️ **多模态支持**：覆盖文本、图像理解、图像生成、图像编辑、翻译等场景
- 🎯 **多服务集成**：豆包、DeepSeek、BizyAir、AutoDL、Geeknow 等主流 AI 服务一站式接入
- 🧩 **统一输出规范**：所有字符串输出端口统一命名为 `string`，便于上下游连接
- 🪄 **动态图片输入**：BizyAir / AutoDL / Geeknow 图生图节点提供 `inputcount` 与「更新图片输入」按钮，动态增减图像端口
- 💰 **节点价格标签**：BizyAir 节点显示蓝色金币价目，AutoDL 节点显示绿色人民币 Token 价目（右上角 Badge）
- 📤 **SSH 文件上传**：将工作流中的图片、音频、文本、模型文件或本地目录上传到 SSH 服务器
- 📦 **自动安装依赖**：首次加载时自动按 `requirements.txt` 安装缺失依赖（可通过环境变量关闭）
- ⚡ **异步并发执行**：所有 API 节点（BizyAir / AutoDL / Geeknow / 豆包 / DeepSeek）全部 `async def`，多个无依赖节点可并行请求，BizyAir ModelZoo 节点更使用 `aiohttp` 实现真正的异步 IO

## 节点一览

所有节点都注册在 ComfyUI 节点菜单的 `🍎MYAPI` 分类下。

### 文本生成 / 多模态理解

| 节点 | 服务 | 图像输入 | 主要参数 | 适用场景 |
|------|------|---------|----------|----------|
| 🥟 豆包MMM | 火山引擎 Doubao | 最多 5 张 | `model`、`max_tokens`、`reasoning_effort`（中文显示）、`seed` | 多模态理解、思维链推理、图文对比分析 |
| 🔎 DeepSeek V4 | DeepSeek | 0 张 | `model`、`enable_thinking`、`reasoning_effort`、`system_prompt`、`max_tokens`、`stream` | 长文本、代码、思考模式 |
| 🍎 AutodL API | AutoDL | 最多 5 张 | `model`、`system_prompt`、`user_prompt`、`seed` | 多模态对话、图文理解 |

**豆包 MMM 节点内可选模型：**

| 模型 ID | 定位 |
|---------|------|
| `doubao-seed-2-0-pro-260215` | 旗舰全能，复杂推理与 Agent |
| `doubao-seed-2-0-lite-260428` | 均衡型，质量与速度兼顾 |
| `doubao-seed-2-0-mini-260428` | 低时延高并发，支持多档思考深度 |
| `doubao-seed-1-8-251228` | 上一代通用 Agent，256K 上下文 |

**AutodL API 节点内可选模型：** qwen3.6-plus、Qwen3.5-397B-A17B、Kimi-K2.5、Kimi-K2.6、gpt-5.4-nano、gpt-5.4-mini、gpt-5.4、gpt-5.5、gemini-3.1-pro-preview

### 翻译

| 节点 | 服务 | 主要参数 | 适用场景 |
|------|------|---------|----------|
| 🥟 豆包翻译模型 | 火山引擎 Doubao Seed Translation | `source_language`、`target_language` | 30+ 种语言互译，文档与多语言内容处理 |

### 图像生成 / 图像编辑

#### BizyAir 系列（需充值金币，ModelZoo OpenAPI）

| 节点 | 输入图片 | 主要参数 | 适用场景 |
|------|---------|----------|----------|
| 🌐 BizyAir NanoBanana2 第三方渠道版 文生图 | 0 张 | `prompt`、`aspect_ratio`、`resolution`（1K/2K/4K） | 纯文本生成 |
| 🌐 BizyAir NanoBanana2 第三方渠道版 图生图 | 1–10 张（动态） | 同上 + `inputcount` | 多参考图编辑（OSS 上传） |
| 🌐 BizyAir NanoBanana2 官方版 文生图 | 0 张 | `prompt`、`aspect_ratio`、`resolution`（0.5K/1K/2K/4K）、`seed`、`web_search` | 官方线路文生图 |
| 🌐 BizyAir NanoBanana2 官方版 图生图 | 1–10 张（动态） | 同上 + `inputcount` | 官方线路图生图，支持联网搜索 |
| 🌐 BizyAir GPT-IMAGE-2 第三方渠道版 文生图 | 0 张 | `prompt`、`aspect_ratio`、`resolution`（1K/2K/4K） | 纯文本生成 |
| 🌐 BizyAir GPT-IMAGE-2 第三方渠道版 图生图 | 1–10 张（动态） | 同上 + `inputcount` | 多参考图合成（4K+1:1 自动降为 2K） |
| 🌐 BizyAir GPT-IMAGE-2 官方版 文生图 | 0 张 | `prompt`、`aspect_ratio`、`resolution`、`quality` | 官方线路，UI 比例映射为 width/height |
| 🌐 BizyAir GPT-IMAGE-2 官方版 图生图 | 1–16 张（动态） | 同上 + `inputcount` | 官方线路图生图，最多 16 参考图 |

#### AutoDL 系列（需 AutoDL 大模型 Token）

| 节点 | 输入图片 | 主要参数 | 适用场景 |
|------|---------|----------|----------|
| 🍎 AutodL Nano Banana 2 文生图 | 0 张 | `prompt`、`aspect_ratio`、`image_size`（0.5K/1K/2K/4K）、`seed` | 纯文本生成图像 |
| 🍎 AutodL Nano Banana 2 图生图 | 1–14 张（动态） | 同上 + `inputcount` | 多参考图融合 / 编辑 |
| 🍎 AutodL GPT-IMAGE-2 文生图 | 0 张 | `prompt`、`quality`、`resolution`（1K/2K/4K/auto）、`aspect_ratio`、`seed` | 纯文本生成图像 |
| 🍎 AutodL GPT-IMAGE-2 图生图 | 1–10 张（动态） | 同上 + `inputcount` | 多参考图图像合成 |

#### Geeknow 系列（需 Geeknow API 密钥）

| 节点 | 输入图片 | 主要参数 | 适用场景 |
|------|---------|----------|----------|
| 🍆 Geeknow GPT-IMAGE-2 文生图 | 0 张 | `line`、`model`、`quality`、`resolution`、`aspect_ratio`、`seed` | OpenAI Images 兼容文生图 |
| 🍆 Geeknow GPT-IMAGE-2 图生图 | 1–10 张（动态） | 同上 + `reference_mode`、`inputcount` | 多参考图合成 / 编辑 |
| 🍆 Geeknow Gemini 图像 文生图 | 0 张 | `line`、`model`、`aspect_ratio`、`image_size`（1K/2K）、`seed` | Gemini generateContent 文生图 |
| 🍆 Geeknow Gemini 图像 图生图 | 1–10 张（动态） | 同上 + `inputcount` | 多参考图融合 / 编辑 |

**Geeknow GPT-IMAGE-2 节点内可选模型：**

| 模型 ID | 说明 |
|---------|------|
| `gpt-image-2` | 基础档，最高 1K 分辨率 |
| `gpt-image-2-pro` | 高级档，支持 2K / 4K 分辨率 |

**Geeknow Gemini 图像节点内可选模型：**

| 模型 ID | 说明 |
|---------|------|
| `gemini-3-pro-image-preview` | Pro 档，支持 1K / 2K |
| `gemini-2.5-flash-image-preview` | Flash 档，2K 实际回落为 1K |
| `gemini-3.1-flash-image-preview` | Flash 档，2K 实际回落为 1K |

**Geeknow API 线路（`line` 参数）：**

| 选项 | 说明 |
|------|------|
| `https://geeknow.ai/v1 (cn2线路)` | cn2 线路（默认） |
| `https://api.geeknow.ai/v1 (cdn线路推荐国内用户)` | CDN 线路，推荐国内用户 |

**Geeknow 图生图参考图传递方式（仅 GPT-IMAGE-2 图生图）：**

| `reference_mode` | 说明 |
|------------------|------|
| `base64 内嵌` | 参考图以 Base64 直接写入请求 JSON（默认） |
| `上传获取URL` | 先调用 Geeknow 预签名上传接口获取公网 URL，再传入 `image` 字段 |

**Geeknow API 文档：**

- GPT-IMAGE-2：<https://docs.geeknow.top/api-reference/images/gpt-image-2/generation>
- GPT-IMAGE-2 Pro：<https://docs.geeknow.top/api-reference/images/gpt-image-2-pro/generation>
- Gemini 图像：<https://docs.geeknow.top/api-reference/images/gemini-image/generation>
- 图片上传：<https://docs.geeknow.top/api-reference/uploads/image-upload>

#### 豆包图像

| 节点 | 服务 | 主要参数 | 适用场景 |
|------|------|---------|----------|
| 🥟 豆包 SEEDREAM 4.5 | 火山引擎 Doubao | `prompt`、`size` / 自定义宽高、`seed`、`watermark`、`stream` | 自定义尺寸高质量图像生成 |

**图像节点支持的宽高比**

文生图 / 图生图类节点（BizyAir NanoBanana2、GPT-IMAGE-2，AutoDL / Geeknow 对应节点）均可在节点上选择宽高比，例如：`1:1`、`2:3`、`3:2`、`3:4`、`4:3`、`4:5`、`5:4`、`9:16`、`16:9`、`21:9` 等（以节点下拉选项为准）。

- **分辨率**：在节点的 `resolution` 或 `image_size` 中选择 `0.5K` / `1K` / `2K` / `4K`（AutoDL GPT-IMAGE-2 另有 `auto`）
- **图生图**：连接 `image`、`image2` … 参考图，用 `inputcount` 控制端口数量（见下文「动态图片输入」）

### 文本处理

| 节点 | 主要特性 | 适用场景 |
|------|---------|----------|
| 📝 文本分割 | 按关键词分割，支持包含 / 排除关键词，最多 20 段输出（`string_1` … `string_20`） | 提示词预处理、批量任务拆分 |

### 文件传输 / SSH 上传

| 节点 | 输入 | 主要参数 | 输出 | 适用场景 |
|------|------|---------|------|----------|
| 📤 SSH 文件上传 | `anyting` 通配输入 | `ssh_command`、`server`、`port`、`username`、`password`、`remote_dir`、`remote_filename` | `upload_info` | 上传图片、音频、文本、模型、目录到远程服务器 |

### 输出端口规范

- 所有图像输出端口统一为 `image`
- 所有字符串输出端口统一为 `string`（多输出节点为 `string_1`、`string_2` …）

## 价格参考

节点右上角会显示参考价格（ComfyUI 界面标签）。**实际扣费以各平台账单为准**。

### BizyAir（蓝色标签：金币 / 次）

| 节点 | 价格 |
|------|------|
| 🌐 BizyAir NanoBanana2 第三方渠道版 文生图 / 图生图 | **200** 金币 / 次（1K / 2K）；**250** 金币 / 次（4K） |
| 🌐 BizyAir NanoBanana2 官方版 文生图 / 图生图 | **550** 金币 / 次（0.5K / 1K）；**850** 金币 / 次（2K）；**1100** 金币 / 次（4K） |
| 🌐 BizyAir GPT-IMAGE-2 第三方渠道版 文生图 / 图生图 | **100** 金币 / 次 |
| 🌐 BizyAir GPT-IMAGE-2 官方版 文生图 / 图生图 | 见下表（依 `resolution` + `quality` 分档） |

**BizyAir GPT-IMAGE-2 官方版分档价目（金币 / 次）：**

| resolution | low | medium | high |
|------------|-----|--------|------|
| 1K | 161 | 378 | 1120 |
| 2K | 182 | 630 | 2149 |
| 4K | 224 | 966 | 3486 |

充值与余额查询：[BizyAir 官网](https://bizyair.cn)

### 火山方舟 / 豆包（人民币）

计费单位以 [火山方舟模型价格](https://www.volcengine.com/docs/82379/1544106) 为准。以下为**在线推理、输入 ≤32K** 档参考价（输入更长时分档加价）。

#### 豆包 MMM（随节点内 `model` 切换）

| 模型 ID | 输入 | 输出 | 缓存命中 |
|---------|------|------|----------|
| `doubao-seed-2-0-pro-260215` | ¥3.2 / M | ¥16 / M | ¥0.64 / M |
| `doubao-seed-2-0-lite-260428` | ¥0.6 / M | ¥3.6 / M | ¥0.12 / M |
| `doubao-seed-2-0-mini-260428` | ¥0.2 / M | ¥2 / M | ¥0.04 / M |
| `doubao-seed-1-8-251228` | ¥0.8 / M | ¥2 / M | ¥0.16 / M |

#### 豆包翻译

| 节点 | 输入 | 输出 |
|------|------|------|
| 🥟 豆包翻译模型 | ¥1.2 / 百万字符 | ¥3.6 / 百万字符 |

#### 豆包图像

| 节点 | 模型 | 价格 |
|------|------|------|
| 🥟 豆包 SEEDREAM 4.5 | `doubao-seedream-4-5-251128` | **¥0.25 / 张**（按生成图片数计费，prompt 不计费） |

密钥与账单：[火山方舟控制台](https://console.volcengine.com/ark)

### DeepSeek（人民币 / 百万 Token）

价格来源：[DeepSeek 官方价目](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)

| 模型 | 输入（缓存命中） | 输入（未命中） | 输出 |
|------|----------------|--------------|------|
| `deepseek-v4-flash` | ¥0.02 / M | ¥1 / M | ¥2 / M |
| `deepseek-v4-pro` | ¥0.025 / M | ¥3 / M | ¥6 / M |

`deepseek-v4-pro` 当前为限时优惠价（截至 2026-05-31），之后恢复为 ¥0.1 / ¥12 / ¥24。

### AutoDL（绿色标签：人民币 / 百万 Token）

计费单位：**¥ / M Token**（M = 一百万 Token）。图像按生成计费，聊天按输入 / 输出分别计费。

#### 图像节点

| 节点 | 输入 | 输出 |
|------|------|------|
| 🍎 AutodL Nano Banana 2 文生图 / 图生图 | ¥2.625 / M | ¥315.000 / M |
| 🍎 AutodL GPT-IMAGE-2 文生图 / 图生图 | ¥28.000 / M | ¥168.000 / M |

#### AutodL API 聊天（随节点内 `model` 切换）

| 模型 | 输入 | 输出 |
|------|------|------|
| qwen3.6-plus | ¥1.600 / M | ¥9.600 / M |
| Qwen3.5-397B-A17B | ¥0.720 / M | ¥4.320 / M |
| Kimi-K2.5 | ¥2.400 / M | ¥12.600 / M |
| Kimi-K2.6 | ¥3.900 / M | ¥16.200 / M |
| gpt-5.4-nano | ¥0.630 / M | ¥3.938 / M |
| gpt-5.4-mini | ¥2.363 / M | ¥14.175 / M |
| gpt-5.4 | ¥7.875 / M | ¥47.250 / M |
| gpt-5.5 | ¥17.500 / M | ¥105.000 / M |
| gemini-3.1-pro-preview | ¥10.500 / M | ¥63.000 / M |

令牌管理：[AutoDL 大模型 Token](https://autodl.art/large-model/tokens)

### Geeknow（紫色标签：人民币 / 次）

按生成次数计费，随节点内 `model` 切换。节点右上角绿色标签实时显示当前模型单价（格式 `GK ¥x.xx/次`）。

| 模型 | 节点 | 价格 |
|------|------|------|
| `gpt-image-2` | 🍆 Geeknow GPT-IMAGE-2 文生图 / 图生图 | **¥0.04 / 次** |
| `gpt-image-2-pro` | 🍆 Geeknow GPT-IMAGE-2 文生图 / 图生图 | **¥0.08 / 次** |
| `gemini-3-pro-image-preview` | 🍆 Geeknow Gemini 图像 文生图 / 图生图 | **¥0.22 / 次** |
| `gemini-3.1-flash-image-preview` | 🍆 Geeknow Gemini 图像 文生图 / 图生图 | **¥0.15 / 次** |
| `gemini-2.5-flash-image-preview` | 🍆 Geeknow Gemini 图像 文生图 / 图生图 | **¥0.06 / 次** |

密钥与余额查询：[Geeknow 官网](https://geeknow.ai)

## 动态图片输入

以下图生图节点支持按数量增减参考图端口：

| 节点 | 最大图片数 |
|------|-----------|
| 🌐 BizyAir NanoBanana2 第三方/官方版 图生图 | 10 |
| 🌐 BizyAir GPT-IMAGE-2 第三方渠道版 图生图 | 10 |
| 🌐 BizyAir GPT-IMAGE-2 官方版 图生图 | 16 |
| 🍎 AutodL Nano Banana 2 图生图 | 14 |
| 🍎 AutodL GPT-IMAGE-2 图生图 | 10 |
| 🍆 Geeknow GPT-IMAGE-2 图生图 | 10 |
| 🍆 Geeknow Gemini 图像 图生图 | 10 |

交互方式：

1. 修改 `inputcount` 后，图片输入端口（`image` / `image2` …）会自动增减
2. 点击节点底部 **「更新图片输入」** 可手动同步端口数量

不会删除已有连线的低位端口，只会清理超过 `inputcount` 的尾部端口。重启 ComfyUI 或刷新前端页面后生效。

## SSH 文件上传

`📤 SSH 文件上传` 节点用于把 ComfyUI 工作流中的文件或内容通过 SFTP 上传到目标服务器。

### 连接信息

- `ssh_command`：可直接粘贴 `ssh -p 21656 root@connect.bjb2.seetacloud.com` 这类命令
- `server` / `port` / `username`：前端 JS 会根据 `ssh_command` 自动填写，也可手动修改
- `password`：SSH 密码，仅在本地用于连接目标服务器
- `remote_dir`：远端目标目录，例如 `/root/uploads`
- `remote_filename`：远端文件名模板，默认 `{stem}_{timestamp_ms}{ext}`

`remote_filename` 支持 `{basename}`、`{stem}`、`{ext}`、`{timestamp_ms}`、`{timestamp}`、`{uuid}` 占位符。批量工作流建议保留时间戳或 UUID，避免同名覆盖。

### 支持的输入

`anyting` 是通配输入端口，常见输入会按以下方式处理：

- 本地文件路径：直接上传原文件
- 本地目录路径：递归上传整个目录，保留子目录结构
- `IMAGE` 张量：临时保存为 PNG 后上传
- 常见 `AUDIO` 字典：临时保存为 WAV 后上传
- 文本字符串：若不是本地路径，则保存为 TXT 后上传
- `bytes` / `bytearray`：保存为 BIN 后上传
- 普通字典：优先识别音频 / 文本，否则保存为 JSON 后上传

上传完成后，节点会输出 `upload_info` 文本，其中包含是否成功、本地路径、远端路径等简要信息。

## API 密钥

各服务的密钥统一在对应节点的 `api_key` 输入框中填写，不读取插件目录下的配置文件。

| 服务 | 获取入口 |
|------|---------|
| 🥟 豆包 / 火山方舟 | <https://www.volcengine.com/experience/ark> |
| 🌐 BizyAir | <https://bizyair.cn> |
| 🔎 DeepSeek | <https://platform.deepseek.com/> |
| 🍎 AutoDL | <https://autodl.art/large-model/tokens> |
| 🍆 Geeknow | <https://geeknow.ai>（密钥在平台控制台获取） |

## 安装

### 1. 安装插件

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zealman2025/Comfyui-MyApi.git
```

### 2. 安装依赖

首次启动时会按 `requirements.txt` 自动安装缺失依赖。如需手动安装：

**官方版 ComfyUI**

```bash
G:\ComfyUI安装目录\python_embeded\python.exe -m pip install -r requirements.txt
```

**秋叶版 ComfyUI**

```bash
G:\ComfyUI安装目录\python\python.exe -m pip install -r requirements.txt
```

如要禁用自动安装，可设置环境变量：

```text
COMFYUI_MYAPI_SKIP_AUTO_INSTALL=1
```

### 3. 重启 ComfyUI

完成依赖安装与节点注册后，重启 ComfyUI 即可在 `🍎MYAPI` 分类中看到所有节点及价格标签。

## 使用建议

### 密钥管理

- 日常使用：在节点 `api_key` 中直接填写
- 分享工作流：导出前清空 `api_key`，避免密钥泄露

### 图像输入

- 推荐使用适中分辨率，节点会在上传前自动压缩 / 缩放，避免超过服务端体积限制
- BizyAir 图生图需先上传参考图，网络较慢时请耐心等待
- Geeknow 节点请选择正确的 `line` 线路；国内用户建议优先使用 CDN 线路
- Geeknow GPT-IMAGE-2 图生图若参考图较大，可切换 `reference_mode` 为「上传获取URL」以减小请求体积
- Geeknow Gemini 图生图参考图通过 `inlineData` Base64 内嵌传递，节点会自动压缩参考图
- 多图节点请按 `image` / `image2` / `image3` … 顺序连接，与 `inputcount` 对齐

### 常用参数

| 参数 | 说明 |
|------|------|
| `seed` | 固定随机性，便于复现（AutoDL 图像节点） |
| `aspect_ratio` | 图像宽高比 |
| `resolution` / `image_size` | 分辨率档位（各节点命名略有不同） |
| `line` | Geeknow API 线路：cn2 线路 / CDN 线路 |
| `quality` | GPT-IMAGE-2 画质：`low` / `medium` / `high` / `auto` |
| `reference_mode` | Geeknow GPT-IMAGE-2 图生图参考图传递：`base64 内嵌` / `上传获取URL` |
| `inputcount` | 动态图像端口数量 |
| `reasoning_effort` | 豆包 MMM 思考深度：不思考 / 轻量 / 均衡 / 深度 |
| `max_tokens` | 单次输出最大 Token 数 |

## 注意事项

- 各服务都有调用频率与额度限制，请遵守对应服务条款
- BizyAir 节点需在平台充值金币后使用；AutoDL / Geeknow / 火山方舟 / DeepSeek 按 Token 或按张计费，请保证账户余额充足
- 海外 / 中转服务需要稳定的网络连接
- 所有节点的密钥仅在本地节点中使用，不会上传到第三方
- 旧版单一节点「🍎AutodL Nano Banana 2」已拆成「文生图」「图生图」两个节点，旧工作流请改连新节点

## 许可证

本项目基于 MIT 许可证开源，详见 `LICENSE`。

## 反馈

欢迎在 GitHub Issue 中提交 Bug、改进建议或新节点需求。

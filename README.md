# 🍎 ComfyUI MyAPI - 多模态 AI 节点集合

[![Version](https://img.shields.io/badge/version-2.0.3-blue.svg)](https://github.com/zealman2025/Comfyui-MyApi/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个面向 ComfyUI 的多模态 AI 节点集合，集成豆包、DeepSeek、BizyAir、AutoDL 等服务，覆盖文本生成、视觉理解、图像生成、图像编辑、翻译、文本处理等常见场景。所有节点都遵循统一的密钥与输入输出规范，便于在工作流中混搭使用。

## 主要特性

- 🔑 **节点内填写密钥**：每个节点的 `api_key` 输入框中独立填写，不读取本地配置文件
- 🖼️ **多模态支持**：覆盖文本、图像理解、图像生成、图像编辑、翻译等场景
- 🎯 **多服务集成**：豆包、DeepSeek、BizyAir、AutoDL 等主流 AI 服务一站式接入
- 🧩 **统一输出规范**：所有字符串输出端口统一命名为 `string`，便于上下游连接
- 🪄 **动态图片输入**：BizyAir / AutoDL 图生图节点提供 `inputcount` 与「更新图片输入」按钮，动态增减图像端口
- 💰 **节点价格标签**：BizyAir 节点显示蓝色金币价目，AutoDL 节点显示绿色人民币 Token 价目（右上角 Badge）
- 📤 **SSH 文件上传**：将工作流中的图片、音频、文本、模型文件或本地目录上传到 SSH 服务器
- 📦 **自动安装依赖**：首次加载时自动按 `requirements.txt` 安装缺失依赖（可通过环境变量关闭）

## 节点一览

所有节点都注册在 ComfyUI 节点菜单的 `🍎MYAPI` 分类下。

### 文本生成 / 多模态理解

| 节点 | 服务 | 图像输入 | 主要参数 | 适用场景 |
|------|------|---------|----------|----------|
| 🥟 豆包MMM | 火山引擎 Doubao | 最多 5 张 | `model`、`max_tokens`、`reasoning_effort`（中文显示）、`seed` | 多模态理解、思维链推理、图文对比分析 |
| 🔎 DeepSeek V4 | DeepSeek | 0 张 | `model`（`deepseek-v4-pro` / `deepseek-v4-flash`）、`enable_thinking`、`reasoning_effort`、`system_prompt`、`max_tokens`、`stream`；输出 `string` + `reasoning` | 长文本理解、代码生成、思考模式推理 |
| 🍎 AutodL API | AutoDL 中转 | 最多 5 张 | `model`、`system_prompt`、`user_prompt`、`seed` | 通过 AutoDL 中转访问的多模态聊天 |

**AutodL API 可选模型：** `qwen3.6-plus`、`Qwen3.5-397B-A17B`、`Kimi-K2.5`、`Kimi-K2.6`、`gpt-5.4-nano`、`gpt-5.4-mini`、`gpt-5.4`、`gpt-5.5`、`gemini-3.1-pro-preview`

### 翻译

| 节点 | 服务 | 主要参数 | 适用场景 |
|------|------|---------|----------|
| 🥟 豆包翻译模型 | 火山引擎 Doubao Seed Translation | `source_language`、`target_language` | 30+ 种语言互译，文档与多语言内容处理 |

### 图像生成 / 图像编辑

#### BizyAir 系列（平台封装，需充值金币）

| 节点 | 输入图片 | 主要参数 | 适用场景 |
|------|---------|----------|----------|
| 🌐 BizyAir NanoBanana2 | 1–10 张（动态） | `prompt`、`aspect_ratio`、`resolution`（1K/2K/4K）、`inputcount` | 多图融合 / 编辑 |
| 🌐 BizyAir GPT-IMAGE-2 文生图 | 0 张 | `prompt`、`aspect_ratio`、`resolution`（1k/2k/4k） | 纯文本驱动图像生成 |
| 🌐 BizyAir GPT-IMAGE-2 图生图 | 1–10 张（动态） | `prompt`、`aspect_ratio`、`resolution`、`inputcount` | 多参考图图像合成 |

#### AutoDL 系列（官方协议中转）

| 节点 | 协议 | 输入图片 | 主要参数 | 适用场景 |
|------|------|---------|----------|----------|
| 🍎 AutodL Nano Banana 2 文生图 | Gemini `generateContent` | 0 张 | `prompt`、`aspect_ratio`、`image_size`（0.5K/1K/2K/4K）、`seed` | 官方 Gemini 文生图 |
| 🍎 AutodL Nano Banana 2 图生图 | Gemini `generateContent` | 1–14 张（动态） | 同上 + `inputcount` | 官方 Gemini 多参考图编辑 |
| 🍎 AutodL GPT-IMAGE-2 文生图 | OpenAI Responses | 0 张 | `prompt`、`quality`、`resolution`（1K/2K/4K/auto）、`aspect_ratio`、`seed` | 官方 Responses 文生图 |
| 🍎 AutodL GPT-IMAGE-2 图生图 | OpenAI Responses | 1–10 张（动态） | 同上 + `inputcount` | 官方 Responses 多参考图编辑 |

#### 豆包图像

| 节点 | 服务 | 主要参数 | 适用场景 |
|------|------|---------|----------|
| 🥟 豆包 SEEDREAM 4.5 | 火山引擎 Doubao | `prompt`、`size` / 自定义宽高、`seed`、`watermark`、`stream` | 自定义尺寸高质量图像生成 |

**宽高比说明**

- BizyAir / AutoDL NanoBanana2 / AutoDL GPT-IMAGE-2 均支持：`1:1` `2:3` `3:2` `3:4` `4:3` `4:5` `5:4` `9:16` `16:9` `21:9`
- AutoDL GPT-IMAGE-2 在节点内选择 `resolution` + `aspect_ratio`，后台自动映射为官方 `size`（如 1K + 16:9 → `1824x1024`）；选 `auto` 则传官方 `size: auto`

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

节点右上角会显示实时价格标签（前端 JS 扩展）。**实际扣费以各平台账单为准**，下表为当前插件内置的参考价目。

### BizyAir（蓝色标签：金币 / 次）

| 节点 | 价格 |
|------|------|
| 🌐 BizyAir GPT-IMAGE-2 文生图 | **100** 金币 / 张（1k / 2k / 4k 同价） |
| 🌐 BizyAir GPT-IMAGE-2 图生图 | **100** 金币 / 张（1k / 2k / 4k 同价） |
| 🌐 BizyAir NanoBanana2 | **200** 金币 / 张（1K / 2K）；**250** 金币 / 张（4K） |

充值与余额查询：[BizyAir 官网](https://bizyair.cn)

### AutoDL（绿色标签：人民币 / 百万 Token）

计费单位：**¥ / M Token**（M = 1,000,000 Token）。图像类与聊天类均按输入 / 输出 Token 分别计费。节点右上角标签与下表一致。

#### 图像节点

| 节点 | 计费模型 | 输入 | 输出 |
|------|---------|------|------|
| 🍎 AutodL Nano Banana 2 文生图 / 图生图 | `nano-banana-2` | ¥2.625 / M | ¥315.000 / M |
| 🍎 AutodL GPT-IMAGE-2 文生图 / 图生图 | `gpt-image-2` | ¥28.000 / M | ¥168.000 / M |

#### AutodL API 聊天模型（随 `model` 切换）

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

## 动态图片输入

以下节点接入了前端 JS 扩展 `web/dynamic_image_inputs.js`：

| 节点 | 最大图片数 |
|------|-----------|
| 🌐 BizyAir NanoBanana2 | 10 |
| 🌐 BizyAir GPT-IMAGE-2 图生图 | 10 |
| 🍎 AutodL Nano Banana 2 图生图 | 14 |
| 🍎 AutodL GPT-IMAGE-2 图生图 | 10 |

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

上传结果输出 `upload_info` 字符串（JSON），包含 `success`、`local_path`、`remote_path`、`file_url`、`file_size`、`error` 等字段。

## API 密钥

各服务的密钥统一在对应节点的 `api_key` 输入框中填写，不读取插件目录下的配置文件。

| 服务 | 获取入口 |
|------|---------|
| 🥟 豆包 / 火山方舟 | <https://www.volcengine.com/experience/ark> |
| 🌐 BizyAir | <https://bizyair.cn> |
| 🔎 DeepSeek | <https://platform.deepseek.com/> |
| 🍎 AutoDL | <https://autodl.art/large-model/tokens> |

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
- BizyAir 系列节点会先通过 OSS 三步上传，再发起生成请求
- 多图节点请按 `image` / `image2` / `image3` … 顺序连接，与 `inputcount` 对齐

### 常用参数

| 参数 | 说明 |
|------|------|
| `seed` | 固定随机性，便于复现（AutoDL 图像节点） |
| `aspect_ratio` | 图像宽高比 |
| `resolution` / `image_size` | 分辨率档位（各节点命名略有不同） |
| `quality` | GPT-IMAGE-2 渲染质量：`low` / `medium` / `high` / `auto` |
| `inputcount` | 动态图像端口数量 |
| `reasoning_effort` | 豆包 MMM 思考深度：不思考 / 轻量 / 均衡 / 深度 |
| `max_tokens` | 单次输出最大 Token 数 |

## 注意事项

- 各服务都有调用频率与额度限制，请遵守对应服务条款
- BizyAir 节点需在平台充值金币后使用；AutoDL 按 Token 计费，需保证账户余额充足
- 海外 / 中转服务需要稳定的网络连接
- 所有节点的密钥仅在本地节点中使用，不会上传到第三方
- 旧版 `🍎AutodL Nano Banana 2`（单节点）已拆分为文生图 / 图生图两个节点，旧工作流需手动替换

## 许可证

本项目基于 MIT 许可证开源，详见 `LICENSE`。

## 反馈

欢迎在 GitHub Issue 中提交 Bug、改进建议或新节点需求。

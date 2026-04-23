# 🍎 ComfyUI MyAPI - 多模态 AI 节点集合

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/zealman2025/Comfyui-MyApi/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个面向 ComfyUI 的多模态 AI 节点集合，集成豆包、DeepSeek、BizyAir、AutoDL 等服务，覆盖文本生成、视觉理解、图像生成、图像编辑、翻译、文本处理等常见场景。所有节点都遵循统一的密钥与输入输出规范，便于在工作流中混搭使用。

## 主要特性

- 🔑 **节点内填写密钥**：每个节点的 `api_key` 输入框中独立填写，不读取本地配置文件
- 🖼️ **多模态支持**：覆盖文本、图像理解、图像生成、图像编辑、翻译等场景
- 🎯 **多服务集成**：豆包、DeepSeek、BizyAir、AutoDL 等主流 AI 服务一站式接入
- 🧩 **统一输出规范**：所有字符串输出端口统一命名为 `string`，便于上下游连接
- 🪄 **动态图片输入**：BizyAir 系列节点提供 `inputcount` 数量控制和「更新图片输入」按钮，动态增减图像端口
- 📦 **自动安装依赖**：首次加载时自动按 `requirements.txt` 安装缺失依赖（可通过环境变量关闭）

## 节点一览

所有节点都注册在 ComfyUI 节点菜单的 `🍎MYAPI` 分类下。

### 文本生成 / 多模态理解

| 节点 | 服务 | 图像输入 | 主要参数 | 适用场景 |
|------|------|---------|----------|----------|
| 🥟 豆包MMM | 火山引擎 Doubao | 最多 5 张 | `model`、`max_tokens`、`reasoning_effort`（中文显示）、`seed` | 多模态理解、思维链推理、图文对比分析 |
| 🔎 DeepSeek V3.2 | DeepSeek | 0 张 | `model`、`system_prompt`、`temperature`、`max_tokens`、`top_p`、`stream` | 长文本理解、代码生成、推理对话 |
| 🍎 AutodL API | AutoDL 中转 | 最多 5 张 | `model`、`system_prompt`、`user_prompt`、`seed` | 通过 AutoDL 中转访问的多模态聊天 |

### 翻译

| 节点 | 服务 | 主要参数 | 适用场景 |
|------|------|---------|----------|
| 🥟 豆包翻译模型 | 火山引擎 Doubao Seed Translation | `source_language`、`target_language` | 30+ 种语言互译，文档与多语言内容处理 |

### 图像生成 / 图像编辑

| 节点 | 服务 | 输入图片 | 主要参数 | 适用场景 |
|------|------|---------|----------|----------|
| 🌐 BizyAir NanoBanana2 | BizyAir | 1–6 张（动态） | `prompt`、`aspect_ratio`、`resolution`、`inputcount`、`mode` | 多图融合 / 编辑，需 BizyAir 充值金币 |
| 🌐 BizyAir GPT-IMAGE-2 文生图 | BizyAir | 0 张 | `prompt`、`aspect_ratio` | 纯文本驱动的图像生成 |
| 🌐 BizyAir GPT-IMAGE-2 图生图 | BizyAir | 1–4 张（动态） | `prompt`、`aspect_ratio`、`inputcount` | 多参考图驱动的图像合成 |
| 🥟 豆包 SEEDREAM 5 | 火山引擎 Doubao | 自定义 | `prompt`、`size`/`custom_width`/`custom_height`、`seed`、`watermark`、`stream` | 自定义尺寸的高质量图像生成 |
| 🍎 AutodL Nano Banana 2 | AutoDL 中转 | 1 张 | `prompt`、`aspect_ratio`、`image_resolution`、`seed` | 通过 AutoDL 中转的图像生成 / 编辑 |

GPT-IMAGE-2 系列支持的宽高比：`1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 4:5 / 5:4 / 9:16 / 16:9 / 21:9`。

### 文本处理

| 节点 | 主要特性 | 适用场景 |
|------|---------|----------|
| 📝 文本分割 | 按关键词分割文本，支持包含 / 排除关键词，最多 20 段输出（`string_1` … `string_20`） | 提示词预处理、批量任务拆分 |

### 输出端口规范

- 所有图像输出端口统一为 `image`
- 所有字符串输出端口统一为 `string`（多输出节点为 `string_1`、`string_2` …）

## 动态图片输入（BizyAir 系列）

`BizyAir NanoBanana2` 与 `BizyAir GPT-IMAGE-2 图生图` 接入了一份前端 JS 扩展，提供两种交互：

1. 修改 `inputcount` 数值后，节点上的图片输入端口会自动按数量增减
2. 节点底部的 `更新图片输入` 按钮可以手动触发同步

不会删除已有连线的低位端口，只会清理超过 `inputcount` 的尾部端口。重启 ComfyUI 或刷新前端页面后即可看到效果。

## API 密钥

各服务的密钥统一在对应节点的 `api_key` 输入框中填写，不读取插件目录下的配置文件。

### 各服务获取入口

- 🥟 豆包 / 火山方舟 Ark：<https://www.volcengine.com/experience/ark>
- 🌐 BizyAir：<https://bizyair.cn>
- 🔎 DeepSeek：<https://platform.deepseek.com/>
- 🍎 AutoDL：<https://autodl.art/large-model/tokens>

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

完成依赖安装与节点注册后，重启 ComfyUI 即可在 `🍎MYAPI` 分类中看到所有节点。

## 使用建议

### 密钥管理

- 日常使用：在节点 `api_key` 中直接填写
- 分享工作流：导出前清空 `api_key`，避免密钥泄露

### 图像输入

- 推荐使用适中分辨率，节点会在上传前自动压缩 / 缩放，避免超过服务端体积限制
- BizyAir 系列节点会先通过 OSS 三步上传，再发起生成请求；网络较差时建议适当增加超时
- 多图节点请按 `image / image2 / image3 …` 的顺序连接，与 `inputcount` 对齐

### 参数说明

- `seed`：固定随机性，便于复现
- `temperature` / `top_p`：控制输出随机性（仅部分文本节点支持）
- `max_tokens`：单次输出最大 token 数
- `reasoning_effort`（豆包 MMM）：控制思考深度，分为 `不思考 / 轻量思考 / 均衡思考 / 深度思考`
- `aspect_ratio`：图像宽高比
- `inputcount`：动态图像端口数量（BizyAir 系列）

## 注意事项

- 各服务都有调用频率与额度限制，请遵守对应服务条款
- BizyAir 节点需在 BizyAir 平台充值金币后使用
- 海外服务（如部分 AutoDL 中转、外部 LLM）需要稳定的网络连接
- 所有节点的密钥仅在本地节点中使用，不会上传到第三方

## 许可证

本项目基于 MIT 许可证开源，详见 `LICENSE`。

## 反馈

欢迎在 GitHub Issue 中提交 Bug、改进建议或新节点需求。

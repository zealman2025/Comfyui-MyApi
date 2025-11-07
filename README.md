# 🍎 ComfyUI MyAPI - 多模态AI节点集合

一个功能强大的ComfyUI插件，集成了多个主流AI服务，支持文本生成、图像理解、图像生成等多模态AI功能。

<img width="1682" height="1479" alt="image" src="https://github.com/user-attachments/assets/b0828902-4add-48ac-868e-69fb0931770e" />
<img width="1856" height="1186" alt="image" src="https://github.com/user-attachments/assets/c476ae65-7969-40cc-b58e-3186fcb1e4c0" />

## ✨ 主要特性

- 🔑 **双重API密钥机制** - 支持节点输入和配置文件两种密钥管理方式
- 🖼️ **多模态支持** - 文本生成、图像理解、图像生成一应俱全
- 🎯 **多服务集成** - 支持Qwen、豆包、DeepSeek、XAI Grok、Gemini、OpenRouter等主流AI服务
- 🛠️ **灵活配置** - 可自定义模型列表和参数设置
- 🔄 **智能回退** - 配置加载失败时自动使用默认模型
- 📝 **详细日志** - 完整的调试信息和错误提示

## 🚀 支持的AI节点

### 📝 文本生成节点
- **🍭 Qwen AI** - 阿里通义千问，支持2张图片输入
- **🥟 豆包 AI** - 字节跳动豆包，支持2张图片输入
- **🍭 DeepSeek V3.2 Exp** - DeepSeek V3.2实验版，支持深度推理
- **🚀 XAI Grok** - xAI的Grok模型，支持2张图片输入
- **🌟 Gemini AI** - Google Gemini，支持2张图片输入

### 🌐 翻译节点
- **🥟 Doubao-Seed-Translation | 豆包翻译模型** - 豆包Seed翻译模型，支持30+种语言互译

### 🎨 图像生成节点
- **🌐 Gemini 2.5 Flash Image Preview** - 通过OpenRouter调用，支持5张图片输入
- **🍌 BizyAir NanoBanana** - BizyAir图像生成服务，需BizyAir.cn充值金币
- **🌈 BizyAir Seedream4** - BizyAir高级图像生成，需BizyAir.cn充值金币
- **🥟 豆包 SEEDREAM 4.0** - 豆包图像生成，支持10张图片输入

### ✏️ 图像编辑节点
- **🍭 Qwen Image Edit Plus** - Qwen图像编辑模型，支持1-3张图片输入，智能图像编辑和合成

## 🔑 双重API密钥机制

本插件支持两种API密钥管理方式，按优先级自动选择：

### **优先级1: 节点输入**
在每个节点的第一个输入框中直接输入API密钥，适用于：
- 临时切换不同的API密钥
- 测试新的API密钥
- 分享工作流时保护隐私

### **优先级2: 配置文件**
在`config.json`中预设API密钥，适用于：
- 日常使用的常用密钥
- 避免重复输入
- 批量管理多个密钥

## ⚙️ 配置文件设置

### 1. 初始化配置文件
首先将`config.json.example`重命名为`config.json`（避免更新时覆盖您的配置）

### 2. 配置文件结构
```json
{
    "qwen_api_key": "你的apikey",
    "doubao_api_key": "你的apikey",
    "xai_api_key": "你的apikey",
    "gemini_api_key": "你的apikey",
    "openrouter_api_key": "你的apikey",
    "bizyair_api_key": "你的apikey",
    "deepseek_api_key": "你的apikey",
    "models": {
        "qwen": {
            "qwen3-vl-plus": "qwen3-vl-plus",
            "qwen3-vl-flash": "qwen3-vl-flash"
        },
        "qwen_image_edit": {
            "qwen-image-edit-plus": "Qwen Image Edit Plus"
        },
        "doubao": {
            "doubao-1-5-thinking-vision-pro-250428": "Doubao-1.5-thinking-vision-pro",
            "doubao-seed-1-6-250615": "豆包Seed1.6版"
        },
        "doubao_translation": {
            "doubao-seed-translation-250915": "豆包Seed翻译模型"
        },
        "doubao_seedream": {
            "doubao-seedream-4-0-250828": "豆包SEEDREAM 4.0"
        },
        "xai": {
            "grok-2-vision-1212": "Grok 2 Vision 1212",
            "grok-4": "grok-4"
        },
        "gemini": {
            "gemini-2.5-pro": "Gemini 2.5 pro",
            "gemini-2.5-flash": "Gemini 2.5 flash"
        },
        "deepseek": {
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-reasoner": "DeepSeek Reasoner"
        }
    }
}
```

## 🔧 API密钥获取指南

### 🍭 Qwen (通义千问)
- **官网**: https://bailian.console.aliyun.com/?tab=api#/api
- **说明**: 阿里云百炼平台，国内可直接访问
- **配置**: `qwen_api_key`

### 🥟 豆包 (Doubao)
- **官网**: https://www.volcengine.com/experience/ark
- **说明**: 字节跳动火山引擎，国内可直接访问
- **配置**: `doubao_api_key`

### 🚀 XAI Grok
- **官网**: https://x.ai/api
- **说明**: xAI官方API，需要魔法上网
- **配置**: `xai_api_key`

### 🌟 Gemini
- **官网**: https://aistudio.google.com/
- **说明**: Google AI Studio，需要魔法上网
- **配置**: `gemini_api_key`

### 🌐 OpenRouter
- **官网**: https://openrouter.ai/
- **说明**: 多模型API聚合平台，支持支付宝和微信充值
- **配置**: `openrouter_api_key`

### 🍌 BizyAir
- **官网**: https://bizyair.cn
- **说明**: BizyAir图像生成平台，需要充值金币，国内可直接访问
- **配置**: `bizyair_api_key`

### 🍭 DeepSeek
- **官网**: https://platform.deepseek.com/
- **说明**: DeepSeek AI平台，支持深度推理模型，国内可直接访问
- **配置**: `deepseek_api_key`

## 📋 节点详细功能

### 📝 文本生成节点功能对比

| 节点 | 图片输入 | 主要特性 | 适用场景 |
|------|----------|----------|----------|
| 🍭 Qwen AI | 2张 | 中文优化，响应快速 | 中文对话、文档分析 |
| 🥟 豆包 AI | 2张 | 思维链推理，深度分析 | 复杂推理、学术研究 |
| 🍭 DeepSeek V3.2 Exp | 0张 | 深度推理，代码生成 | 代码编写、逻辑推理 |
| 🚀 XAI Grok | 2张 | 实时信息，幽默风格 | 新闻分析、创意写作 |
| 🌟 Gemini AI | 2张 | 多模态理解，精准分析 | 图像理解、代码生成 |

### 🌐 翻译节点功能

| 节点 | 支持语言 | 主要特性 | 适用场景 |
|------|----------|----------|----------|
| 🥟 豆包翻译模型 | 30+种 | 高质量翻译，支持多种语言对 | 文档翻译、多语言内容处理 |

### 🎨 图像生成节点功能对比

| 节点 | 输入图片 | 输出 | 主要特性 |
|------|----------|------|----------|
| 🌐 Gemini Image Preview | 5张 | 图像+文本 | 图像编辑、风格转换 |
| 🍌 BizyAir NanoBanana | 1张 | 图像+状态 | 快速生成、需充值金币 |
| 🌈 BizyAir Seedream4 | 1张 | 图像+状态 | 高质量生成、需充值金币 |
| 🥟 豆包 SEEDREAM 4.0 | 10张 | 图像+状态 | 多图合成、专业级生成 |

### ✏️ 图像编辑节点功能对比

| 节点 | 输入图片 | 输出 | 主要特性 |
|------|----------|------|----------|
| 🍭 Qwen Image Edit Plus | 1-3张 | 图像+状态 | 智能编辑、多图合成、姿势迁移 |

## 🛠️ 安装和使用

### 1. 安装插件
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zealman2025/Comfyui-MyApi.git
```

### 2. 安装依赖
大多数情况下无需手动安装依赖，ComfyUI的其他插件通常已包含所需依赖。

如需手动安装：

**官方版ComfyUI:**
```bash
G:\ComfyUI安装目录\python_embeded\python.exe -m pip install openai google-genai requests Pillow numpy dashscope
```

**秋叶版ComfyUI:**
```bash
G:\ComfyUI安装目录\python\python.exe -m pip install openai google-genai requests Pillow numpy dashscope
```

### 3. 配置API密钥
- 将`config.json.example`重命名为`config.json`
- 在配置文件中填入您的API密钥，或在节点中直接输入

### 4. 重启ComfyUI
修改配置后需要重启ComfyUI以加载新设置

## 💡 使用技巧

### 🔑 API密钥管理
- **日常使用**: 在config.json中配置常用密钥
- **临时测试**: 在节点输入框中输入测试密钥
- **分享工作流**: 清空节点输入框，避免泄露密钥

### 🖼️ 图像输入优化
- **单图分析**: 连接到第一个图像输入
- **对比分析**: 使用多个图像输入进行对比
- **图像格式**: 支持PNG、JPEG、WEBP等常见格式
- **分辨率**: 建议使用适中分辨率，避免过大图像

### ⚙️ 参数调优
- **Temperature**: 控制输出随机性 (0.0-2.0)
- **Top_p**: 核采样参数 (0.0-1.0)
- **Max_tokens**: 最大输出长度
- **Seed**: 种子值，确保输出一致性

## 🔧 自定义配置

### 添加新模型
在对应的模型类别下添加新模型：
```json
"qwen": {
    "existing-models": "...",
    "new-model-id": "新模型显示名称"
}
```

### 修改显示名称
```json
"models": {
    "qwen": {
        "qwen3-vl-plus": "自定义显示名称"
    }
}
```

## ⚠️ 注意事项

### 配置文件
- 确保JSON格式正确
- 使用UTF-8编码保存
- 修改后需重启ComfyUI
- 建议备份原配置文件

### 网络要求
- **国内服务** (Qwen、豆包、DeepSeek): 直接访问
- **国外服务** (Gemini、XAI、OpenRouter): 需要稳定的网络连接

### 使用限制
- 遵守各API服务的使用条款
- 注意API调用频率限制
- 不同模型有不同的token限制

## 🐛 故障排除

### 常见问题
1. **模型显示undefined**: 检查config.json格式和模型配置
2. **API调用失败**: 验证API密钥和网络连接
3. **图像处理错误**: 检查图像格式和大小
4. **依赖缺失**: 手动安装所需Python包

### 调试信息
插件提供详细的日志输出，查看ComfyUI控制台获取调试信息：
- `[NodeName] 使用输入的API密钥` - 使用节点输入的密钥
- `[NodeName] 使用config.json中的API密钥` - 使用配置文件密钥
- `[NodeName] 加载模型: {...}` - 模型加载状态

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 支持

如有问题或建议，请在GitHub上提交Issue。

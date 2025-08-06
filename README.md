<img width="1145" height="1178" alt="image" src="https://github.com/user-attachments/assets/db1ac98e-1c79-4243-9cd5-9d159b88ea5e" />

# 配置文件说明 (config.json)
首先你需要将config.json.example改名为config.json（这样做是为了之后更新时你本地的配置文件不会被替换）
## 简介
`config.json` 文件是所有API节点的统一配置文件，您可以通过修改此文件来：
- 更新各种API的密钥
- 添加或修改模型配置
- 自定义模型显示名称

## 配置文件结构

```json
{
    "qwen_api_key": "your_qwen_api_key_here",
    "doubao_api_key": "your_doubao_api_key_here", 
    "xai_api_key": "your_xai_api_key_here",
    "gemini_api_key": "your_gemini_api_key_here",
    "models": {
        "qwen": {
            "model_id": "显示名称",
            "qwen-plus": "Qwen Plus",
            "qwen-max": "Qwen Max"
        },
        "doubao": {
            "doubao-1-5-vision-pro-32k-250115": "豆包1.5视觉专业版32k"
        },
        "doubao_image_edit": {
            "doubao-seededit-3-0-i2i-250628": "豆包图像编辑3.0版"
        },
        "doubao_text_to_image": {
            "doubao-seedream-3-0-t2i-250415": "豆包文生图3.0版"
        },
        "xai": {
            "grok-2-vision-1212": "Grok 2 Vision 1212"
        },
        "gemini": {
            "gemini-2.0-flash": "Gemini 2.0 Flash",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-pro": "Gemini 2.5 Pro"
        }
    }
}
```

## 如何添加新模型

### 1. 添加Qwen模型
在 `models.qwen` 节点下添加：
```json
"qwen": {
    "existing-models": "...",
    "qwen-new-model": "新模型显示名称"
}
```

### 2. 添加豆包模型  
在 `models.doubao` 节点下添加：
```json
"doubao": {
    "existing-models": "...",
    "doubao-new-model": "新豆包模型"
}
```

### 3. 添加豆包图像编辑模型
在 `models.doubao_image_edit` 节点下添加：
```json
"doubao_image_edit": {
    "existing-models": "...",
    "doubao-new-edit-model": "新图像编辑模型"
}
```

### 4. 添加豆包文生图模型
在 `models.doubao_text_to_image` 节点下添加：
```json
"doubao_text_to_image": {
    "existing-models": "...",
    "doubao-new-t2i-model": "新文生图模型"
}
```

### 5. 添加XAI模型
在 `models.xai` 节点下添加：
```json
"xai": {
    "existing-models": "...",
    "grok-new-model": "新Grok模型"
}
```

### 6. 添加Gemini模型
在 `models.gemini` 节点下添加：
```json
"gemini": {
    "existing-models": "...",
    "gemini-new-model": "新Gemini模型"
}
```

## 修改API密钥

直接修改对应的API密钥字段：
```json
{
    "qwen_api_key": "新的qwen密钥",
    "doubao_api_key": "新的豆包密钥",
    "xai_api_key": "新的xai密钥",
    "gemini_api_key": "新的gemini密钥"
}
```

## 注意事项

1. **格式正确性**：确保JSON格式正确，注意逗号和引号
2. **编码格式**：建议使用UTF-8编码保存文件
3. **重启加载**：修改配置后需要重启ComfyUI才能生效
4. **备份配置**：建议修改前备份原配置文件
5. **模型ID**：模型ID（键名）必须是API支持的真实模型名称
6. **显示名称**：显示名称（键值）可以自定义，用于界面显示

## 豆包文生图节点尺寸说明

豆包文生图节点支持自定义宽度和高度：
- **宽度范围**：64-2048像素，步长8
- **高度范围**：64-2048像素，步长8  
- **默认尺寸**：768x1024（竖版）
- **常用尺寸参考**：
  - 768x1024 (3:4竖版)
  - 1024x768 (4:3横版)
  - 1024x1024 (1:1正方形)
  - 1024x576 (16:9宽屏)
  - 576x1024 (9:16手机竖屏)

## Gemini节点使用说明

### API密钥获取
1. Gemini(使用需要魔法)访问https://aistudio.google.com/
2. Xai(使用需要魔法)访问https://x.ai/api
3. QWEN访问https://bailian.console.aliyun.com/?tab=api#/api
4. Doubao访问https://www.volcengine.com/experience/ark

### 节点功能
- ✅ 文本生成和对话
- ✅ 图像理解和分析
- ✅ 支持多轮对话
- ✅ 可调节温度、top_p、top_k等参数
- ✅ 种子控制，支持确定性输出

### 依赖安装
```bash
pip install google-genai pillow numpy
```

### 使用注意事项
1. **API密钥**：确保在config.json中正确配置api_key
2. **网络连接**：需要稳定的网络连接到Google服务
3. **图像格式**：支持PNG、JPEG、WEBP等常见格式
4. **Token限制**：不同模型有不同的输入输出token限制

## 默认配置恢复

如果配置文件损坏，删除对应的models节点，程序会自动使用默认配置。

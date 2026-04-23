import torch
import os
import json

# 中文语言包
LANG_ZH = {
    "node_name": "文本分割",
    "text": "文本",
    "split_keyword": "分割关键词",
    "remove_text": "删除文本",
    "include_keyword": "包含关键词",
    "segment": "片段"
}

class SegmentTextNode:
    """
    文本分割节点：将输入的文本按照指定的关键词进行分割
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 检查ComfyUI是否使用中文语言
        lang = cls.get_language()
        
        if lang == "zh":
            return {
                "required": {
                    LANG_ZH["text"]: ("STRING", {"multiline": True}),
                    LANG_ZH["split_keyword"]: ("STRING", {"default": ","}),
                    LANG_ZH["remove_text"]: ("STRING", {"default": ""}),
                    LANG_ZH["include_keyword"]: ("BOOLEAN", {"default": True}),
                }
            }
        else:
            return {
                "required": {
                    "text": ("STRING", {"multiline": True}),
                    "split_keyword": ("STRING", {"default": ","}),
                    "remove_text": ("STRING", {"default": ""}),
                    "include_keyword": ("BOOLEAN", {"default": True}),
                }
            }
    
    @classmethod
    def get_language(cls):
        # 尝试检测ComfyUI的语言设置
        # 这里使用一个简单的方法，实际应用中可能需要更复杂的检测逻辑
        try:
            # 尝试读取ComfyUI的配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if "language" in config and config["language"] == "zh":
                        return "zh"
        except Exception:
            pass
        return "en"
    
    RETURN_TYPES = tuple(["STRING"] * 20)
    RETURN_NAMES = tuple([f"string_{i + 1}" for i in range(20)])
    FUNCTION = "segment_text"
    CATEGORY = "🍎MYAPI"

    def __init__(self):
        pass
    
    def segment_text(self, **kwargs):
        # 获取参数，处理中英文参数名称
        lang = self.get_language()
        
        if lang == "zh":
            text = kwargs.get(LANG_ZH["text"], "")
            split_keyword = kwargs.get(LANG_ZH["split_keyword"], ",")
            remove_text = kwargs.get(LANG_ZH["remove_text"], "")
            include_keyword = kwargs.get(LANG_ZH["include_keyword"], True)
        else:
            text = kwargs.get("text", "")
            split_keyword = kwargs.get("split_keyword", ",")
            remove_text = kwargs.get("remove_text", "")
            include_keyword = kwargs.get("include_keyword", True)
        
        # 如果有需要移除的文本，先进行移除
        if remove_text:
            text = text.replace(remove_text, "")
        
        # 根据是否包含关键词进行不同的分割处理
        if include_keyword:
            # 包含关键词的分割方式
            segments = []
            remaining_text = text
            
            for i in range(20):  # 最多支持20个输出
                if split_keyword in remaining_text:
                    # 找到下一个关键词的位置
                    next_keyword_pos = remaining_text.find(split_keyword, len(split_keyword))
                    
                    if next_keyword_pos != -1:
                        # 如果找到下一个关键词，截取到下一个关键词之前
                        segment = remaining_text[:next_keyword_pos]
                        segments.append(segment.strip())
                        # 更新剩余文本，从下一个关键词开始
                        remaining_text = remaining_text[next_keyword_pos:]
                    else:
                        # 如果没有找到下一个关键词，将剩余文本作为最后一个片段
                        segments.append(remaining_text.strip())
                        break
                else:
                    # 如果没有关键词，将剩余文本作为最后一个片段
                    segments.append(remaining_text.strip())
                    break
        else:
            # 不包含关键词的分割方式（传统分割）
            raw_segments = text.split(split_keyword)
            segments = []
            
            # 确保第一个元素不为空
            for seg in raw_segments:
                seg = seg.strip()
                if seg or len(segments) > 0:  # 只有第一个元素允许为空
                    segments.append(seg)
        
        # 限制片段数量
        segments = segments[:20]
        
        # 如果片段数量不足，用空字符串填充
        while len(segments) < 20:
            segments.append("")
        
        return tuple(segments)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "segment_text": SegmentTextNode
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "segment_text": "📝文本分割"
} 
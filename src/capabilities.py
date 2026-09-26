"""
能力维度体系与难度等级的唯一权威定义。

所有提示词（打分模型、任务画像模型、以及外部适配模块的题库标注模型）
都必须从本模块获取能力维度描述，保证 10 个维度的画像描述在不同提示词中完全一致。
"""

from collections import OrderedDict

# 10 个核心能力维度（编号 C1-C10），与方案文档 §3 保持一致。
# key 为英文字段名（用于 JSON 输出与画像存储），value 为 (中文名称, 含义描述)。
CAPABILITIES = OrderedDict([
    ("instruction_following", (
        "意图理解与指令遵循",
        "准确理解用户目标、约束条件、输出格式、隐含意图，并按照要求完成任务。",
    )),
    ("knowledge_qa", (
        "通用知识与事实问答",
        "事实准确性、常识判断、百科知识、领域基础知识覆盖。",
    )),
    ("reasoning", (
        "逻辑推理与数学能力",
        "多步推理、数学计算、证明、规划、因果分析、复杂问题拆解。",
    )),
    ("coding", (
        "代码与软件工程能力",
        "代码生成、代码解释、调试、测试、工程结构理解、API 使用。",
    )),
    ("long_context", (
        "文档理解与长上下文能力",
        "长文本、多文档、跨段整合、上下文保持、引用依据、全局理解。",
    )),
    ("data_analysis", (
        "数据/表格/结构化信息处理",
        "表格分析、JSON/XML 解析、数据抽取、统计解释、结构化转换。",
    )),
    ("multilingual_translation", (
        "多语言与翻译能力",
        "跨语言理解、翻译、本地化表达、多语言任务处理。",
    )),
    ("writing", (
        "写作与风格控制",
        "改写、总结、扩写、公文写作、创意写作、语气和风格控制。",
    )),
    ("tool_use", (
        "工具调用与代理执行能力",
        "函数调用、检索、代码执行、外部工具编排、多步骤 agent 任务。",
    )),
    ("safety", (
        "安全合规与拒答边界",
        "高风险识别、合规回答、拒答稳定性、隐私保护、安全边界控制。",
    )),
])

CAPABILITY_FIELDS = list(CAPABILITIES.keys())

# L1-L4 四级难度定义，与方案文档 §4.2 保持一致。
DIFFICULTY_LEVELS = OrderedDict([
    ("L1", (
        "常规型",
        "单步、事实性、简单格式转换、简单摘要、低风险任务。例如翻译一句话、改写短文本、简单分类、简单问答。",
    )),
    ("L2", (
        "标准型",
        "需要一定综合能力，但任务边界清楚。例如普通代码修复、常规文档总结、一般业务问答、简单表格分析。",
    )),
    ("L3", (
        "复杂型",
        "多步骤推理、多约束、多文档、较强专业性。例如方案设计、复杂代码定位、长文档比较、多因素决策分析。",
    )),
    ("L4", (
        "专家/代理型",
        "高专业、高风险、多工具、多轮规划或复杂上下文任务。例如深度研究、复杂 agent 任务、法律/医学辅助分析、大型代码库问题定位。",
    )),
])

DIFFICULTY_ORDER = list(DIFFICULTY_LEVELS.keys())

# 连续难度分与离散难度等级的映射区间（方案 §4.2）
DIFFICULTY_SCORE_RANGES = {
    "L1": (0.0, 0.25),
    "L2": (0.25, 0.50),
    "L3": (0.50, 0.75),
    "L4": (0.75, 1.00),
}

# 全局能力档位（弱 → 专家）
TIERS = ["weak", "standard", "strong", "expert"]

TIER_NAMES_ZH = {
    "weak": "弱模型",
    "standard": "标准模型",
    "strong": "强模型",
    "expert": "专家模型",
}

# 任务难度 → 默认匹配模型档位（方案 §6）
DEFAULT_DIFFICULTY_TIER_MAP = {
    "L1": "weak",
    "L2": "standard",
    "L3": "strong",
    "L4": "expert",
}


def format_capability_descriptions() -> str:
    """将 10 个能力维度渲染为提示词中的统一描述块。"""
    lines = []
    for idx, (field, (name_zh, desc)) in enumerate(CAPABILITIES.items(), start=1):
        lines.append(f"C{idx}. {field}（{name_zh}）：{desc}")
    return "\n".join(lines)


def format_difficulty_descriptions() -> str:
    """将 L1-L4 难度等级渲染为提示词中的统一描述块。"""
    lines = []
    for level, (name_zh, desc) in DIFFICULTY_LEVELS.items():
        lines.append(f"{level}（{name_zh}）：{desc}")
    return "\n".join(lines)


def normalize_capability_key(key: str) -> str:
    """
    将能力维度 key 归一化为标准英文字段名。

    兼容 LLM 偶尔输出的 C1-C10 编号形式；无法识别时返回空字符串。
    """
    key = str(key).strip()
    if key in CAPABILITIES:
        return key
    normalized = key.upper().replace("C-", "C").replace(" ", "")
    if normalized.startswith("C") and normalized[1:].isdigit():
        idx = int(normalized[1:])
        if 1 <= idx <= len(CAPABILITY_FIELDS):
            return CAPABILITY_FIELDS[idx - 1]
    return ""


def score_to_level(difficulty_score: float) -> str:
    """连续难度分映射到离散难度等级。"""
    for level, (low, high) in DIFFICULTY_SCORE_RANGES.items():
        if low <= difficulty_score < high:
            return level
    return "L4" if difficulty_score >= 1.0 else "L1"

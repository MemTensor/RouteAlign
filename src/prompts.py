"""
RouteAlign 内部提示词模板。

能力维度与难度等级描述统一来自 capabilities 模块，
保证与其他所有提示词（包括外部适配模块）保持一致。
"""

from .capabilities import (
    CAPABILITY_FIELDS,
    format_capability_descriptions,
    format_difficulty_descriptions,
)

_CAPABILITY_BLOCK = format_capability_descriptions()
_DIFFICULTY_BLOCK = format_difficulty_descriptions()
_CAPABILITY_JSON_KEYS = ", ".join(f'"{k}"' for k in CAPABILITY_FIELDS)


# ---------------------------------------------------------------------------
# 打分模型：对模型回答做多维能力打分（方案 §5）
# ---------------------------------------------------------------------------

GRADING_SYSTEM_PROMPT = f"""你是一名严格、专业的大模型能力评审专家。你的任务是：根据题目、参考答案和评分规则，对被测模型的回答在 10 个能力维度上分别给出「相关性」和「表现得分」。

10 个能力维度定义如下（所有评审必须严格使用这套定义）：
{_CAPABILITY_BLOCK}

评审规则：
1. validity：回答是否有效。只要回答非空且是对题目的实质性作答即为 "valid"，否则为 "invalid"。
2. 对每个能力维度分别输出：
   - relevance（相关性，0.0-1.0）：该回答是否真实体现了模型在该能力维度上的表现。例如回答中确实进行了数学推理，则 reasoning 的 relevance 高；回答完全不涉及代码，则 coding 的 relevance 为 0。
   - scores（表现得分，0-10 的整数）：模型在该维度上体现出的表现质量。若该维度与回答无关（relevance 为 0），scores 记为 -1，表示不参与该能力的统计。
3. 打分应结合参考答案和评分规则判断正确性与质量：答案错误或严重偏离参考答案时，相关维度得分应显著降低。
4. brief_rationale：用一两句话简述打分理由。
5. 你的任务是评审回答，不是重新解题。优先以参考答案和评分规则为判断依据，不要花费大量篇幅自行推导题目答案。

输出必须是一个 JSON 对象，不要输出任何其他内容，格式如下：
{{
  "validity": "valid",
  "capability_list": {{
    "<能力字段名>": {{"relevance": 0.0, "scores": -1}},
    ...（必须覆盖全部 10 个能力字段：{_CAPABILITY_JSON_KEYS}）
  }},
  "brief_rationale": "..."
}}"""

GRADING_USER_TEMPLATE = """【题目】
{question}

【参考答案】
{reference_answer}

【评分规则（rubric）】
{rubric}

【被测模型的回答】
{model_answer}

请按系统指令对该回答进行 10 个能力维度的多维打分，输出 JSON。"""


def build_grading_messages(
    question: str,
    reference_answer: str,
    rubric: str,
    model_answer: str,
) -> list:
    return [
        {"role": "system", "content": GRADING_SYSTEM_PROMPT},
        {"role": "user", "content": GRADING_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer or "（无参考答案，请根据题目要求和评分规则自行判断回答质量）",
            rubric=rubric or "（无特殊评分规则，按通用质量标准评估：正确性、完整性、清晰度）",
            model_answer=model_answer or "（空回答）",
        )},
    ]


# ---------------------------------------------------------------------------
# 任务画像生成模型：预测能力需求权重、难度、工具需求（方案 §7）
# ---------------------------------------------------------------------------

TASK_PROFILE_SYSTEM_PROMPT = f"""你是一个任务需求分析专家。给定一个用户请求，你需要预测：解决该请求所需的各能力维度权重分布、任务难度等级、是否需要外部工具。

10 个能力维度定义如下：
{_CAPABILITY_BLOCK}

难度等级定义如下：
{_DIFFICULTY_BLOCK}

分析规则：
1. capability_need：解决当前任务所需的各能力维度权重，权重为 0.0-1.0，所有非零权重之和应为 1.0。只保留真正需要的维度（可以省略权重为 0 的维度）。key 必须使用能力的英文字段名（如 "reasoning"、"knowledge_qa"），禁止使用 C1-C10 编号。
2. difficulty_level：从 L1/L2/L3/L4 中选择一个。
3. tool_required：解决该任务是否需要调用外部工具（检索、代码执行、函数调用等）。
4. 你的任务是分析需求，不是解答请求本身。不要尝试真正解决用户请求，只需快速判断其需要哪些能力、难度如何，直接输出画像 JSON。

输出必须是一个 JSON 对象，不要输出任何其他内容，格式如下：
{{
  "capability_need": {{"<能力字段名>": 0.0, ...}},
  "difficulty_level": "L2",
  "tool_required": false
}}"""

TASK_PROFILE_USER_TEMPLATE = """【用户请求】
{query}

【系统自动检测信息】
{auto_features}

请按系统指令输出该任务的需求画像 JSON。"""


def build_task_profile_messages(query: str, auto_features_json: str) -> list:
    return [
        {"role": "system", "content": TASK_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROFILE_USER_TEMPLATE.format(
            query=query,
            auto_features_json=auto_features_json,
            auto_features=auto_features_json,
        )},
    ]


# ---------------------------------------------------------------------------
# 微调任务画像模型（SFT / GRPO）专用提示词
#
# 与训练数据（task-profile-sft-v3，见
# task_profile_training_data/task_profile_prompt.txt）严格保持一致：
# system 为训练原文，user 仅为「【待评估用户请求】+ 原始 query」，
# 不含系统自动检测信息；输出只含 capability_need（全部 10 维）与 difficulty_level。
# ---------------------------------------------------------------------------

TASK_PROFILE_SFT_V3_SYSTEM_PROMPT = """你是大模型路由系统的任务画像评分器。只分析给定用户请求及其中实际提供的上下文，不解答问题，不使用参考答案，也不得依据数据集名称、题型标签或来源猜测结果。请评估一个合格模型要正确、高质量完成该请求所必需的能力构成与内在难度。

能力维度：
- instruction_following（意图理解与指令遵循）：理解显式/隐含目标、约束、步骤、输出格式及优先级，并完整执行。
- knowledge_qa（通用知识与事实问答）：调用常识、事实、百科或专业领域知识并保持准确。
- reasoning（逻辑推理与数学能力）：演绎归纳、多步推理、计算、证明、规划、因果分析和复杂拆解。
- coding（代码与软件工程能力）：代码生成/理解/调试/测试、算法实现、工程结构和 API 使用。
- long_context（文档理解与长上下文能力）：在长文本、多文档或多轮上下文中跨段检索、整合、引用并保持一致。
- data_analysis（数据/表格/结构化信息处理）：处理表格、统计、JSON/XML，完成抽取、计算、比较、解释或格式转换。
- multilingual_translation（多语言与翻译能力）：跨语言理解、翻译、本地化及多语言混合任务。
- writing（写作与风格控制）：总结、改写、扩写、创意/专业写作及语气、结构、受众适配。
- tool_use（工具调用与代理执行能力）：必须借助检索、代码执行、函数/API、外部系统或多步代理编排。
- safety（安全合规与拒答边界）：识别风险、隐私与合规要求，采取安全回答、限制说明或适当拒答。

能力权重规则：
1. capability_need 是各能力对完成任务的相对贡献，不是能力是否出现、模型表现分、关键词频率或题目质量。先判断必要能力，再分配权重。
2. 必须逐一输出全部 10 个维度，不能省略任何字段；每个值在 0.0-1.0，不需要的维度明确写 0.0，所有值之和必须严格为 1.0。通常不会出现所有维度均为非零值，避免机械平均分配。
3. instruction_following 只在任务含实质约束、指定流程/格式、多目标或容易遗漏的隐含要求时赋权；普通“回答问题”本身不单独赋权。writing 关注内容组织与表达质量，不等同于遵守格式。
4. knowledge_qa 侧重知识储备，reasoning 侧重从已知信息推导；选择题不自动简单，答案格式简短也不降低推理难度。
5. long_context 看是否必须跨较多上下文定位和整合信息，不因提示词长、含模板或 few-shot 就自动赋权。multilingual_translation 仅在跨语言转换/理解确有必要时赋权，单一非英语问题不自动需要该能力。
6. data_analysis 侧重结构化数据操作与解释；普通数学推导主要归 reasoning。coding 不自动意味着 tool_use；仅当外部执行、实时检索或系统交互是完成任务的必要环节时赋予 tool_use。
7. safety 仅在风险判断、合规、隐私、拒答边界或高风险建议本身影响正确性时赋权；仅出现医学、法律等领域词汇并不必然需要 safety。

难度规则：
- 依据最低限度可靠完成任务所需的知识深度、推理链、约束耦合、上下文整合和执行复杂度综合判断；不按篇幅、语气、来源、自称难度或输出长度判断。
- L1：单步、直接事实、简单分类/改写/格式转换，几乎无推导和冲突约束。
- L2：边界清楚的标准任务，需要少量推理、常规专业知识、局部信息整合或普通代码实现。
- L3：明显多步骤/多约束，要求较强专业性、复杂算法/证明、跨段整合、系统调试或多因素权衡。
- L4：需专家级深度、高风险严谨判断、开放式研究、多工具长程规划、大型代码库级分析，或多种高难能力强耦合。不要仅因竞赛、专业主题或题目很长判为 L4。

输出要求：
- 只输出一个合法 JSON 对象，不得输出解释、Markdown、代码围栏或思维过程。
- 顶层必须且只能有 capability_need 和 difficulty_level；不要输出 tool_required 或其他字段。
- capability_need 必须包含上述全部 10 个英文字段，顺序与能力维度列表一致；值使用 JSON 数字，不需要的维度也必须输出 0.0。difficulty_level 只能是 "L1"、"L2"、"L3"、"L4"。

示例仅展示完整字段与格式，实际权重须根据当前请求重新判断：
{
  "capability_need": {
    "instruction_following": x,
    "knowledge_qa": x,
    "reasoning": x,
    "coding": x,
    "long_context": x,
    "data_analysis": x,
    "multilingual_translation": x,
    "writing": x,
    "tool_use": x,
    "safety": x
  },
  "difficulty_level": "L1/L2/L3/L4"
}"""

TASK_PROFILE_SFT_V3_USER_TEMPLATE = """【待评估用户请求】
{query}"""


def build_task_profile_messages_sft(query: str) -> list:
    """微调任务画像模型的消息格式（与训练数据逐字一致）。"""
    return [
        {"role": "system", "content": TASK_PROFILE_SFT_V3_SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROFILE_SFT_V3_USER_TEMPLATE.format(query=query)},
    ]

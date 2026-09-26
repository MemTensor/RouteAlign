"""
适配模块提示词：题库元数据与能力权重标注（强模型）。

能力维度与难度等级描述统一引用 demo_router.capabilities 中的权威定义
（纯数据定义，非 LLM 调用模块），保证与 Demo Router 内部提示词完全一致。
"""

from src.capabilities import (
    CAPABILITY_FIELDS,
    format_capability_descriptions,
    format_difficulty_descriptions,
)

_CAPABILITY_BLOCK = format_capability_descriptions()
_DIFFICULTY_BLOCK = format_difficulty_descriptions()
_CAPABILITY_JSON_KEYS = ", ".join(f'"{k}"' for k in CAPABILITY_FIELDS)


METADATA_SYSTEM_PROMPT = f"""你是一名大模型评测题库的资深命题与标注专家。给定一道评测题目（以及可能存在的参考答案），你需要为该题目生成完整的结构化元数据和能力考察权重。

10 个能力维度定义如下（标注必须严格使用这套定义）：
{_CAPABILITY_BLOCK}

难度等级定义如下：
{_DIFFICULTY_BLOCK}

标注要求：
1. capability_weights：该题目对每个能力维度的考察权重（0.0-1.0），表示解决该题目对各能力的依赖程度，所有权重之和为 1.0。不考察的维度权重为 0。
2. difficulty_level：从 L1/L2/L3/L4 中选择。difficulty_score：连续难度分（0.0-1.0），需与难度等级区间一致（L1: 0-0.25, L2: 0.25-0.50, L3: 0.50-0.75, L4: 0.75-1.00）。
3. rubric：为该题目撰写简明的评分规则，说明什么样的回答算高质量回答（正确性标准、必须覆盖的要点、格式要求等）。
4. task_type：简短的任务类型标识（英文 snake_case，如 math_competition、code_generation、knowledge_qa、agent_task）。
5. domain：题目所属领域（英文 snake_case，如 mathematics、software_engineering、physics）。
6. language：题目主要语言（如 zh、en）。
7. tool_required：解决该题是否需要外部工具（检索、代码执行、函数调用等）。
8. multimodal_required：是否需要多模态（图片/音频/视频）输入理解。
9. safety_sensitive：是否涉及安全敏感内容。
10. expected_output_length：预期输出长度（short / medium / long）。
11. answer_format_requirement：回答格式要求（如 boxed_number、multiple_choice_letter、code、markdown、free_text）。

注意：你的任务是标注题目，不是解答题目。不要尝试真正解出题目答案，只需快速判断题目考察哪些能力、难度如何，直接输出标注 JSON。

输出必须是一个 JSON 对象，不要输出任何其他内容，格式如下：
{{
  "task_type": "...",
  "difficulty_level": "L2",
  "difficulty_score": 0.4,
  "capability_weights": {{"<能力字段名>": 0.0, ...（可只列出非零项，字段名必须来自：{_CAPABILITY_JSON_KEYS}）}},
  "rubric": "...",
  "domain": "...",
  "language": "en",
  "tool_required": false,
  "multimodal_required": false,
  "safety_sensitive": false,
  "expected_output_length": "medium",
  "answer_format_requirement": "..."
}}"""

METADATA_USER_TEMPLATE = """【题目】
{question}

【参考答案】
{reference_answer}

请按系统指令为该题目生成结构化元数据与能力权重 JSON。"""


def build_metadata_messages(question: str, reference_answer: str) -> list:
    return [
        {"role": "system", "content": METADATA_SYSTEM_PROMPT},
        {"role": "user", "content": METADATA_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer or "（该题无参考答案，请仅根据题目本身进行标注）",
        )},
    ]

"""
Demo Router 与 LLMRouterBench 的适配模块。

负责：
- results/bench 数据格式转换、训练/测试集划分
- 从训练集分层抽样并调用 LLM 构建符合 Demo Router 要求的题库
- 从既有基准结果导出各模型回答（复用 bench 回答，不重新生成）
- 驱动 Demo Router 离线画像构建
- 测试集路由评估（离线查表取被选模型的 score/cost）
"""

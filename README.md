# LangGraph 多智能体分类作业

## 目标
关键词优先；未命中或跨类别冲突时由兼容 OpenAI Chat Completions API 的模型分类；将用户请求分发到 couplet、joke、travel、other，最后汇总。

> 作业文档写的是 `jev` 模型，但未提供模型 ID、接口地址、密钥和案例。当前使用可配置的 `MODEL_NAME`、`MODEL_BASE_URL`，示例是 DeepSeek，不代表 `jev` 就是 DeepSeek。请根据老师的真实 API 信息修改。

## Windows / VS Code 操作
1. 解压项目，用 VS Code 打开文件夹。
2. 在终端输入 `py -3.11 -m venv .venv`。
3. PowerShell 中运行 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`（无需修改 PowerShell 执行策略）。
4. 复制 `.env.example` 到 `.env`，填写自己的 API Key、模型 API base URL 和模型 ID。不要公开密钥。
5. 运行 `.\.venv\Scripts\python.exe -m unittest -v` 执行离线测试。
6. 运行 `.\.venv\Scripts\python.exe main.py`，按提示输入问题，`#exit` 退出。
7. 运行时会尝试生成 `workflow.png`，图片导出失败不会阻止工作流执行。

## 测试用例
- `帮我写一副春节对联` -> couplet / keyword
- `讲个笑话` -> joke / keyword
- `给我推荐上海旅游攻略` -> travel / keyword
- `说点有趣的让我开心` -> 预计 joke / llm（依赖模型回答）
- `帮我设计一个三天的北京观光计划` -> 预计 travel / llm（依赖模型回答）
- `什么是 Python` -> other / llm

## 提交参考地址
1. 在 GitHub 创建仓库（例如 `langgraph-classification-homework`）。
2. 上传 `main.py`, `requirements.txt`, `README.md`, `.env.example`, `.gitignore`, `test_classify.py`，以及成功生成时的 `workflow.png`。
3. 不要上传 `.env` 和真实 API Key。
4. 复制仓库 URL 作为个人参考地址，并按老师要求附上测试截图。

## 说明
这里的四个业务分支为了能够独立运行，提供了基础模型回复实现；如果课堂原项目已有这四个节点，可保留原节点，只替换 `classify` 与条件路由即可。

"""LangGraph homework: keyword-first classification, LLM fallback.
Run: python main.py
"""
import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv()
Category = Literal["couplet", "joke", "travel", "other"]

class AgentState(TypedDict, total=False):
    user_input: str
    category: Category
    classify_method: str
    draft: str
    answer: str

KEYWORDS: dict[str, list[str]] = {
    "couplet": ["对联", "上联", "下联", "横批", "春联"],
    "joke": ["笑话", "段子", "搞笑", "逗我", "幽默"],
    "travel": ["旅游", "旅行", "游玩", "景点", "行程", "攻略", "自由行"],
}
VALID_CATEGORIES = {"couplet", "joke", "travel", "other"}


def keyword_classify(query: str) -> Category | None:
    """If different categories match, avoid guessing: ask LLM to disambiguate."""
    matches = [category for category, words in KEYWORDS.items()
               if any(word in query for word in words)]
    if len(matches) == 1:
        return matches[0]  # type: ignore[return-value]
    return None


def get_client() -> OpenAI:
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 MODEL_API_KEY：请在 .env 中填写模型 API Key")
    return OpenAI(api_key=api_key, base_url=os.getenv("MODEL_BASE_URL", "https://api.deepseek.com/v1"), timeout=30.0)


def llm_text(prompt: str, system: str) -> str:
    model = os.getenv("MODEL_NAME", "deepseek-chat")
    result = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return (result.choices[0].message.content or "").strip()


def llm_classify(query: str) -> Category:
    system = (
        "你是一个意图分类器，只能回复一个英文小写标签，不能附加解释："
        "couplet、joke、travel、other。"
        "couplet=对联创作；joke=笑话或幽默娱乐；"
        "travel=旅游攻略或行程；other=其他问题。"
        "请判断用户最主要的需求，模糊或不相关时回复 other。"
    )
    raw = llm_text(query, system).lower().strip()
    # Strict output validation: do not let arbitrary model text select a node.
    cleaned = raw.strip(' \n\t`"\'。.,:：')
    return cleaned if cleaned in VALID_CATEGORIES else "other"  # type: ignore[return-value]


def classify(state: AgentState) -> dict:
    query = state["user_input"].strip()
    category = keyword_classify(query)
    if category is not None:
        return {"category": category, "classify_method": "keyword"}
    try:
        category = llm_classify(query)
        return {"category": category, "classify_method": "llm"}
    except Exception as exc:
        print(f"[提示] 模型分类失败，使用 other 兜底：{exc}")
        return {"category": "other", "classify_method": "fallback_error"}


def route(state: AgentState) -> Category:
    return state["category"]


def generate(state: AgentState, task: str) -> dict:
    prompt = f"用户问题：{state['user_input']}\n任务：{task}"
    try:
        answer = llm_text(prompt, "请使用中文回答，内容清晰、实用、简洁。")
    except Exception as exc:
        answer = f"[生成失败] 请检查 .env 中的 API 配置：{exc}"
    return {"draft": answer}


def couplet(state: AgentState) -> dict:
    return generate(state, "写一副合适的对联，包含上联、下联、横批。")


def joke(state: AgentState) -> dict:
    return generate(state, "讲一个简短、友好、有趣的笑话。")


def travel(state: AgentState) -> dict:
    return generate(state, "提供实用旅行建议；如果缺少目的地，先询问必要信息。")


def other(state: AgentState) -> dict:
    return generate(state, "回答用户的实际问题；不清楚时要求澄清。")


def synthesize(state: AgentState) -> dict:
    return {"answer": state.get("draft", "未生成内容")}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("couplet", couplet)
    graph.add_node("joke", joke)
    graph.add_node("travel", travel)
    graph.add_node("other", other)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, {
        "couplet": "couplet", "joke": "joke", "travel": "travel", "other": "other"
    })
    for node in ("couplet", "joke", "travel", "other"):
        graph.add_edge(node, "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def main():
    app = build_graph()
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png)
        print("已生成工作流图：workflow.png")
    except Exception as exc:
        print(f"工作流可运行，但未能导出图片（可忽略）：{exc}")
    print("LangGraph 分类系统已启动。输入 #exit 退出。")
    while True:
        query = input("\n请输入问题：").strip()
        if query == "#exit":
            break
        if not query:
            continue
        result = app.invoke({"user_input": query})
        print("分类结果：", result["category"])
        print("分类方式：", result["classify_method"])
        print("最终回复：\n", result["answer"])


if __name__ == "__main__":
    main()

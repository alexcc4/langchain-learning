import sys
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 加载环境变量
load_dotenv()

# 确保标准输出使用 UTF-8 编码
sys.stdout.reconfigure(encoding='utf-8')

# 配置
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "xiyouji_collection"
EMBEDDING_MODEL = "qwen3-embedding:latest"  
CHAT_MODEL = "qwen3:latest"  

# 初始化 LLM
response_model = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.2,  # 降低温度使回答更准确
)

grader_model = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,  # 评分时使用确定性输出
)

# 获取向量存储
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# 创建检索器和工具
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_xiyouji",
    "从西游记原文中检索相关信息。使用此工具查找关于人物、情节、地点等内容。",
)

# ============= 节点函数 =============

def generate_query_or_respond(state: MessagesState):
    """
    根据当前状态调用 LLM 生成响应。
    决定是使用检索工具，还是直接回答用户。
    """
    print("📍 节点: 生成查询或响应")
    response = (
        response_model
        .bind_tools([retriever_tool])
        .invoke(state["messages"])
    )
    return {"messages": [response]}


# 文档评分的结构化输出
class GradeDocuments(BaseModel):
    """使用二元评分检查文档相关性。"""
    binary_score: str = Field(
        description="相关性评分: 'yes' 表示相关, 'no' 表示不相关"
    )


GRADE_PROMPT = """你是一个评估检索文档与用户问题相关性的评分员。

检索到的文档内容:
{context}

用户问题:
{question}

如果文档包含与用户问题相关的关键词或语义内容，则评为相关。
给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。
"""


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """判断检索到的文档是否与问题相关。"""
    print("📍 节点: 文档评分")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    # 使用结构化输出进行评分
    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )
    
    score = response.binary_score
    print(f"   相关性评分: {score}")
    
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


REWRITE_PROMPT = """分析输入问题，理解其背后的语义意图和含义。

原始问题:
------- 
{question}
-------

请重新表述一个改进的问题，使其更容易检索到相关信息:
"""


def rewrite_question(state: MessagesState):
    """重写原始用户问题以提高检索效果。"""
    print("📍 节点: 问题重写")
    messages = state["messages"]
    question = messages[0].content
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    print(f"   重写后: {response.content}")
    return {"messages": [{"role": "user", "content": response.content}]}


GENERATE_PROMPT = """你是一个西游记知识问答助手。
根据检索到的内容回答问题。
如果你不知道答案，就说你不知道。
最多使用三句话，保持回答简洁。

问题: {question}

检索到的内容: 
{context}
"""


def generate_answer(state: MessagesState):
    """生成最终答案。"""
    print("📍 节点: 生成答案")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}


# ============= 构建图 =============

workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# 添加边
workflow.add_edge(START, "generate_query_or_respond")

# 条件边：决定是否检索
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # 评估 LLM 决策（调用工具还是直接响应）
    {
        "tools": "retrieve",  # 如果需要调用工具，前往检索节点
        END: END,  # 否则结束
    },
)

# 条件边：评估检索结果
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # 评估文档相关性
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# 编译图
agent = workflow.compile()


def chat(query: str):
    """与 Agentic RAG 系统对话"""
    print(f"\n{'='*60}")
    print(f"❓ 问题: {query}")
    print(f"{'='*60}\n")
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
    ):
        for node, update in chunk.items():
            if node != "__end__":
                print(f"\n--- 来自节点 '{node}' 的更新 ---")
                if update.get("messages"):
                    update["messages"][-1].pretty_print()
    
    print(f"\n{'='*60}\n")


def main():
    print("="*60)
    print("🚀 西游记 Agentic RAG 问答系统")
    print("   (带文档评分与问题重写)")
    print("="*60)
    
    # 检查向量库
    try:
        count = vector_store._collection.count()
        print(f"✓ 向量数据库已加载: {count} 个文档块\n")
    except Exception as e:
        print(f"✗ 向量数据库加载失败: {e}")
        print("请先运行 main.py 索引 PDF 文档")
        return
    
    # 测试问题
    test_queries = [
        "孙悟空是怎么出生的？",
    ]
    
    for query in test_queries:
        chat(query)
        print("\n")
    
    # 交互式问答
    print(f"{'='*60}")
    print("现在可以开始提问了（输入 'quit' 退出）：")
    print(f"{'='*60}\n")
    
    while True:
        try:
            query = input("你的问题: ").strip()
            if query.lower() in ['quit', 'exit', 'q', '退出']:
                print("再见！")
                break
            if not query:
                continue
            chat(query)
        except KeyboardInterrupt:
            print("\n\n再见！")
            break


if __name__ == "__main__":
    main()

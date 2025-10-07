import sys
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent   

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
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.7,
)

# 获取向量存储
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# 创建检索工具
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """从西游记文档中检索相关信息来回答问题。"""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"来源: 第{doc.metadata.get('page', 'N/A')}页\n内容: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

# 创建提示词
prompt = (
    "你是一个专业的西游记知识助手。你可以使用 retrieve_context 工具从西游记原文中检索相关内容。"
    "请基于检索到的内容回答用户问题，如果检索不到相关内容，请告诉用户你不知道。"
    "回答要准确、简洁。"
)

# 创建 Agent
agent = create_react_agent(llm, tools, prompt=prompt)


def chat(query: str):
    """与 RAG 系统对话"""
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"{'='*60}\n")
    
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


def main():
    print("="*60)
    print("西游记 RAG 问答系统")
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
        "唐僧为什么要去西天取经？",
        "猪八戒的前世是什么？",
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

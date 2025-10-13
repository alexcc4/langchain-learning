import sys
import re

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

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
    temperature=0.1,
)

# 获取向量存储
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ============= ReAct Prompt =============

REACT_PROMPT = """你是一个使用 ReAct (Reasoning and Acting) 方法的西游记问答助手。

你需要通过以下循环来回答问题：
Thought（思考）→ Action（行动）→ Observation（观察）→ ... → Answer（最终答案）

**可用的工具：**
- retrieve: 从西游记原文中检索相关信息。输入应该是搜索查询。

**输出格式要求：**
每一步必须严格按照以下格式之一输出：

1. 需要检索时：
Thought: [你的推理过程，解释为什么需要检索以及检索什么]
Action: retrieve
Action Input: [具体的搜索查询]

2. 得出最终答案时：
Thought: [你的推理过程，说明为什么可以给出答案了]
Answer: [最终答案，简洁明了，最多三句话]

**重要规则：**
- 每次只能输出一个 Thought，然后是一个 Action 或 Answer
- 如果选择 Action，必须等待 Observation 结果
- 收到 Observation 后，继续下一个 Thought
- 最多进行 5 次 Action，之后必须给出 Answer
- Answer 必须基于 Observation 的内容，如果信息不足就说不知道

**问题：** {question}

{history}

开始！"""


# ============= 工具执行 =============

def retrieve(query: str) -> str:
    """从西游记中检索信息"""
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关信息"
    
    context = "\n\n".join([f"[片段 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
    return context


# ============= ReAct 循环 =============

def parse_react_output(text: str) -> tuple[str, str, str]:
    """
    解析 ReAct 输出
    返回: (thought, action, action_input) 或 (thought, "answer", answer_text)
    """
    text = text.strip()
    
    # 提取 Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Answer):|$)', text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    
    # 检查是否是最终答案
    answer_match = re.search(r'Answer:\s*(.+)', text, re.DOTALL)
    if answer_match:
        return thought, "answer", answer_match.group(1).strip()
    
    # 提取 Action
    action_match = re.search(r'Action:\s*(\w+)', text)
    action = action_match.group(1).strip() if action_match else ""
    
    # 提取 Action Input
    action_input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', text, re.DOTALL)
    action_input = action_input_match.group(1).strip() if action_input_match else ""
    
    return thought, action, action_input


def react_agent(question: str, max_steps: int = 5, verbose: bool = True):
    """ReAct Agent 主循环"""
    
    history = ""
    
    for step in range(max_steps):
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 第 {step + 1} 步")
            print(f"{'='*60}")
        
        # 1. LLM 生成 Thought + Action/Answer
        prompt = REACT_PROMPT.format(question=question, history=history)
        response = llm.invoke([{"role": "user", "content": prompt}])
        output = response.content
        
        if verbose:
            print(f"\n🤖 LLM 输出:\n{output}")
        
        # 2. 解析输出
        thought, action, action_input = parse_react_output(output)
        
        if not thought:
            if verbose:
                print("\n⚠️  警告：无法解析 Thought，重试...")
            continue
        
        # 3. 如果是最终答案，返回
        if action == "answer":
            if verbose:
                print(f"\n💡 思考: {thought}")
                print(f"\n✅ 最终答案:\n{action_input}")
            return action_input
        
        # 4. 执行 Action
        if action == "retrieve":
            if verbose:
                print(f"\n💡 思考: {thought}")
                print(f"\n🔍 执行检索: {action_input}")
            
            observation = retrieve(action_input)
            
            if verbose:
                print(f"\n📚 观察结果:\n{observation[:300]}..." if len(observation) > 300 else f"\n📚 观察结果:\n{observation}")
            
            # 5. 更新历史
            history += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
        else:
            if verbose:
                print(f"\n⚠️  未知的 Action: {action}")
            history += f"\nThought: {thought}\n[错误：未知的 Action '{action}']\n"
    
    # 达到最大步数
    if verbose:
        print(f"\n⚠️  已达到最大步数 ({max_steps})，强制生成答案...")
    
    final_prompt = REACT_PROMPT.format(
        question=question, 
        history=history + "\n你必须现在给出最终答案（使用 Answer: 格式）"
    )
    response = llm.invoke([{"role": "user", "content": final_prompt}])
    thought, action, answer = parse_react_output(response.content)
    
    if action == "answer":
        return answer
    else:
        return "抱歉，无法在限定步骤内得出答案。"


# ============= 主程序 =============

def main():
    print("="*60)
    print("🧠 西游记 ReAct 问答系统")
    print("   (Reasoning and Acting)")
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
        "唐僧师徒一共有几个人？",
    ]
    
    for query in test_queries:
        print(f"\n\n{'#'*60}")
        print(f"❓ 问题: {query}")
        print(f"{'#'*60}")
        
        answer = react_agent(query, max_steps=5, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"📝 最终答案: {answer}")
        print(f"{'='*60}\n")
    
    # 交互式问答
    print(f"\n{'='*60}")
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
            
            print(f"\n{'#'*60}")
            print(f"❓ 问题: {query}")
            print(f"{'#'*60}")
            
            answer = react_agent(query, max_steps=5, verbose=True)
            
            print(f"\n{'='*60}")
            print(f"📝 最终答案: {answer}")
            print(f"{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break


if __name__ == "__main__":
    main()


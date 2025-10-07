# LangChain 学习项目 - 中文 PDF 语义搜索与 RAG 问答

基于 LangChain 实现的中文 PDF 文档语义搜索引擎和 RAG 智能问答系统，使用 Ollama 本地模型和 ChromaDB 向量数据库。

## 功能特性

- 📄 加载和处理中文 PDF 文档
- 🔍 基于语义的文档搜索
- 💬 **RAG 智能问答系统**（基于检索增强生成）
- 🚀 **Agentic RAG** - 带文档评分与智能问题重写
- 🤖 使用 Ollama 本地模型 (qwen3-embedding + qwen3:latest)
- 💾 ChromaDB 向量数据库持久化存储
- 🐳 Docker 部署 ChromaDB
- 🔄 Agent 自动检索和回答

## 环境要求

- Python 3.12+
- Docker
- Ollama (需要安装 qwen3-embedding 和 qwen3:latest 模型)

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或者使用 pip
pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`（如果还没有的话）：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置 Ollama 地址（默认是 localhost:11434）。

### 3. 启动 ChromaDB

```bash
# 启动 ChromaDB 容器
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f chromadb
```

详细的 Docker 使用说明请查看 [DOCKER.md](DOCKER.md)。

### 4. 准备 Ollama 模型

确保已经安装了 Ollama 并拉取所需模型：

```bash
# 拉取 embedding 模型
ollama pull qwen3-embedding

# 拉取 chat 模型（RAG 问答系统需要）
ollama pull qwen3:latest

# 验证模型
ollama list
```

### 5. 运行程序

#### 索引 PDF 文档

```bash
python main.py
```

程序会：

1. 批量加载 `data/西游记.pdf`（每次 10 页）
2. 分割文档为小块
3. 生成 embeddings
4. 存储到 ChromaDB
5. 显示实时进度
6. 执行示例查询

**断点续传**: 如果中途按 `Ctrl+C` 中断，重新运行会从上次位置继续。

#### 运行 Agentic RAG 问答系统

```bash
python rag_app.py
```

**前置条件**:

- 已运行 `main.py` 完成文档索引
- 已安装 `qwen3:latest` 模型 (`ollama pull qwen3:latest`)

系统会自动执行测试问题，然后进入交互模式。输入 `quit` 退出。

## 项目结构

```bash
.
├── data/                   # PDF 文档目录
│   └── 西游记.pdf
├── chroma_data/           # ChromaDB 数据持久化目录
├── main.py                # 索引程序（文档处理与向量化）
├── rag_app.py             # Agentic RAG 问答系统
├── docker-compose.yml     # Docker Compose 配置
├── DOCKER.md             # Docker 使用说明
├── .env                  # 环境变量配置（不提交到 git）
├── .env.example          # 环境变量示例
├── pyproject.toml        # Python 项目配置
└── README.md             # 本文件
```

## 使用说明

### 核心特性

`main.py` 已集成以下功能（无感使用）：

- ✅ **批量处理**: 每次处理 10 页，不会爆内存
- ✅ **断点续传**: `Ctrl+C` 中断后重新运行自动继续
- ✅ **自动去重**: 重复运行不会产生冗余数据
- ✅ **进度保存**: 实时保存到 `chroma_data/progress.json`

### 基本使用

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 加载向量存储
embeddings = OllamaEmbeddings(model="qwen3-embedding")
vector_store = Chroma(
    collection_name="xiyouji_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_data",
)

# 搜索
results = vector_store.similarity_search("孙悟空", k=3)
for doc in results:
    print(doc.page_content)
```

### 带相似度分数的搜索

```python
results = vector_store.similarity_search_with_score("唐僧取经", k=3)
for doc, score in results:
    print(f"相似度: {score}")
    print(doc.page_content)
```

### 使用 Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
results = retriever.invoke("猪八戒")
```

### 使用 Agentic RAG 智能问答系统

直接运行：

```bash
python rag_app.py
```

或在代码中使用：

```python
from rag_app import chat

chat("孙悟空是怎么出生的？")
chat("唐僧为什么要去西天取经？")
chat("猪八戒的前世是什么？")
```

#### Agentic RAG 特性

与传统 RAG 不同，Agentic RAG 系统具有以下智能特性：

1. **文档相关性评分** - 自动评估检索到的文档是否真正相关
2. **智能问题重写** - 当文档不相关时，自动重写问题并重新检索
3. **条件路由** - 根据评分结果智能决定下一步操作
4. **流程可视化** - 打印每个节点的执行过程，便于调试

**工作流程**：

```text
用户提问 
  → 生成查询或响应（决定是否检索）
  → 检索文档
  → 文档评分
  ├─ 相关 → 生成答案
  └─ 不相关 → 重写问题 → 重新检索
```

## 注意事项

1. **中文编码**: 代码中已经处理了 UTF-8 编码问题，确保中文正确显示
2. **模型选择**: 默认使用 `qwen3-embedding`，可以根据需要修改为其他模型
3. **chunk_size**: 中文文档建议使用较小的 chunk_size (500)，可根据实际情况调整
4. **数据持久化**: 向量数据保存在 `chroma_data` 目录，删除该目录可重新生成

## 清理数据

如果需要重新生成向量存储：

```bash
# 删除向量数据
rm -rf chroma_data

# 重新运行程序
python main.py
```

## 停止服务

```bash
# 停止 ChromaDB
docker-compose stop

# 停止并删除容器
docker-compose down
```

## 参考资料

- [LangChain 文档](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)

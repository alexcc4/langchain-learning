import sys
import json
from pathlib import Path
import hashlib
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

# 确保标准输出使用 UTF-8 编码
sys.stdout.reconfigure(encoding='utf-8')

# 配置
PDF_PATH = "data/西游记.pdf"
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "xiyouji_collection"
EMBEDDING_MODEL = "qwen3-embedding"
BATCH_SIZE = 10  # 每批处理的页数
PROGRESS_FILE = "./chroma_data/progress.json"

# 文本分割器 - 中文文档使用较小的 chunk_size 更合适
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)


def get_document_id(doc: Document) -> str:
    """生成文档唯一 ID（防止重复）"""
    content = f"{doc.page_content}|{doc.metadata.get('source')}|{doc.metadata.get('page')}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def load_progress():
    """加载处理进度"""
    progress_path = Path(PROGRESS_FILE)
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_pages": 0, "total_pages": 0, "processed_chunks": 0}


def save_progress(progress):
    """保存处理进度"""
    progress_path = Path(PROGRESS_FILE)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def process_batch(
    pages: List[Document],
    vector_store: Chroma,
    start_page: int
) -> int:
    """批量处理页面"""
    if not pages:
        return 0
    
    print(f"\n处理页面 {start_page} - {start_page + len(pages) - 1} ({len(pages)} 页)")
    
    # 分割文档
    splits = text_splitter.split_documents(pages)
    print(f"  分割成 {len(splits)} 个文档块")
    
    if not splits:
        return 0
    
    # 生成唯一 ID（自动去重）
    ids = [get_document_id(split) for split in splits]
    
    # 批量添加
    print(f"  正在生成 embeddings 并存储...")
    try:
        vector_store.add_documents(documents=splits, ids=ids)
        print(f"  ✓ 成功添加 {len(splits)} 个文档块")
    except Exception as e:
        print(f"  ✗ 添加失败: {e}")
        raise
    
    return len(splits)


def load_and_index_pdf(pdf_path: str, batch_size: int = BATCH_SIZE):
    """
    增量加载和索引 PDF（支持断点续传）
    """
    # 创建或加载向量存储
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    
    # 加载进度
    progress = load_progress()
    start_page = progress.get("processed_pages", 0)
    
    # 加载 PDF
    print(f"正在加载 PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    all_pages = list(loader.lazy_load())  # 懒加载
    total_pages = len(all_pages)
    print(f"✓ PDF 共 {total_pages} 页")
    
    # 检查是否已完成
    if start_page >= total_pages:
        print(f"✓ 所有页面已处理完成（共 {progress['processed_chunks']} 个文档块）")
        return vector_store
    
    # 显示继续处理信息
    if start_page > 0:
        print(f"📍 从第 {start_page} 页继续（已处理 {progress['processed_chunks']} 个文档块）")
    
    # 批量处理
    progress["total_pages"] = total_pages
    current_page = start_page
    total_chunks = progress.get("processed_chunks", 0)
    
    try:
        while current_page < total_pages:
            end_page = min(current_page + batch_size, total_pages)
            batch = all_pages[current_page:end_page]
            
            # 处理这一批
            chunks_added = process_batch(batch, vector_store, current_page)
            total_chunks += chunks_added
            
            # 更新进度
            current_page = end_page
            progress["processed_pages"] = current_page
            progress["processed_chunks"] = total_chunks
            save_progress(progress)
            
            # 显示进度
            percent = (current_page / total_pages) * 100
            print(f"  📊 进度: {current_page}/{total_pages} ({percent:.1f}%) - 累计 {total_chunks} 个文档块")
        
        print(f"\n✓ 所有页面处理完成！共 {total_chunks} 个文档块")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断！进度已保存")
        print(f"   已处理到第 {current_page} 页，下次运行将继续")
        return vector_store
    
    return vector_store


def get_vector_store():
    """获取现有的向量存储"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


def search_documents(vector_store, query, k=3):
    """搜索相关文档"""
    print(f"\n查询: {query}")
    results = vector_store.similarity_search_with_score(query, k=k)
    
    print(f"找到 {len(results)} 个相关结果:\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- 结果 {i} (相似度: {score:.4f}) ---")
        print(f"内容: {doc.page_content[:150]}...")
        print(f"页码: {doc.metadata.get('page', 'N/A')}")
        print()
    
    return results


def main():
    print("="*60)
    print("PDF 语义搜索引擎")
    print("="*60)
    
    # 加载和索引（自动支持断点续传）
    vector_store = load_and_index_pdf(PDF_PATH)
    
    # 获取统计信息
    try:
        count = vector_store._collection.count()
        print(f"\n📊 向量数据库统计: {count} 个文档块")
    except:
        pass
    
    # 测试搜索
    print("\n" + "="*60)
    print("测试搜索")
    print("="*60)
    
    search_documents(vector_store, "孙悟空的来历", k=2)
    search_documents(vector_store, "唐僧取经", k=2)
    
    print("\n" + "="*60)
    print("💡 提示:")
    print("  - 程序支持断点续传，Ctrl+C 中断后重新运行会继续处理")
    print("  - 自动去重，重复运行不会产生冗余数据")
    print("  - 如需重新索引，删除 chroma_data 目录")
    print("="*60)


if __name__ == "__main__":
    main()

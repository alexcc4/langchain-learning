#!/usr/bin/env python3
"""检查环境配置是否正确"""

import sys
import subprocess
from pathlib import Path

def check_mark(passed):
    """返回对应的标记"""
    return "✓" if passed else "✗"

def check_docker():
    """检查 Docker 是否运行"""
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def check_chromadb():
    """检查 ChromaDB 容器是否运行"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=langchain-chromadb", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "langchain-chromadb" in result.stdout
    except Exception:
        return False

def check_ollama():
    """检查 Ollama 是否运行"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout
    except Exception:
        return False, ""

def check_ollama_model(model_name="qwen3-embedding"):
    """检查 Ollama 模型是否已安装"""
    success, output = check_ollama()
    if not success:
        return False
    return model_name in output

def main():
    print("="*60)
    print("环境配置检查")
    print("="*60)
    
    # 检查 Python 版本
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 12
    print(f"\n{check_mark(py_ok)} Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("  ⚠️  需要 Python 3.12+")
    
    # 检查必要文件
    files_to_check = {
        ".env": Path(".env"),
        "data/西游记.pdf": Path("data/西游记.pdf"),
        "docker-compose.yml": Path("docker-compose.yml"),
    }
    
    print("\n文件检查:")
    all_files_ok = True
    for name, path in files_to_check.items():
        exists = path.exists()
        all_files_ok = all_files_ok and exists
        print(f"  {check_mark(exists)} {name}")
        if not exists:
            print(f"     ⚠️  文件不存在: {path}")
    
    # 检查 Docker
    print("\nDocker 检查:")
    docker_ok = check_docker()
    print(f"  {check_mark(docker_ok)} Docker 运行状态")
    if not docker_ok:
        print("     ⚠️  请启动 Docker")
    
    chromadb_ok = check_chromadb()
    print(f"  {check_mark(chromadb_ok)} ChromaDB 容器")
    if not chromadb_ok:
        print("     ⚠️  请运行: docker-compose up -d")
    
    # 检查 Ollama
    print("\nOllama 检查:")
    ollama_ok, ollama_output = check_ollama()
    print(f"  {check_mark(ollama_ok)} Ollama 运行状态")
    if not ollama_ok:
        print("     ⚠️  请安装并启动 Ollama")
    
    if ollama_ok:
        model_ok = check_ollama_model("qwen3-embedding")
        print(f"  {check_mark(model_ok)} qwen3-embedding 模型")
        if not model_ok:
            print("     ⚠️  请运行: ollama pull qwen3-embedding")
        
        if ollama_output:
            print("\n  已安装的模型:")
            for line in ollama_output.strip().split('\n')[1:]:  # 跳过标题行
                if line.strip():
                    print(f"    - {line.split()[0]}")
    
    # 总结
    print("\n" + "="*60)
    all_ok = py_ok and all_files_ok and docker_ok and chromadb_ok and ollama_ok and model_ok
    if all_ok:
        print("✓ 所有检查通过！可以运行 main.py")
    else:
        print("⚠️  部分检查未通过，请根据上述提示进行配置")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())


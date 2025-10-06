# Docker 使用说明

## ChromaDB 向量数据库

### 启动服务

```bash
# 使用 docker-compose 启动
docker-compose up -d

# 或者直接使用 docker 命令
docker run -d \
  --name langchain-chromadb \
  -p 8000:8000 \
  -v $(pwd)/chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest
```

### 查看状态

```bash
# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f chromadb
```

### 停止服务

```bash
# 停止服务
docker-compose stop

# 停止并删除容器
docker-compose down

# 停止并删除容器和数据卷（慎用！会删除所有数据）
docker-compose down -v
```

### 访问

- ChromaDB API: http://localhost:8000
- 数据持久化目录: `./chroma_data`

### 注意事项

1. 确保 Docker 已启动
2. 确保端口 8000 未被占用
3. `chroma_data` 目录会自动创建，用于持久化存储向量数据
4. 如果需要清空数据，删除 `chroma_data` 目录即可


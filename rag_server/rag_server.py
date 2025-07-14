import os
import torch
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Dict, Union
from rag_schemas import SearchRequest, ScrapeRequest, SearchResult, ScrapeResult
from rag_config import RAGConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定使用的GPU设备

load_dotenv()


def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. \
        Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


# 任务描述
TASK = "Given a web search query, retrieve relevant passages that answer the query"

config = RAGConfig()

print("正在加载模型和元数据...")


print(f"正在连接Milvus: {config.milvus_endpoint}")
client = MilvusClient(uri=config.milvus_endpoint, token=config.milvus_token)

# 检查Collection是否存在
if not client.has_collection(collection_name=config.milvus_collection_name):
    raise SystemExit(
        f"错误: Milvus中不存在名为 '{config.milvus_collection_name}' 的集合。请先运行`build_datasets.ipynb`脚本来创建集合。"
    )

ml_models: Dict[str, Union[SentenceTransformer, CrossEncoder]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期事件，用于在应用启动时初始化模型。
    """
    print("正在初始化模型...")
    # 加载用于生成查询向量的嵌入模型
    ml_models["embedding_model"] = SentenceTransformer(
        f"Qwen/{config.embedding_model_name}",
        cache_folder=f"{config.cache_dir}/{config.embedding_model_name}",
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda",
    )

    # 加载用于重排序的Cross-Encoder模型
    ml_models["reranker_model"] = CrossEncoder(
        f"tomaarsen/{config.reranker_model_name}",
        cache_folder=f"{config.cache_dir}/{config.reranker_model_name}",
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={
            "padding_side": "left",
        },
        device="cuda",
    )
    # 在这里可以添加其他初始化逻辑
    yield  # 应用运行时保持活跃
    ml_models.clear()
    print("应用关闭，清理资源...")  # 应用关闭时的清理逻辑


app = FastAPI(
    title="shallow-research RAG Server",
    description="一个模拟搜索引擎和网页抓取功能的RAG服务，用于复现Jan-nano训练环境。",
    lifespan=lifespan,
)


@app.post("/retrieve", response_model=list[SearchResult], summary="模拟搜索引擎")
async def web_search(request: SearchRequest):
    """
    接收一个查询，执行两阶段检索（向量检索+重排序），返回最相关的文档片段。
    """
    # **向量检索 (Dense Retrieval)**
    # 为查询生成向量
    if isinstance(request.queries, str):
        query_list = [request.queries]
    query_vector = ml_models["embedding_model"].encode_query(
        query_list, normalize_embeddings=True
    )

    # 在Milvus中进行批量搜索
    # results 的长度和 request.queries 的长度相同
    results = client.search(
        collection_name=config.milvus_collection_name,
        data=query_vector,
        anns_field="vector",  # 您在Milvus中定义的向量字段名
        search_params={"metric_type": "IP", "params": {}},
        limit=config.top_k_retrieval,  # 返回的文档数量
        output_fields=["text", "title", "id"],  # 输出字段包括文本和ID和标题
    )

    # 针对每一个查询去重
    unique_docs = []
    for result in results:
        temp = []
        seen_titles = set()
        for hit in result:
            title = hit["entity"]["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                temp.append(
                    {
                        "id": hit["entity"]["id"],  # 保存原始ID
                        "title": title,
                        "text": hit["entity"]["text"],
                    }
                )
        unique_docs.append(unique_docs)

    # 如果没有找到独特的文档，直接返回空列表
    if not unique_docs:
        return []

    query_for_reranker = [format_queries(query, TASK) for query in query_list]
    documents_for_reranker = [
        format_document(doc["text"]) for docs in unique_docs for doc in docs
    ]

    # **重排序 (Reranking)
    # 使用Cross-Encoder模型计算相关性分数
    ranks = [
        ml_models["reranker_model"].rank(
            query=query,
            documents=doc,
        )
        for query, doc in zip(query_for_reranker, documents_for_reranker)
    ]

    # 格式化并返回最终结果
    final_results = []
    for rank in ranks:
        temp = []
        # 只处理分数最高的TOP_K_RERANKED个结果
        for rank_info in rank[: config.top_k_rerank]:
            # `corpus_id`是reranker输入列表的索引，也对应我们`unique_docs`列表的索引
            corpus_id = rank_info["corpus_id"]

            # 从我们保存的原始文档列表中获取完整信息
            original_doc = unique_docs[corpus_id]

            # 构建最终返回的SearchResult对象
            temp.append(
                SearchResult(
                    id=original_doc["id"],
                    title=original_doc["title"],
                    # 从原始文本生成预览
                    preview=original_doc["text"][:150] + "...",
                    rerank_score=rank_info["score"],
                )
            )
        final_results.append(temp)

    return final_results


@app.post("/scrape", response_model=ScrapeResult, summary="模拟网页抓取")
async def scrape_document(request: ScrapeRequest):
    """
    根据文档ID，返回该文档的完整内容。
    """
    doc_id = request.doc_id
    results = client.query(
        collection_name=config.milvus_collection_name,
        ids=[doc_id],
        output_fields=["text", "title"],
    )
    if not results:
        # 如果查询结果为空，说明该ID在数据库中不存在
        raise HTTPException(
            status_code=404, detail=f"文档ID {doc_id} 在数据库中未找到。"
        )
    doc = results[0]
    return ScrapeResult(id=doc_id, title=doc["title"], full_text=doc["text"])


@app.get("/", summary="服务状态检查")
def read_root():
    return {
        "status": "RAG Service is running",
        "milvus_collection": config.milvus_collection_name,
    }


# 启动服务 ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# 加载 .env 环境变量
load_dotenv()

class RAGConfig:
    """
    存储配置参数（路径、模型名称、Milvus信息等）
    """
    cache_dir = "./models"
    embedding_model_name = "Qwen3-Embedding-0.6B"
    reranker_model_name = "Qwen3-Reranker-0.6B-seq-cls"
    milvus_collection_name = "Musique"
    top_k_retrieval = 50
    top_k_rerank = 10

    # 从环境变量中读取 Milvus 地址和 Token
    milvus_endpoint = os.getenv("MILVUS_ENDPOINT")
    milvus_token = os.getenv("MILVUS_TOKEN")

    @classmethod
    def validate(cls):
        if not cls.milvus_endpoint or not cls.milvus_token:
            raise ValueError("请确保 .env 文件中已设置 MILVUS_ENDPOINT 和 MILVUS_TOKEN")


class RAGModelRegistry:
    """
    用于初始化和持久化嵌入模型和重排序模型
    """
    embedding_model = None
    reranker_model = None

    @classmethod
    def init_models(cls):
        """
        初始化嵌入模型和重排序模型，只执行一次
        """
        if cls.embedding_model is None:
            print("正在加载嵌入模型...")
            cls.embedding_model = SentenceTransformer(
                f"Qwen/{RAGConfig.EMBEDDING_MODEL_NAME}",
                cache_folder=f"{RAGConfig.CACHE_DIR}/{RAGConfig.EMBEDDING_MODEL_NAME}",
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.bfloat16,
                },
                tokenizer_kwargs={"padding_side": "left"},
                device="cuda",
            )

        if cls.reranker_model is None:
            print("正在加载重排序模型...")
            cls.reranker_model = CrossEncoder(
                f"tomaarsen/{RAGConfig.RERANKER_MODEL_NAME}",
                cache_folder=f"{RAGConfig.CACHE_DIR}/{RAGConfig.RERANKER_MODEL_NAME}",
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.bfloat16,
                },
                tokenizer_kwargs={"padding_side": "left"},
                device="cuda",
            )

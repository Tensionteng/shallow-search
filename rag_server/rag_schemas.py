from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., description="用户的搜索查询语句。")
    top_k: int = Field(50, description="返回的最相关文档数量。")


class ScrapeRequest(BaseModel):
    doc_id: str = Field(..., description="要抓取全文的文档ID。")


class SearchResult(BaseModel):
    id: str
    title: str
    preview: str
    rerank_score: float


class ScrapeResult(BaseModel):
    id: str
    title: str
    full_text: str

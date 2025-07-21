import requests
from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库

mcp = FastMCP("LocalServer")
RAG_ADRESS = "10.10.100.62:8000"  # RAG服务地址

@mcp.tool()
def query_rag(query: str, topk: int = 50):
    """MCP RAG Query Tool (Synchronous Version)

    Args:
        query: query text
        topk: The default number of documents returned is 10

    Returns:
        str: The formatted query result
    """
    try:
        # 构建请求数据
        request_data = {"query": [query], "topk": topk, "return_scores": True}
        # 设置请求头和代理
        headers = {"Content-Type": "application/json"}
        # 使用本地连接，绕过代理
        proxies = {"http": None, "https": None}

        response = requests.post(
            f"http://{RAG_ADRESS}/retrieve",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10,
        )

        response.raise_for_status()

        # 解析响应
        result = response.json()

        return result

    except requests.exceptions.Timeout:
        return "⚠️ RAG service request timeout, please check if the service is running properly"
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to RAG service, please ensure that the service is running"
    except requests.exceptions.RequestException as e:
        return f"⚠️ RAG service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') else 'No detail'}"
    except Exception as e:
        return f"⚠️ RAG query failed: {str(e)}\nError type: {type(e).__name__}"


@mcp.tool()
def scrape_rag(id: str):
    """MCP RAG Query Tool (Synchronous Version)

    Args:
        id: Document ID to scrape

    Returns:
        str: The formatted scrape result
    """
    try:
        # 构建请求数据
        request_data = {
            "id": [id],
        }
        # 设置请求头和代理
        headers = {"Content-Type": "application/json"}
        # 使用本地连接，绕过代理
        proxies = {"http": None, "https": None}

        response = requests.post(
            f"http://{RAG_ADRESS}/scrape",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10,
        )

        response.raise_for_status()

        # 解析响应
        result = response.json()

        return result

    except requests.exceptions.Timeout:
        return "⚠️ RAG service request timeout, please check if the service is running properly"
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to RAG service, please ensure that the service is running"
    except requests.exceptions.RequestException as e:
        return f"⚠️ RAG service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') else 'No detail'}"
    except Exception as e:
        return f"⚠️ RAG query failed: {str(e)}\nError type: {type(e).__name__}"


if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport="stdio")

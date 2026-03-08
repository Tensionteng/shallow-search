# 从头实现deep-search

本项目灵感来源于[Jan Nano 4B](https://menloresearch.github.io/deep-research/)，由于其技术报告的细节比较少，我花了很多时间和功夫寻找相关教程，[RL-Factory](https://github.com/Simple-Efficient/RL-Factory)挺不错的，但是search-r1还是太耗费资源了，对于我这样的学生党来说实在是力不从心（光是faiss-gpu索引就高达65GB），所以我打算自己训练一个不那么好但是效果还说得过去的模型，同时提升自己的理解。

相比 Search-R1，本项目的优势：
- **轻量级**：使用 4B 参数模型，无需 65GB+ FAISS 索引
- **多算法支持**：GRPO、PPO、DAPO、ReMax 等
- **MCP 工具协议**：灵活接入外部工具（搜索、代码解释器等）
- **多轮工具调用**：支持同步 rollout 的多轮交互训练

---

# 快速开始

## 1. 环境安装

本项目推荐使用[uv](https://uv.doczh.com/getting-started/installation/)

```bash
git clone https://github.com/Tensionteng/shallow-search.git && cd shallow-search

uv sync

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

uv add flash_attn --no-build-isolation
uv add flashinfer-python --no-build-isolation
```

如果未安装 uv：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 如果因网络原因无法安装，推荐下面的安装方式
pip install pipx
pipx install uv
```

### 安装 Milvus（向量数据库）

使用 Docker 快速启动：
```bash
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest \
  milvus run standalone
```

创建 `.env` 文件配置 Milvus 连接：
```bash
echo "MILVUS_ENDPOINT=http://localhost:19530" > .env
echo "MILVUS_TOKEN=" >> .env
```

---

## 2. 模型下载

本项目使用 Qwen 系列模型：

```bash
mkdir -p models

# 主模型 (4B 参数，用于训练)
modelscope download --model Qwen/Qwen3-4B --local_dir ./models/Qwen3-4B

# Embedding 模型 (用于 RAG 检索)
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./models/Qwen3-Embedding-0.6B

# Reranker 模型 (用于结果重排序)
modelscope download --model dengcao/Qwen3-Reranker-0.6B-seq-cls --local_dir ./models/Qwen3-Reranker-0.6B-seq-cls
```

可选：下载 Judge Model 用于模型评判奖励
```bash
modelscope download --model Qwen/QwQ-32B --local_dir ./models/QwQ-32B
```

---

## 3. 数据准备

### 准备训练/测试数据

运行数据集处理 notebook：

datasets/download_datasets.ipynb

该 notebook 会：
- 从 HuggingFace 下载 Musique 数据集
- 处理成训练所需的 parquet 格式
- 输出到 `datasets/Musique_RL/train.parquet` 和 `test.parquet`

### 构建 RAG 语料库

运行语料库构建 notebook：
运行`rag_server/build_datasets.ipynb`

该 notebook 会：
- 使用 Qwen3-Embedding-0.6B 对语料进行编码
- 创建 Milvus Collection 并插入向量数据
- 构建用于检索的向量数据库（约需 30-60 分钟）

---

## 4. 启动 RAG Server

RAG Server 为训练提供搜索能力，是训练的必要依赖。

### 修改工具配置

编辑 `envs/tools/search.py`，修改 RAG 服务地址（如果是本机运行可保持默认）：
```python
RAG_ADRESS = "localhost:8000"
```

### 启动服务

```bash
# 方法1: 直接启动
python rag_server/rag_server.py

# 方法2: 后台启动
bash rag_server/launch.sh

# 方法3: 自行用tmux启动
```

测试 RAG 服务是否正常：
```python
import requests
response = requests.post("http://localhost:8000/retrieve", 
                        json={"query": "What is the capital of France?", "top_k": 5})
print(response.json())
```

---

## 5. 开始训练

### 训练前检查清单

- [ ] RAG Server 已启动 (`curl http://localhost:8000/` 返回状态)
- [ ] 模型已下载到 `./models/` 目录
- [ ] 训练数据已准备到 `datasets/Musique_RL/`
- [ ] Milvus 数据库已启动且数据已导入

### GRPO 训练（推荐，显存占用低）

```bash
# 修改脚本中的路径配置
export MODEL_PATH=./models/Qwen3-4B
export RESULT_DIR=./results/grpo_search

# 运行训练
bash main_grpo.sh
```

### DAPO 训练

```bash
export MODEL_PATH=./models/Qwen3-4B
export RESULT_DIR=./results/dapo_search

bash main_dapo.sh
```

### PPO 训练（需要更多显存）

```bash
export MODEL_PATH=./models/Qwen3-4B
export REWARD_MODEL_PATH=./models/QwQ-32B  # 可选
export TRAIN_DATA=./datasets/Musique_RL/train.parquet
export TEST_DATA=./datasets/Musique_RL/test.parquet

bash main_ppo.sh
```

### 关键训练参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `data.train_batch_size` | 训练批次大小 | 4-128（根据显存调整） |
| `data.max_prompt_length` | 最大输入长度 | 4096 |
| `data.max_response_length` | 最大输出长度 | 512 |
| `actor_rollout_ref.rollout.n` | 采样数量 | 4 |
| `actor_rollout_ref.rollout.max_turns` | 最大工具调用轮数 | 2-4 |
| `trainer.n_gpus_per_node` | 使用 GPU 数量 | 根据硬件配置 |

---

## 6. 模型评估

```bash
export MODEL_PATH=./results/grpo_search/checkpoint-100  # 你的检查点路径
export TEST_DATA=./datasets/Musique_RL/test.parquet

bash main_eval.sh
```

### 查看训练日志

```bash
# TensorBoard
tensorboard --logdir=./results

# 查看日志文件
tail -f grpo.log
```

---

# 详细文档

- [最小实现教程](docs/rl_factory/zh/main_tutorial.md) - 以复现 Search-R1 为例的完整教程
- [工具定义指南](docs/rl_factory/zh/tools.md) - MCP 工具、多轮调用详解
- [奖赏计算指南](docs/rl_factory/zh/rewards.md) - 规则/Judge/工具验证奖励计算
- [Android 环境搭建](environments/README_android.md) - AndroidWorld 移动 Agent 训练环境配置

---

# 常见问题

**Q: 显存不足怎么办？**  
A: 尝试：1) 减小 `data.train_batch_size`；2) 使用 GRPO 代替 PPO；3) 启用 `optimizer_offload`

**Q: RAG Server 连接失败？**  
A: 检查：1) Milvus 是否启动；2) `.env` 配置是否正确；3) `envs/tools/search.py` 中的地址配置

**Q: 训练时提示找不到数据？**  
A: 确认 `datasets/Musique_RL/*.parquet` 文件是否存在，参考"数据准备"章节

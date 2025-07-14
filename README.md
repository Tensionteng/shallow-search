# 从头实现deep-search
本项目灵感来源于[Jan Nano 4B](https://menloresearch.github.io/deep-research/)，由于其技术报告的细节比较少，我花了很多时间和功夫寻找相关教程，[RL-Factory](https://github.com/Simple-Efficient/RL-Factory)挺不错的，但是search-r1还是太耗费资源了，对于我这样的学生党来说实在是力不从心（光是faiss-gpu索引就高达65GB），所以我打算自己训练一个不那么好但是效果还说得过去的模型，同时提升自己的理解。

# 快速开始
本项目推荐使用[uv](https://uv.doczh.com/getting-started/installation/)
```
git clone https://github.com/Tensionteng/shallow-search.git && cd shallow-search

uv sync
```
如果未安装uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
# 如果因网络原因无法安装，推荐下面的安装方式
pip install pipx
pipx install uv
```

# 数据集
还没整理好，整理好之后发modelscope和huggingface

# 模型

本项目使用Qwen系列（RL-Factory目前也只支持Qwen）
- Qwen3-4B
- Qwen3-Embedding-0.6B
- Qwen3-Reranker-0.6B-seq-cls（这是sentence-transform的特供版，使用起来比较方便）

# 保姆级教程
还没开始写，写完发知乎

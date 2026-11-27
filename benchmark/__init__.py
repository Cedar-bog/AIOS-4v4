"""AIOS 调度实验包。

在导入 aios 之前设置环境变量，避免 litellm 启动时联网拉取价格表，
否则每次运行会多出十几秒等待。
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "True")

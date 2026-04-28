"""Notebook-only PPG heart-rate protocol package.

这个包是从原始 ``python`` 子项目中剥离出的最小 Notebook 运行闭环。
这里只保留批量协议需要的参数、预处理工具、少量核心函数和 experimental
实验流程，避免 CLI、GUI 与旧版求解器逻辑在 Notebook 环境中交叉导入。
"""

from __future__ import annotations

__version__ = "0.3.1-notebook-base"

__all__ = ["__version__"]

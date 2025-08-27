#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频对齐工具 - 主入口文件
可以直接运行此文件启动程序
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 导入并运行主程序
from integrated_app.gui.app import main

if __name__ == '__main__':
    main()

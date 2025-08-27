#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频对齐检测器 - 应用程序入口
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir

from audio_alignment.gui.main_window import MainWindow
from audio_alignment import __version__


def main():
    """应用程序入口函数"""
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("音频对齐检测器")
    app.setApplicationVersion(__version__)
    
    # 设置工作目录为应用程序目录
    application_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(application_path)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 进入事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
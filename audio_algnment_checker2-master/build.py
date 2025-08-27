#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频对齐检测器 - 打包脚本
使用 PyInstaller 将应用程序打包成单个 exe 文件
"""

import os
import shutil
import subprocess
import sys
import PyInstaller.__main__

# 打包配置
APP_NAME = "音频对齐检测器"
MAIN_FILE = "main.py"
ICON_FILE = None  # 如果有图标文件，请在这里指定路径

# 添加需要包含的数据文件
datas = [
    ("config.json", "."),
    ("audio_alignment", "audio_alignment"),
    ("ml_data", "ml_data"),
]

# 添加额外的隐藏导入（PyInstaller可能无法自动检测的依赖）
hidden_imports = [
    "scipy.signal",
    "scipy.sparse.csgraph",
    "pandas",
    "librosa",
    "matplotlib",
    "sklearn.neighbors._partition_nodes",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree._utils",
    "joblib.externals.cloudpickle.cloudpickle",
    "openpyxl",
    "PyQt6.sip"
]

# 清理之前的构建
def clean_build():
    print("清理之前的构建...")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    # 删除.spec文件
    spec_file = f"{APP_NAME}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)

# 使用PyInstaller打包应用
def build_app():
    print(f"开始打包 {APP_NAME}...")
    
    # 构建PyInstaller命令行参数
    args = [
        MAIN_FILE,
        "--name", APP_NAME,
        "--onefile",  # 生成单个可执行文件
        "--windowed",  # 不显示控制台窗口
        "--clean",  # 清理PyInstaller缓存
        "--noconfirm",  # 不询问确认
    ]
    
    # 添加图标（如果有）
    if ICON_FILE and os.path.exists(ICON_FILE):
        args.extend(["--icon", ICON_FILE])
    
    # 添加数据文件
    for src, dst in datas:
        if os.path.exists(src):
            args.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # 添加隐藏导入
    for imp in hidden_imports:
        args.extend(["--hidden-import", imp])
    
    # 执行打包命令
    PyInstaller.__main__.run(args)
    
    print(f"打包完成！可执行文件位于 dist/{APP_NAME}.exe")

if __name__ == "__main__":
    clean_build()
    build_app() 
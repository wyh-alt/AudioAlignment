#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频对齐工具 - 打包脚本
使用 PyInstaller 将程序打包为独立的 exe 文件
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def build_exe():
    """打包为 exe 文件"""
    print("开始打包音频对齐工具...")
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    app_dir = project_root / 'integrated_app'
    
    # 确保在正确的目录
    os.chdir(project_root)
    
    # PyInstaller 命令
    cmd = [
        'pyinstaller',
        '--name=音频对齐工具',
        '--windowed',  # 无控制台窗口
        '--onefile',   # 打包为单个文件
        '--icon=integrated_app/assets/icon.svg',
        '--add-data=integrated_app/assets;integrated_app/assets',
        '--add-data=audio_algnment_checker2-master;audio_algnment_checker2-master',
        '--add-data=原唱伴奏对齐;原唱伴奏对齐',
        '--hidden-import=librosa',
        '--hidden-import=librosa.core',
        '--hidden-import=librosa.feature',
        '--hidden-import=librosa.effects',
        '--hidden-import=librosa.util',
        '--hidden-import=soundfile',
        '--hidden-import=scipy.signal',
        '--hidden-import=scipy.sparse',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=openpyxl',
        '--hidden-import=joblib',
        '--hidden-import=sklearn.ensemble',
        '--hidden-import=sklearn.model_selection',
        '--hidden-import=sklearn.metrics',
        '--hidden-import=matplotlib',
        '--hidden-import=PyQt6',
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.sip',
        '--collect-all=librosa',
        '--collect-all=scipy',
        '--collect-all=sklearn',
        '--collect-all=matplotlib',
        '--collect-all=PyQt6',
        'main.py'
    ]
    
    try:
        # 执行打包命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("打包成功！")
        print(f"输出文件: {project_root}/dist/音频对齐工具.exe")
        
        # 创建启动器批处理文件
        create_launcher(project_root)
        
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 pyinstaller，请先安装: pip install pyinstaller")
        return False
    
    return True

def create_launcher(project_root: Path):
    """创建启动器批处理文件"""
    launcher_path = project_root / "启动音频对齐工具.bat"
    exe_path = project_root / "dist" / "音频对齐工具.exe"
    
    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write('@echo off\n')
        f.write('echo 正在启动音频对齐工具...\n')
        f.write(f'start "" "{exe_path}"\n')
    
    print(f"创建启动器: {launcher_path}")

def clean_build():
    """清理构建文件"""
    project_root = Path(__file__).resolve().parent.parent
    
    # 删除构建目录
    build_dirs = ['build', 'dist', '__pycache__']
    for dir_name in build_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"已删除: {dir_path}")
    
    # 删除 .spec 文件
    spec_file = project_root / "音频对齐工具.spec"
    if spec_file.exists():
        spec_file.unlink()
        print(f"已删除: {spec_file}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'clean':
            clean_build()
            return
    
    # 检查依赖
    try:
        import PyInstaller
    except ImportError:
        print("请先安装 PyInstaller:")
        print("pip install pyinstaller")
        return
    
    # 执行打包
    if build_exe():
        print("\n打包完成！")
        print("使用方法:")
        print("1. 直接运行: dist/音频对齐工具.exe")
        print("2. 使用启动器: 启动音频对齐工具.bat")
    else:
        print("打包失败，请检查错误信息")

if __name__ == "__main__":
    main()

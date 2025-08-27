"""
文件工具模块 - 提供文件操作相关的实用函数
"""

import os
from typing import List, Tuple


def get_supported_audio_formats() -> List[str]:
    """
    获取支持的音频格式列表
    
    Returns:
        List[str]: 支持的音频格式扩展名列表
    """
    return ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']


def is_audio_file(file_path: str) -> bool:
    """
    检查文件是否为音频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为音频文件
    """
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in audio_extensions


def get_file_name(file_path: str) -> str:
    """
    获取文件名（带扩展名）
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件名
    """
    return os.path.basename(file_path)


def get_file_basename(file_path: str) -> str:
    """
    获取文件基本名（不带扩展名）
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件基本名
    """
    try:
        # 确保返回的是不带扩展名的文件名
        basename = os.path.basename(file_path)
        return os.path.splitext(basename)[0]
    except Exception as e:
        print(f"获取文件名出错: {str(e)}")
        return os.path.basename(file_path)


def get_filename_without_extension(file_path: str) -> str:
    """
    获取不带扩展名的文件名
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 不带扩展名的文件名
    """
    basename = os.path.basename(file_path)
    return os.path.splitext(basename)[0]


def create_output_filename(original_path: str, suffix: str = "_aligned", output_dir: str = None) -> str:
    """
    创建输出文件名
    
    Args:
        original_path: 原始文件路径
        suffix: 添加到文件名的后缀
        output_dir: 输出目录，如果为None则使用原始文件的目录
        
    Returns:
        str: 新的输出文件路径
    """
    dirname = output_dir if output_dir else os.path.dirname(original_path)
    filename = get_filename_without_extension(original_path)
    ext = os.path.splitext(original_path)[1]
    
    new_filename = f"{filename}{suffix}{ext}"
    return os.path.join(dirname, new_filename)


def ensure_dir_exists(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory) 
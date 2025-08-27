"""
配置管理模块 - 处理应用程序配置
"""

import os
import json
from typing import Dict, Any


class ConfigManager:
    """配置管理类，负责保存和加载应用程序配置"""
    
    def __init__(self, config_file="config.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """
        加载配置
        
        Returns:
            dict: 配置数据
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _get_default_config(self):
        """
        获取默认配置
        
        Returns:
            dict: 默认配置数据
        """
        return {
            'alignment_threshold': 0.02,  # 对齐阈值（秒）
            'id_pattern': r'(\d+)',  # ID匹配模式，只提取数字ID
            'recent_files': [],  # 最近文件列表
            'window_size': [1200, 800],  # 窗口大小
            'window_pos': [100, 100],  # 窗口位置
            'enable_vocal_exclusion': True,  # 启用人声排除功能
            'vocal_energy_ratio_threshold': 8.0,  # 人声检测能量比阈值
            'vocal_frame_duration': 0.2,  # 人声检测分析帧长度(秒)
            'min_non_vocal_segment_length': 1.0  # 最小非人声段落长度(秒)
        }
    
    def get(self, key, default=None):
        """
        获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        设置配置值
        
        Args:
            key: 配置键名
            value: 配置值
        """
        self.config[key] = value
    
    def save(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置失败: {str(e)}")
    
    def _ensure_config_dir(self) -> None:
        """确保配置目录存在"""
        if not os.path.exists(os.path.dirname(self.config_file)):
            os.makedirs(os.path.dirname(self.config_file))
    
    def add_recent_file(self, file_path: str) -> None:
        """
        添加最近使用的文件
        
        Args:
            file_path: 文件路径
        """
        recent_files = self.get("recent_files", [])
        
        # 如果文件已经在列表中，则移除
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # 将文件添加到列表开头
        recent_files.insert(0, file_path)
        
        # 限制列表长度
        max_files = self.get("max_recent_files", 10)
        if len(recent_files) > max_files:
            recent_files = recent_files[:max_files]
        
        self.set("recent_files", recent_files)
    
    def clear_recent_files(self) -> None:
        """清空最近使用的文件列表"""
        self.set("recent_files", []) 
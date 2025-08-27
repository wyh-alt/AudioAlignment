"""
音频处理模块 - 负责音频文件的读取、处理和波形数据提取
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List


class AudioProcessor:
    """音频处理类，提供音频文件的加载、处理和分析功能"""
    
    def __init__(self):
        """初始化音频处理器"""
        self.sample_rate = None
        self.audio_data = None
        self.file_path = None
        self.duration = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Tuple[np.ndarray, int]: 音频数据和采样率
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            self.audio_data = audio_data
            self.sample_rate = sample_rate
            self.file_path = file_path
            self.duration = len(audio_data) / sample_rate
            return audio_data, sample_rate
        except Exception as e:
            raise Exception(f"加载音频文件失败: {str(e)}")
    
    def load_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件（与load_audio方法相同，提供兼容性）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Tuple[np.ndarray, int]: 音频数据和采样率
        """
        return self.load_audio(file_path)
    
    def get_waveform_data(self) -> Dict[str, np.ndarray]:
        """
        获取波形数据用于可视化
        
        Returns:
            Dict[str, np.ndarray]: 包含时间和振幅数据的字典
        """
        if self.audio_data is None or self.sample_rate is None:
            raise Exception("没有加载音频数据")
            
        # 创建时间轴
        time = np.linspace(0, self.duration, len(self.audio_data))
        
        return {
            "time": time,
            "amplitude": self.audio_data
        }
    
    def detect_beats(self, threshold: float = 0.5) -> np.ndarray:
        """
        检测音频中的节奏点
        
        Args:
            threshold: 节奏检测的阈值
            
        Returns:
            np.ndarray: 节奏点的时间位置
        """
        if self.audio_data is None or self.sample_rate is None:
            raise Exception("没有加载音频数据")
            
        # 使用librosa的节奏检测
        tempo, beat_frames = librosa.beat.beat_track(y=self.audio_data, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        
        return beat_times
    
    def normalize_audio(self) -> np.ndarray:
        """
        归一化音频数据
        
        Returns:
            np.ndarray: 归一化后的音频数据
        """
        if self.audio_data is None:
            raise Exception("没有加载音频数据")
            
        # 对音频数据进行归一化处理
        normalized_data = self.audio_data / np.max(np.abs(self.audio_data))
        self.audio_data = normalized_data
        
        return normalized_data
    
    @staticmethod
    def trim_silence(audio_data: np.ndarray, sample_rate: int, threshold: float = 0.02) -> np.ndarray:
        """
        裁剪音频中的静音部分
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            threshold: 静音检测阈值
            
        Returns:
            np.ndarray: 裁剪后的音频数据
        """
        # 使用librosa的静音裁剪功能
        trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=20)
        return trimmed_audio
    
    def get_data(self) -> Dict:
        """
        获取音频数据用于分析
        
        Returns:
            Dict: 包含音频数据的字典
        """
        if self.audio_data is None or self.sample_rate is None:
            raise Exception("没有加载音频数据")
            
        return {
            'data': self.audio_data,
            'sample_rate': self.sample_rate,
            'duration': self.duration
        }
    
    def get_audio_data(self) -> Dict:
        """
        获取音频数据用于对齐分析（与get_data保持兼容性）
        
        Returns:
            Dict: 包含音频数据的字典
        """
        return self.get_data()
    
    def is_loaded(self) -> bool:
        """
        检查是否已加载音频数据
        
        Returns:
            bool: 如果已加载返回True，否则返回False
        """
        return self.audio_data is not None and self.sample_rate is not None 
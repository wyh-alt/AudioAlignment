"""
音频播放器组件 - 用于播放和控制音频
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QSlider, QStyle, QSizePolicy)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QTimer, QTime
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QIcon


class AudioPlayer(QWidget):
    """音频播放器组件，提供音频播放和控制功能"""
    
    # 自定义信号
    positionChanged = pyqtSignal(float)
    durationChanged = pyqtSignal(int)
    
    def __init__(self, parent=None, show_controls=True):
        super().__init__(parent)
        
        # 控制是否显示播放控制按钮
        self.show_controls = show_controls
        
        # 初始化播放器
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # 当前加载的文件路径
        self.current_file = None
        
        # 初始化UI
        self.setup_ui()
        self.setup_connections()
        
        # 初始化更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # 100毫秒更新一次
        self.update_timer.timeout.connect(self.update_position)
        
        # 初始化状态
        self.duration = 0
        self.is_playing = False
        
    def setup_ui(self):
        """设置UI界面"""
        # 垂直布局
        layout = QVBoxLayout(self)
        
        # 文件名标签
        self.file_label = QLabel("未加载文件")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_label)
        
        # 播放控制布局
        control_layout = QHBoxLayout()
        
        # 根据show_controls参数决定是否显示播放控制按钮
        if self.show_controls:
            # 播放/暂停按钮
            self.play_button = QPushButton()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_button.setEnabled(False)
            control_layout.addWidget(self.play_button)
            
            # 停止按钮
            self.stop_button = QPushButton()
            self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.stop_button.setEnabled(False)
            control_layout.addWidget(self.stop_button)
        else:
            # 创建隐藏的按钮，以便在其他函数中可以引用
            self.play_button = QPushButton()
            self.play_button.setVisible(False)
            self.stop_button = QPushButton()
            self.stop_button.setVisible(False)
        
        # 当前时间标签
        self.time_label = QLabel("00:00 / 00:00")
        control_layout.addWidget(self.time_label)
        
        # 进度条
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        control_layout.addWidget(self.position_slider)
        
        # 音量按钮和滑块根据show_controls决定是否显示
        if self.show_controls:
            # 音量按钮
            self.volume_button = QPushButton()
            self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
            control_layout.addWidget(self.volume_button)
            
            # 音量滑块
            self.volume_slider = QSlider(Qt.Orientation.Horizontal)
            self.volume_slider.setRange(0, 100)
            self.volume_slider.setValue(70)  # 默认音量70%
            self.volume_slider.setMaximumWidth(100)
            control_layout.addWidget(self.volume_slider)
        else:
            # 创建隐藏的按钮和滑块
            self.volume_button = QPushButton()
            self.volume_button.setVisible(False)
            self.volume_slider = QSlider(Qt.Orientation.Horizontal)
            self.volume_slider.setVisible(False)
        
        # 添加控制布局到主布局
        layout.addLayout(control_layout)
        
        # 设置布局
        self.setLayout(layout)
    
    def setup_connections(self):
        """设置信号连接"""
        # 播放器信号
        self.player.playbackStateChanged.connect(self.update_play_state)
        self.player.durationChanged.connect(self.update_duration)
        
        # 只有在显示控制按钮时才连接按钮信号
        if self.show_controls:
            # 按钮信号
            self.play_button.clicked.connect(self.toggle_play)
            self.stop_button.clicked.connect(self.stop)
            self.volume_button.clicked.connect(self.toggle_mute)
            
            # 滑块信号
            self.volume_slider.valueChanged.connect(self.set_volume)
        
        # 始终连接位置滑块，允许用户拖动调整播放位置
        self.position_slider.sliderMoved.connect(self.set_position)
    
    def load_file(self, file_path):
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
        """
        if file_path and os.path.exists(file_path):
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.current_file = file_path
            
            # 设置文件名标签，显示完整文件名
            self.file_label.setText(os.path.basename(file_path))
            # 添加文件路径提示
            self.file_label.setToolTip(file_path)
            
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.update_timer.start()
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    
    def stop(self):
        """停止播放"""
        self.player.stop()
    
    def set_position(self, position):
        """
        设置播放位置
        
        Args:
            position: 播放位置（毫秒）
        """
        self.player.setPosition(position)
    
    def set_volume(self, volume):
        """
        设置音量
        
        Args:
            volume: 音量值（0-100）
        """
        volume_float = volume / 100.0
        self.audio_output.setVolume(volume_float)
        
        # 更新音量图标
        if volume == 0:
            self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolumeMuted))
        else:
            self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
            
        # 如果有显示音量滑块，同步更新滑块位置
        if self.show_controls and hasattr(self, 'volume_slider'):
            self.volume_slider.setValue(volume)
    
    def toggle_mute(self):
        """切换静音状态"""
        if self.audio_output.isMuted():
            self.audio_output.setMuted(False)
            self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
        else:
            self.audio_output.setMuted(True)
            self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolumeMuted))
    
    @pyqtSlot()
    def update_position(self):
        """更新当前播放位置"""
        position = self.player.position()
        self.position_slider.setValue(position)
        self.positionChanged.emit(position / 1000.0)
        
        # 更新时间标签
        duration = self.player.duration()
        if duration > 0:
            self.time_label.setText(f"{QTime(0, 0).addMSecs(position).toString('mm:ss')} / "
                                  f"{QTime(0, 0).addMSecs(duration).toString('mm:ss')}")
    
    def update_duration(self, duration):
        """
        更新音频总时长
        
        Args:
            duration: 音频总时长（毫秒）
        """
        self.position_slider.setRange(0, duration)
        self.durationChanged.emit(duration)
    
    @pyqtSlot()
    def update_play_state(self):
        """更新播放状态按钮图标"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    
    def _format_time(self, milliseconds):
        """
        格式化时间为 MM:SS 格式
        
        Args:
            milliseconds: 毫秒数
            
        Returns:
            str: 格式化后的时间字符串
        """
        seconds = int(milliseconds / 1000)
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def seek_to_time(self, seconds):
        """
        跳转到指定时间点
        
        Args:
            seconds: 时间点（秒）
        """
        milliseconds = int(seconds * 1000)
        self.player.setPosition(milliseconds)
    
    def get_current_position_seconds(self):
        """
        获取当前播放位置（秒）
        
        Returns:
            float: 当前播放位置（秒）
        """
        return self.player.position() / 1000.0
    
    def play(self):
        """直接播放音频"""
        if self.current_file:
            self.player.play()
    
    def pause(self):
        """暂停音频播放"""
        self.player.pause()
    
    def is_loaded(self):
        """
        检查是否已加载音频文件
        
        Returns:
            bool: 是否已加载音频文件
        """
        return self.current_file is not None and os.path.exists(self.current_file) 
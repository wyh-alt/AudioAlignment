"""
主窗口模块 - 应用程序的主界面
"""

import os
import csv
import joblib
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QMessageBox,
                           QSplitter, QStatusBar, QToolBar, QMenuBar, QMenu,
                           QDialog, QDialogButtonBox, QSpinBox, 
                           QGroupBox, QFormLayout, QDoubleSpinBox, QTabWidget,
                           QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
                           QProgressBar, QTextEdit, QScrollArea, QSlider, QCheckBox,
                           QComboBox, QRadioButton, QButtonGroup, QInputDialog)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSettings, QSize
from PyQt6.QtGui import QIcon, QAction, QColor
from PyQt6.QtWidgets import QApplication

from ..core.audio_processor import AudioProcessor
from ..core.alignment_detector import AlignmentDetector
from ..utils.file_utils import is_audio_file, get_file_name, get_file_basename
from ..utils.config import ConfigManager
from .widgets.waveform_view import WaveformView
from .widgets.audio_player import AudioPlayer

import re
import traceback
import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("设置")
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 阈值设置组
        threshold_group = QGroupBox("对齐检测设置")
        threshold_layout = QFormLayout()
        
        # 对齐阈值
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.001, 1.0)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(self.config.get('alignment_threshold', 0.02))
        threshold_layout.addRow("对齐阈值 (秒):", self.threshold_spin)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # 人声排除设置组
        vocal_group = QGroupBox("人声排除设置")
        vocal_layout = QFormLayout()
        
        # 启用人声排除
        self.enable_vocal_exclusion = QCheckBox()
        self.enable_vocal_exclusion.setChecked(self.config.get('enable_vocal_exclusion', True))
        self.enable_vocal_exclusion.stateChanged.connect(self.on_vocal_exclusion_changed)
        vocal_layout.addRow("启用人声排除:", self.enable_vocal_exclusion)
        
        # 人声能量比阈值
        self.vocal_ratio_spin = QDoubleSpinBox()
        self.vocal_ratio_spin.setRange(2.0, 20.0)
        self.vocal_ratio_spin.setSingleStep(0.5)
        self.vocal_ratio_spin.setDecimals(1)
        self.vocal_ratio_spin.setValue(self.config.get('vocal_energy_ratio_threshold', 8.0))
        self.vocal_ratio_spin.setToolTip("能量比阈值，值越大要求两个音频能量差异越明显才判定为人声段落")
        vocal_layout.addRow("人声能量比阈值:", self.vocal_ratio_spin)
        
        # 人声检测帧长度
        self.vocal_frame_spin = QDoubleSpinBox()
        self.vocal_frame_spin.setRange(0.05, 1.0)
        self.vocal_frame_spin.setSingleStep(0.05)
        self.vocal_frame_spin.setDecimals(2)
        self.vocal_frame_spin.setValue(self.config.get('vocal_frame_duration', 0.2))
        self.vocal_frame_spin.setToolTip("分析帧长度(秒)，较短的帧可以更精确地定位人声段落")
        vocal_layout.addRow("分析帧长度 (秒):", self.vocal_frame_spin)
        
        # 最小非人声段落长度
        self.min_segment_spin = QDoubleSpinBox()
        self.min_segment_spin.setRange(0.5, 5.0)
        self.min_segment_spin.setSingleStep(0.1)
        self.min_segment_spin.setDecimals(1)
        self.min_segment_spin.setValue(self.config.get('min_non_vocal_segment_length', 1.0))
        self.min_segment_spin.setToolTip("最小有效的非人声段落长度(秒)，较长的段落分析更稳定")
        vocal_layout.addRow("最小非人声段落长度 (秒):", self.min_segment_spin)
        
        vocal_group.setLayout(vocal_layout)
        layout.addWidget(vocal_group)
        
        # 批量处理设置组
        batch_group = QGroupBox("批量处理设置")
        batch_layout = QFormLayout()
        
        # ID匹配正则表达式
        self.id_pattern_edit = QLineEdit()
        self.id_pattern_edit.setText(self.config.get('id_pattern', r'(\d+)'))
        self.id_pattern_edit.setToolTip("用于从文件名提取数字ID的正则表达式，括号中的分组将作为匹配ID")
        batch_layout.addRow("ID匹配正则表达式:", self.id_pattern_edit)
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # 初始化界面状态
        self.on_vocal_exclusion_changed()
    
    def on_vocal_exclusion_changed(self):
        """人声排除选项变更处理"""
        enabled = self.enable_vocal_exclusion.isChecked()
        self.vocal_ratio_spin.setEnabled(enabled)
        self.vocal_frame_spin.setEnabled(enabled)
        self.min_segment_spin.setEnabled(enabled)
        
    def get_settings(self):
        """获取设置值"""
        return {
            'alignment_threshold': self.threshold_spin.value(),
            'id_pattern': self.id_pattern_edit.text(),
            'enable_vocal_exclusion': self.enable_vocal_exclusion.isChecked(),
            'vocal_energy_ratio_threshold': self.vocal_ratio_spin.value(),
            'vocal_frame_duration': self.vocal_frame_spin.value(),
            'min_non_vocal_segment_length': self.min_segment_spin.value()
        }


class NumericTableWidgetItem(QTableWidgetItem):
    """可按数字排序的表格项"""
    def __init__(self, text, value):
        super().__init__(text)
        self.value = value
        
    def __lt__(self, other):
        """重载小于运算符以支持数字排序"""
        try:
            return float(self.value) < float(other.value)
        except (ValueError, TypeError):
            return super().__lt__(other)


class DetailAnalysisDialog(QDialog):
    """详细分析对话框，显示完整的分析数据"""
    
    def __init__(self, parent=None, analysis_data=None):
        super().__init__(parent)
        self.analysis_data = analysis_data
        self.setWindowTitle("详细分析结果")
        self.resize(800, 600)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 创建内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 文件信息
        if self.analysis_data:
            # 创建详细信息文本区域
            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            
            # 格式化分析数据为文本
            analysis_text = self._format_analysis_data()
            self.text_edit.setPlainText(analysis_text)
            
            content_layout.addWidget(self.text_edit)
        else:
            # 如果没有分析数据，显示提示信息
            no_data_label = QLabel("没有可用的分析数据")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_layout.addWidget(no_data_label)
        
        # 设置滚动区域内容
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # 添加关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _format_analysis_data(self):
        """将分析数据格式化为可读文本"""
        if not self.analysis_data:
            return "无分析数据"
            
        lines = ["=== 详细分析结果 ===\n"]
        
        # 添加基本信息
        lines.append(f"对齐状态: {'对齐' if self.analysis_data.get('is_aligned', False) else '不对齐'}")
        
        # 添加时间偏移信息
        offset_sec = self.analysis_data.get('offset', 0)
        offset_ms = self.analysis_data.get('offset_ms', 0)
        threshold_ms = self.analysis_data.get('threshold_ms', 0)
        
        lines.append(f"时间偏移: {offset_sec:.6f} 秒 ({offset_ms:.2f} 毫秒)")
        lines.append(f"对齐阈值: {threshold_ms:.2f} 毫秒")
        
        # 添加音频时长信息
        has_duration_mismatch = self.analysis_data.get('has_duration_mismatch', False)
        if has_duration_mismatch:
            duration1 = self.analysis_data.get('duration1', 0)
            duration2 = self.analysis_data.get('duration2', 0)
            duration_diff = self.analysis_data.get('duration_diff', 0)
            
            lines.append(f"\n=== 音频时长信息 ===")
            lines.append(f"音频时长是否一致: 否")
            lines.append(f"时长差异: {duration_diff:.2f} 秒")
            lines.append(f"音频1时长: {duration1:.2f} 秒")
            lines.append(f"音频2时长: {duration2:.2f} 秒")
            if duration1 > duration2:
                lines.append(f"说明: 音频1比音频2长 {duration_diff:.2f} 秒")
            else:
                lines.append(f"说明: 音频2比音频1长 {duration_diff:.2f} 秒")
                
            # 添加警告信息
            if duration_diff > 1.0:
                lines.append(f"\n⚠️ 警告: 音频时长差异较大，可能存在音频截断问题！")
                if self.analysis_data.get('is_aligned', False):
                    lines.append(f"虽然音频前段对齐，但由于时长差异较大，建议检查音频是否存在截断问题。")
        
        # 添加人声排除信息
        vocal_excluded = False
        if 'segment_info' in self.analysis_data:
            segment_info = self.analysis_data.get('segment_info', {})
            if 'vocal_excluded' in segment_info:
                vocal_excluded = segment_info.get('vocal_excluded', False)
                lines.append(f"\n=== 人声排除信息 ===")
                lines.append(f"启用人声排除: {'是' if vocal_excluded else '否'}")
                if vocal_excluded:
                    # 如果有分段信息，显示排除人声后使用的段落数量
                    if 'segment_offsets' in segment_info:
                        segment_offsets = segment_info.get('segment_offsets', [])
                        lines.append(f"使用的非人声段落数量: {len(segment_offsets)}")
        
        # 添加一致性判断信息
        if 'is_highly_consistent' in self.analysis_data:
            is_highly_consistent = self.analysis_data.get('is_highly_consistent', False)
            if is_highly_consistent:
                lines.append(f"\n=== 一致性判断 ===")
                lines.append(f"高度一致性检测: 各分段偏移均接近0且变化极小")
                lines.append(f"判断结果: 因分段分析显示高度一致，判定为对齐")
        
        # 添加详细判断过程
        if 'debug_info' in self.analysis_data:
            lines.append(f"\n=== 判断过程 ===")
            lines.append(self.analysis_data.get('debug_info', ''))
        
        # 添加渐进性失调信息
        if 'progressive_misalignment' in self.analysis_data:
            lines.append(f"\n=== 渐进性失调分析 ===")
            progressive = self.analysis_data.get('progressive_misalignment', False)
            lines.append(f"存在渐进性失调: {'是' if progressive else '否'}")
            
            # 添加分段分析结果
            if 'segment_info' in self.analysis_data:
                segment_info = self.analysis_data.get('segment_info', {})
                
                if 'segment_offsets' in segment_info:
                    segment_offsets = segment_info.get('segment_offsets', [])
                    segment_strengths = segment_info.get('segment_strengths', [])
                    
                    lines.append(f"\n分段偏移量:")
                    for i, (offset, strength) in enumerate(zip(segment_offsets, segment_strengths)):
                        lines.append(f"  段落 {i+1}: {offset:.6f}秒 (强度: {strength:.4f})")
                
                # 添加统计信息
                lines.append(f"\n统计信息:")
                lines.append(f"  偏移均值: {segment_info.get('mean_offset', 0):.6f}秒")
                lines.append(f"  偏移标准差: {segment_info.get('offset_std_dev', 0):.6f}秒")
                lines.append(f"  最大偏移差异: {segment_info.get('max_offset_diff', 0):.6f}秒")
                
                # 添加趋势分析
                if 'has_consistent_trend' in segment_info:
                    has_trend = segment_info.get('has_consistent_trend', False)
                    trend_consistency = segment_info.get('trend_consistency', 0)
                    lines.append(f"\n趋势分析:")
                    lines.append(f"  存在单向趋势: {'是' if has_trend else '否'}")
                    lines.append(f"  趋势一致性: {trend_consistency:.2f}")
                    
                # 添加小偏移判断
                if 'small_offset_no_misalignment' in segment_info:
                    small_offset = segment_info.get('small_offset_no_misalignment', False)
                    lines.append(f"  整体偏移较小: {'是' if small_offset else '否'}")
                
                # 添加判断条件
                lines.append(f"\n判断条件:")
                if 'std_dev_check' in segment_info:
                    std_dev_check = segment_info.get('std_dev_check', False)
                    lines.append(f"  标准差检查: {'触发' if std_dev_check else '未触发'}")
                if 'max_diff_check' in segment_info:
                    max_diff_check = segment_info.get('max_diff_check', False)
                    lines.append(f"  最大差异检查: {'触发' if max_diff_check else '未触发'}")
                if 'trend_check' in segment_info:
                    trend_check = segment_info.get('trend_check', False)
                    lines.append(f"  趋势检查: {'触发' if trend_check else '未触发'}")
        
        # 添加算法数据
        peak_method = self.analysis_data.get('peak_method', '')
        if peak_method:
            lines.append(f"\n=== 分析方法信息 ===")
            if peak_method == 'single':
                lines.append("分析方法: 单峰值检测")
            elif peak_method == 'single_fallback':
                lines.append("分析方法: 单峰值回退检测")
            elif peak_method == 'best_of_few':
                num_peaks = self.analysis_data.get('num_peaks', 0)
                lines.append(f"分析方法: 最佳峰值检测")
                lines.append(f"检测到的峰值数量: {num_peaks}")
            elif peak_method == 'multi_peak_avg':
                num_peaks = self.analysis_data.get('num_peaks', 0)
                num_peaks_used = self.analysis_data.get('num_peaks_used', 0)
                lines.append(f"分析方法: 多峰值平均检测")
                lines.append(f"检测到的峰值数量: {num_peaks}")
                lines.append(f"用于计算的峰值数量: {num_peaks_used}")
        
        # 添加可信度信息
        lines.append(f"\n=== 可信度信息 ===")
        correlation_strength = self.analysis_data.get('correlation_strength', 0)
        low_confidence = self.analysis_data.get('low_confidence', False)
        
        lines.append(f"相关性强度: {correlation_strength:.6f}")
        lines.append(f"低可信度标记: {'是' if low_confidence else '否'}")
        
        # 添加一致性信息
        if 'consistency' in self.analysis_data:
            consistency = self.analysis_data.get('consistency', 0)
            lines.append(f"峰值一致性: {consistency:.6f} 秒")
            lines.append(f"一致性说明: 该值越小表示检测到的多个峰值越一致，结果越可靠")
        
        # 添加所有其他信息
        lines.append(f"\n=== 其他分析参数 ===")
        for key, value in self.analysis_data.items():
            # 跳过已经显示过的字段
            if key in ['is_aligned', 'offset', 'offset_ms', 'threshold_ms', 
                       'peak_method', 'num_peaks', 'num_peaks_used',
                       'correlation_strength', 'low_confidence', 'consistency',
                       'id', 'ref_file', 'align_file', 'debug_info', 
                       'progressive_misalignment', 'segment_info', 'is_highly_consistent']:
                continue
                
            # 格式化不同类型的值
            if isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{key}: {'是' if value else '否'}")
            elif isinstance(value, (list, tuple)):
                lines.append(f"{key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)


class MainWindow(QMainWindow):
    """应用程序主窗口"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("音频对齐检测器")
        self.resize(1200, 800)
        
        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        # 初始化成员变量
        self.file1_path = ""
        self.file2_path = ""
        self.recent_files = []
        self.batch_files = []
        self.batch_results = []
        self.aligned_count = 0
        self.not_aligned_count = 0
        self.error_count = 0
        
        # 初始化音频对齐检测器
        self.alignment_detector = AlignmentDetector(threshold_ms=self.config_manager.get('alignment_threshold', 0.02) * 1000)
        # 设置人声排除参数
        self.alignment_detector.enable_vocal_exclusion = self.config_manager.get('enable_vocal_exclusion', True)
        self.alignment_detector.vocal_energy_ratio_threshold = self.config_manager.get('vocal_energy_ratio_threshold', 8.0)
        self.alignment_detector.vocal_frame_duration = self.config_manager.get('vocal_frame_duration', 0.2)
        self.alignment_detector.min_non_vocal_segment_length = self.config_manager.get('min_non_vocal_segment_length', 1.0)
        
        # 设置界面
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        
        # 加载最近使用的文件
        self.load_settings()
        self.update_recent_files_menu()
        
        # 反馈标记功能相关变量
        self.feedback_results = []  # 存储反馈标记数据
        self.selected_feedback_row = -1  # 当前选中的反馈行
        
        # 数据提取功能相关变量
        self.extract_file_data = None  # 存储提取的文件数据
        
        # 历史记录相关变量
        self.history_records = []  # 存储历史检测记录
        self.history_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'history')
        # 确保历史记录目录存在
        if not os.path.exists(self.history_directory):
            os.makedirs(self.history_directory)
        
        # 机器学习功能相关变量
        self.ml_model = None  # 机器学习模型
        self.model_trained = False  # 模型是否已训练
        self.model_features = []  # 模型特征列表
        self.feature_importance = {}  # 特征重要性
    
    def setup_ui(self):
        """设置UI界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 单文件对齐检测标签页
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        
        # 文件选择区域
        file_layout = QHBoxLayout()
        
        # 文件1区域
        file1_group = QGroupBox("音频文件1")
        file1_layout = QVBoxLayout(file1_group)
        self.file1_label = QLabel("未选择文件")
        self.file1_button = QPushButton("选择文件")
        self.file1_path_edit = QLineEdit()
        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.file1_button)
        file1_layout.addWidget(self.file1_path_edit)
        file_layout.addWidget(file1_group)
        
        # 文件2区域
        file2_group = QGroupBox("音频文件2")
        file2_layout = QVBoxLayout(file2_group)
        self.file2_label = QLabel("未选择文件")
        self.file2_button = QPushButton("选择文件")
        self.file2_path_edit = QLineEdit()
        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.file2_button)
        file2_layout.addWidget(self.file2_path_edit)
        file_layout.addWidget(file2_group)
        
        single_layout.addLayout(file_layout)
        
        # 分析按钮
        self.analyze_button = QPushButton("分析对齐情况")
        single_layout.addWidget(self.analyze_button)
        
        # 波形显示区域
        waveform_layout = QVBoxLayout()
        
        # 创建波形显示
        self.waveform_view = WaveformView()
        waveform_layout.addWidget(self.waveform_view)
        
        single_layout.addLayout(waveform_layout)
        
        # 音频播放控制区域
        player_control_layout = QHBoxLayout()
        
        # 统一的播放按钮
        self.play_pause_button = QPushButton("播放")
        self.play_pause_button.setFixedWidth(80)
        player_control_layout.addWidget(self.play_pause_button)
        
        # 添加音量控制滑块
        volume_layout = QHBoxLayout()
        volume_label = QLabel("音量:")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)  # 默认音量100%
        self.volume_slider.setFixedWidth(150)
        self.volume_value_label = QLabel("100%")
        self.volume_value_label.setFixedWidth(50)
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value_label)
        player_control_layout.addLayout(volume_layout)
        
        # 添加空白占位
        player_control_layout.addStretch(1)
        
        # 添加音频播放控制布局
        single_layout.addLayout(player_control_layout)
        
        # 音频播放器区域
        player_layout = QHBoxLayout()
        
        # 创建音频播放器（移除独立的播放按钮）
        player1_group = QGroupBox("音频1波形")
        player1_layout = QVBoxLayout(player1_group)
        self.audio_player1 = AudioPlayer(show_controls=False)
        player1_layout.addWidget(self.audio_player1)
        player_layout.addWidget(player1_group)
        
        player2_group = QGroupBox("音频2波形")
        player2_layout = QVBoxLayout(player2_group)
        self.audio_player2 = AudioPlayer(show_controls=False)
        player2_layout.addWidget(self.audio_player2)
        player_layout.addWidget(player2_group)
        
        single_layout.addLayout(player_layout)
        
        # 结果显示区域
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout(results_group)
        self.result_text = QLabel("请选择两个音频文件并点击分析")
        self.result_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.result_text)
        single_layout.addWidget(results_group)
        
        # 添加单文件检测标签页
        self.tab_widget.addTab(single_tab, "单文件对齐检测")
        
        # ==== 新增: 数据提取标签页 ====
        data_extract_tab = QWidget()
        data_extract_layout = QVBoxLayout(data_extract_tab)
        
        # 文件选择区域
        file_select_group = QGroupBox("CSV/Excel文件选择")
        file_select_layout = QHBoxLayout(file_select_group)
        
        self.extract_file_path = QLineEdit()
        self.extract_file_path.setPlaceholderText("请选择数据文件...")
        self.extract_file_button = QPushButton("选择文件")
        
        file_select_layout.addWidget(self.extract_file_path)
        file_select_layout.addWidget(self.extract_file_button)
        
        data_extract_layout.addWidget(file_select_group)
        
        # 列映射设置区域
        column_mapping_group = QGroupBox("列映射设置")
        column_mapping_layout = QFormLayout(column_mapping_group)
        
        self.id_column_combo = QComboBox()
        self.status_column_combo = QComboBox()
        self.load_columns_button = QPushButton("加载文件列")
        
        column_mapping_layout.addRow("ID列:", self.id_column_combo)
        column_mapping_layout.addRow("对齐状态列:", self.status_column_combo)
        column_mapping_layout.addRow("", self.load_columns_button)
        
        data_extract_layout.addWidget(column_mapping_group)
        
        # 数据转换设置区域
        convert_setting_group = QGroupBox("数据转换设置")
        convert_setting_layout = QFormLayout(convert_setting_group)
        
        self.align_text_edit = QLineEdit("无需对齐")
        self.align_text_edit.setPlaceholderText("例如：无需对齐,无变化（逗号分隔可输入多个）")
        self.align_text_edit.setToolTip("表示「对齐」的源文本，多个值用逗号分隔")
        
        self.not_align_text_edit = QLineEdit("已对齐,无法对齐")
        self.not_align_text_edit.setPlaceholderText("例如：已对齐,无法对齐（逗号分隔可输入多个）")
        self.not_align_text_edit.setToolTip("表示「不对齐」的源文本，多个值用逗号分隔")
        
        self.target_align_text_edit = QLineEdit("对齐")
        self.target_align_text_edit.setToolTip("转换后表示「对齐」的标准文本")
        
        self.target_not_align_text_edit = QLineEdit("不对齐")
        self.target_not_align_text_edit.setToolTip("转换后表示「不对齐」的标准文本")
        
        convert_setting_layout.addRow("源数据'对齐'文本:", self.align_text_edit)
        convert_setting_layout.addRow("转换为:", self.target_align_text_edit)
        convert_setting_layout.addRow("源数据'不对齐'文本:", self.not_align_text_edit)
        convert_setting_layout.addRow("转换为:", self.target_not_align_text_edit)
        
        data_extract_layout.addWidget(convert_setting_group)
        
        # 数据预览表格
        preview_group = QGroupBox("数据预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(3)  # ID, 原始状态, 转换后状态
        self.preview_table.setHorizontalHeaderLabels(["ID", "原始对齐状态", "转换后状态"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        preview_layout.addWidget(self.preview_table)
        
        data_extract_layout.addWidget(preview_group)
        
        # 操作按钮区域
        action_layout = QHBoxLayout()
        
        self.preview_data_button = QPushButton("预览数据")
        self.extract_data_button = QPushButton("提取并保存数据")
        self.extract_data_button.setEnabled(False)  # 初始禁用
        
        action_layout.addWidget(self.preview_data_button)
        action_layout.addWidget(self.extract_data_button)
        
        data_extract_layout.addLayout(action_layout)
        
        # 添加数据提取标签页
        self.tab_widget.addTab(data_extract_tab, "数据提取")
        
        # 批量文件对齐检测标签页
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        
        # 批量文件选择区域
        batch_file_layout = QHBoxLayout()
        
        # 参考文件区域（A组）
        ref_group = QGroupBox("参考文件组 (A)")
        ref_layout = QVBoxLayout(ref_group)
        self.ref_files_label = QLabel("未选择文件")
        ref_buttons_layout = QHBoxLayout()
        self.ref_files_button = QPushButton("选择文件夹(按文件ID批量匹配)")
        self.ref_files_multi_button = QPushButton("选择多个文件")
        ref_buttons_layout.addWidget(self.ref_files_button)
        ref_buttons_layout.addWidget(self.ref_files_multi_button)
        ref_layout.addWidget(self.ref_files_label)
        ref_layout.addLayout(ref_buttons_layout)
        batch_file_layout.addWidget(ref_group)
        
        # 对齐文件区域（B组）
        align_group = QGroupBox("对齐文件组 (B)")
        align_layout = QVBoxLayout(align_group)
        self.align_files_label = QLabel("未选择文件")
        align_buttons_layout = QHBoxLayout()
        self.align_files_button = QPushButton("选择文件夹(按文件ID批量匹配)")
        self.align_files_multi_button = QPushButton("选择多个文件")
        align_buttons_layout.addWidget(self.align_files_button)
        align_buttons_layout.addWidget(self.align_files_multi_button)
        align_layout.addWidget(self.align_files_label)
        align_layout.addLayout(align_buttons_layout)
        batch_file_layout.addWidget(align_group)
        
        batch_layout.addLayout(batch_file_layout)
        
        # 批量分析按钮
        batch_action_layout = QHBoxLayout()
        self.batch_analyze_button = QPushButton("批量分析对齐情况")
        self.export_excel_button = QPushButton("导出Excel表格")
        self.export_excel_button.setEnabled(False)  # 初始状态下禁用导出按钮
        
        # 添加历史记录相关按钮
        self.save_history_button = QPushButton("保存当前结果")
        self.save_history_button.setEnabled(False)  # 初始状态下禁用
        self.view_history_button = QPushButton("查看历史记录")
        
        batch_action_layout.addWidget(self.batch_analyze_button)
        batch_action_layout.addWidget(self.export_excel_button)
        batch_action_layout.addWidget(self.save_history_button)
        batch_action_layout.addWidget(self.view_history_button)
        
        batch_layout.addLayout(batch_action_layout)
        
        # 添加进度条
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setFormat("%p% - 正在处理: %v/%m")
        batch_layout.addWidget(self.batch_progress_bar)
        
        # 批量结果表格
        self.batch_results_table = QTableWidget()
        self.batch_results_table.setColumnCount(6)  # 增加到6列，包括详细按钮列
        self.batch_results_table.setHorizontalHeaderLabels(["ID", "参考文件", "对齐文件", "时间偏差(秒)", "对齐状态", "详细分析"])
        self.batch_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # 自定义设置"详细分析"列宽
        self.batch_results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        # 启用表格排序
        self.batch_results_table.setSortingEnabled(True)
        # 默认按ID列升序排序
        self.batch_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
        # 批量结果表格项鼠标悬停事件处理
        self.batch_results_table.setMouseTracking(True)
        batch_layout.addWidget(self.batch_results_table)
        
        # 批量结果统计区域
        results_stats_group = QGroupBox("分析报表")
        results_stats_layout = QHBoxLayout(results_stats_group)
        
        # 对齐文件数统计
        self.aligned_count_label = QLabel("对齐文件数: 0")
        self.aligned_count_label.setStyleSheet("color: green; font-weight: bold;")
        results_stats_layout.addWidget(self.aligned_count_label)
        
        # 不对齐文件数统计
        self.not_aligned_count_label = QLabel("不对齐文件数: 0")
        self.not_aligned_count_label.setStyleSheet("color: red; font-weight: bold;")
        results_stats_layout.addWidget(self.not_aligned_count_label)
        
        # 总文件对数统计
        self.total_count_label = QLabel("总文件对数: 0")
        results_stats_layout.addWidget(self.total_count_label)
        
        batch_layout.addWidget(results_stats_group)
        
        # 添加批量检测标签页
        self.tab_widget.addTab(batch_tab, "批量对齐检测")
        
        # ==== 新增: 反馈标记标签页 ====
        feedback_tab = QWidget()
        feedback_layout = QVBoxLayout(feedback_tab)
        
        # 添加操作按钮
        feedback_actions_layout = QHBoxLayout()
        self.import_batch_results_button = QPushButton("从批量检测导入结果")
        self.save_feedback_button = QPushButton("保存反馈结果")
        self.load_feedback_button = QPushButton("加载反馈数据")
        self.import_reference_button = QPushButton("导入参考数据并自动反馈")  # 新增按钮
        
        feedback_actions_layout.addWidget(self.import_batch_results_button)
        feedback_actions_layout.addWidget(self.save_feedback_button)
        feedback_actions_layout.addWidget(self.load_feedback_button)
        feedback_actions_layout.addWidget(self.import_reference_button)  # 添加到布局
        
        feedback_layout.addLayout(feedback_actions_layout)
        
        # 添加反馈表格
        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(8)  # ID, 参考文件, 对齐文件, 时间偏差, 系统判定, 实际情况, 反馈, 备注
        self.feedback_table.setHorizontalHeaderLabels(["ID", "参考文件", "对齐文件", "时间偏差(秒)", 
                                                       "系统判定", "实际情况", "反馈", "备注"])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        feedback_layout.addWidget(self.feedback_table)
        
        # 添加反馈操作区域
        feedback_operation_group = QGroupBox("反馈标记操作")
        feedback_operation_layout = QVBoxLayout(feedback_operation_group)
        
        # 反馈选项
        feedback_options_layout = QHBoxLayout()
        
        # 实际情况选择
        actual_status_group = QGroupBox("实际对齐情况")
        actual_status_layout = QVBoxLayout(actual_status_group)
        self.actual_status_aligned = QRadioButton("对齐")
        self.actual_status_not_aligned = QRadioButton("不对齐")
        self.actual_status_unsure = QRadioButton("不确定")
        
        self.actual_status_group = QButtonGroup()
        self.actual_status_group.addButton(self.actual_status_aligned, 1)
        self.actual_status_group.addButton(self.actual_status_not_aligned, 2)
        self.actual_status_group.addButton(self.actual_status_unsure, 3)
        
        actual_status_layout.addWidget(self.actual_status_aligned)
        actual_status_layout.addWidget(self.actual_status_not_aligned)
        actual_status_layout.addWidget(self.actual_status_unsure)
        
        # 反馈选择
        feedback_group = QGroupBox("反馈选项")
        feedback_layout_group = QVBoxLayout(feedback_group)
        self.feedback_correct = QRadioButton("系统判断正确")
        self.feedback_incorrect = QRadioButton("系统判断错误")
        self.feedback_unsure = QRadioButton("不确定")
        
        self.feedback_group = QButtonGroup()
        self.feedback_group.addButton(self.feedback_correct, 1)
        self.feedback_group.addButton(self.feedback_incorrect, 2)
        self.feedback_group.addButton(self.feedback_unsure, 3)
        
        feedback_layout_group.addWidget(self.feedback_correct)
        feedback_layout_group.addWidget(self.feedback_incorrect)
        feedback_layout_group.addWidget(self.feedback_unsure)
        
        feedback_options_layout.addWidget(actual_status_group)
        feedback_options_layout.addWidget(feedback_group)
        
        # 添加备注区域
        remark_layout = QVBoxLayout()
        remark_label = QLabel("备注:")
        self.remark_text = QTextEdit()
        self.remark_text.setMaximumHeight(100)
        remark_layout.addWidget(remark_label)
        remark_layout.addWidget(self.remark_text)
        
        # 应用反馈按钮
        self.apply_feedback_button = QPushButton("应用反馈")
        
        feedback_operation_layout.addLayout(feedback_options_layout)
        feedback_operation_layout.addLayout(remark_layout)
        feedback_operation_layout.addWidget(self.apply_feedback_button)
        
        feedback_layout.addWidget(feedback_operation_group)
        
        # 添加反馈标记标签页
        self.tab_widget.addTab(feedback_tab, "反馈标记")
        
        # ==== 新增: 机器学习标签页 ====
        ml_tab = QWidget()
        ml_layout = QVBoxLayout(ml_tab)
        
        # 添加模型训练区域
        model_training_group = QGroupBox("模型训练")
        model_training_layout = QVBoxLayout(model_training_group)
        
        # 数据来源选择
        data_source_layout = QHBoxLayout()
        data_source_label = QLabel("训练数据来源:")
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["反馈标记数据", "外部CSV文件"])
        data_source_layout.addWidget(data_source_label)
        data_source_layout.addWidget(self.data_source_combo)
        data_source_layout.addStretch(1)
        
        # 外部数据路径选择
        external_data_layout = QHBoxLayout()
        self.external_data_path = QLineEdit()
        self.external_data_path.setPlaceholderText("外部CSV文件路径...")
        self.external_data_button = QPushButton("选择文件")
        external_data_layout.addWidget(self.external_data_path)
        external_data_layout.addWidget(self.external_data_button)
        
        # 训练参数设置
        training_params_layout = QFormLayout()
        
        # 测试集比例
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setDecimals(2)
        training_params_layout.addRow("测试集比例:", self.test_size_spin)
        
        # 随机种子
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        training_params_layout.addRow("随机种子:", self.random_seed_spin)
        
        # 训练操作按钮
        train_button_layout = QHBoxLayout()
        self.train_model_button = QPushButton("训练模型")
        self.save_model_button = QPushButton("保存模型")
        self.load_model_button = QPushButton("加载模型")
        train_button_layout.addWidget(self.train_model_button)
        train_button_layout.addWidget(self.save_model_button)
        train_button_layout.addWidget(self.load_model_button)
        
        # 将所有布局添加到模型训练区域
        model_training_layout.addLayout(data_source_layout)
        model_training_layout.addLayout(external_data_layout)
        model_training_layout.addLayout(training_params_layout)
        model_training_layout.addLayout(train_button_layout)
        
        # 添加模型信息区域
        model_info_group = QGroupBox("模型信息")
        model_info_layout = QVBoxLayout(model_info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setPlaceholderText("训练模型后将显示模型信息和评估结果...")
        model_info_layout.addWidget(self.model_info_text)
        
        # 添加模型应用区域
        model_application_group = QGroupBox("模型应用")
        model_application_layout = QVBoxLayout(model_application_group)
        
        # 使用模型进行检测的按钮
        apply_model_layout = QHBoxLayout()
        self.apply_model_button = QPushButton("使用机器学习模型进行对齐检测")
        self.apply_model_button.setEnabled(False)  # 初始状态禁用
        apply_model_layout.addWidget(self.apply_model_button)
        
        model_application_layout.addLayout(apply_model_layout)
        
        # 将所有组添加到机器学习标签页
        ml_layout.addWidget(model_training_group)
        ml_layout.addWidget(model_info_group)
        ml_layout.addWidget(model_application_group)
        
        # 添加机器学习标签页
        self.tab_widget.addTab(ml_tab, "机器学习")

        # 设置状态栏
        self.statusBar().showMessage("准备就绪")
    
    def setup_menu(self):
        """设置菜单"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        
        # 打开音频文件1
        open_file1_action = QAction("打开音频文件1", self)
        open_file1_action.triggered.connect(self.open_file1)
        file_menu.addAction(open_file1_action)
        
        # 打开音频文件2
        open_file2_action = QAction("打开音频文件2", self)
        open_file2_action.triggered.connect(self.open_file2)
        file_menu.addAction(open_file2_action)
        
        file_menu.addSeparator()
        
        # 批量处理菜单项
        open_batch_ref_action = QAction("打开参考文件组(A)", self)
        open_batch_ref_action.triggered.connect(self.open_ref_files)
        file_menu.addAction(open_batch_ref_action)
        
        open_batch_align_action = QAction("打开对齐文件组(B)", self)
        open_batch_align_action.triggered.connect(self.open_align_files)
        file_menu.addAction(open_batch_align_action)
        
        file_menu.addSeparator()
        
        # 最近文件子菜单
        self.recent_files_menu = QMenu("最近文件", self)
        file_menu.addMenu(self.recent_files_menu)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = self.menuBar().addMenu("工具")
        
        # 设置
        settings_action = QAction("设置", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """设置控件信号连接"""
        # 文件选择按钮
        self.file1_button.clicked.connect(self.open_file1)
        self.file2_button.clicked.connect(self.open_file2)
        
        # 批量处理文件选择按钮
        self.ref_files_button.clicked.connect(self.open_ref_files)
        self.align_files_button.clicked.connect(self.open_align_files)
        self.ref_files_multi_button.clicked.connect(self.open_ref_files_multi)
        self.align_files_multi_button.clicked.connect(self.open_align_files_multi)
        
        # 分析按钮
        self.analyze_button.clicked.connect(self.analyze_alignment)
        self.batch_analyze_button.clicked.connect(self.batch_analyze_alignment)
        
        # Excel导出按钮
        self.export_excel_button.clicked.connect(self.export_to_excel)
        
        # 历史记录相关按钮
        self.save_history_button.clicked.connect(self.save_current_history)
        self.view_history_button.clicked.connect(self.view_history_records)
        
        # 波形视图信号
        if hasattr(self.waveform_view, 'timeSelected'):
            self.waveform_view.timeSelected.connect(self.on_waveform_time_selected)
        if hasattr(self.waveform_view, 'clicked'):
            self.waveform_view.clicked.connect(self.on_waveform_clicked)
        if hasattr(self.waveform_view, 'on_time_selected'):
            self.waveform_view.on_time_selected.connect(self.on_waveform_time_selected)
        if hasattr(self.waveform_view, 'on_clicked'):
            self.waveform_view.on_clicked.connect(self.on_waveform_clicked)
        
        # 统一的播放按钮
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        
        # 音量控制
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        
        # 批量处理表格
        self.batch_results_table.cellDoubleClicked.connect(self.on_batch_table_double_clicked)
        self.batch_results_table.cellClicked.connect(self.on_batch_table_cell_clicked)
        self.batch_results_table.cellEntered.connect(self.on_batch_table_cell_entered)
        
        # ==== 新增: 反馈标记标签页信号连接 ====
        # 从批量检测导入结果
        self.import_batch_results_button.clicked.connect(self.import_batch_results)
        # 保存和加载反馈数据
        self.save_feedback_button.clicked.connect(self.save_feedback_data)
        self.load_feedback_button.clicked.connect(self.load_feedback_data)
        # 应用反馈按钮
        self.apply_feedback_button.clicked.connect(self.apply_feedback)
        # 反馈表格选择行事件
        self.feedback_table.cellClicked.connect(self.on_feedback_table_cell_clicked)
        # 导入参考数据并自动反馈
        self.import_reference_button.clicked.connect(self.import_reference_data)
        
        # ==== 新增: 数据提取标签页信号连接 ====
        # 文件选择
        self.extract_file_button.clicked.connect(self.select_extract_file)
        # 加载列
        self.load_columns_button.clicked.connect(self.load_file_columns)
        # 预览数据
        self.preview_data_button.clicked.connect(self.preview_extract_data)
        # 提取并保存
        self.extract_data_button.clicked.connect(self.extract_and_save_data)
        
        # ==== 新增: 机器学习标签页信号连接 ====
        # 数据源选择
        self.data_source_combo.currentIndexChanged.connect(self.on_data_source_changed)
        # 外部数据文件选择
        self.external_data_button.clicked.connect(self.select_external_data_file)
        # 训练模型
        self.train_model_button.clicked.connect(self.train_model)
        # 保存模型
        self.save_model_button.clicked.connect(self.save_model)
        # 加载模型
        self.load_model_button.clicked.connect(self.load_model)
        # 应用模型
        self.apply_model_button.clicked.connect(self.apply_ml_model)
    
    def open_file1(self):
        """打开音频文件1"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件1", "", "音频文件 (*.wav *.mp3 *.flac *.aac *.ogg);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.audio_processor1.load_file(file_path)
                self.file1_label.setText(get_file_name(file_path))
                self.file1_path_edit.setText(file_path)
                self.audio_player1.load_file(file_path)
                self.update_waveform()
                self.add_recent_file(file_path)
                self.statusBar().showMessage(f"已加载文件1: {get_file_name(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载音频文件1: {str(e)}")
    
    def open_file2(self):
        """打开音频文件2"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件2", "", "音频文件 (*.wav *.mp3 *.flac *.aac *.ogg);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.audio_processor2.load_file(file_path)
                self.file2_label.setText(get_file_name(file_path))
                self.file2_path_edit.setText(file_path)
                self.audio_player2.load_file(file_path)
                self.update_waveform()
                self.add_recent_file(file_path)
                self.statusBar().showMessage(f"已加载文件2: {get_file_name(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载音频文件2: {str(e)}")
    
    def open_ref_files(self):
        """打开参考文件组（A组）文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择参考文件组文件夹")
        if dir_path:
            try:
                # 保存目录路径
                self.ref_dir_path = dir_path
                
                # 获取文件夹中的所有音频文件
                files = self._get_audio_files_in_dir(dir_path)
                if not files:
                    QMessageBox.warning(self, "警告", f"所选文件夹中没有支持的音频文件: {dir_path}")
                    return
                
                # 显示文件列表
                file_list = [os.path.basename(f) for f in files]
                file_list_str = "\n".join(file_list[:10])
                if len(files) > 10:
                    file_list_str += f"\n... 等共 {len(files)} 个文件"
                
                # 更新标签和状态
                self.ref_files_label.setText(f"已选择 {len(files)} 个参考文件")
                self.statusBar().showMessage(f"已加载 {len(files)} 个参考文件")
                
                # 匹配文件
                self._match_batch_files()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载参考文件时出错: {str(e)}")
    
    def open_align_files(self):
        """打开对齐文件组（B组）文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择对齐文件组文件夹")
        if dir_path:
            try:
                # 保存目录路径
                self.align_dir_path = dir_path
                
                # 获取文件夹中的所有音频文件
                files = self._get_audio_files_in_dir(dir_path)
                if not files:
                    QMessageBox.warning(self, "警告", f"所选文件夹中没有支持的音频文件: {dir_path}")
                    return
                
                # 显示文件列表
                file_list = [os.path.basename(f) for f in files]
                file_list_str = "\n".join(file_list[:10])
                if len(files) > 10:
                    file_list_str += f"\n... 等共 {len(files)} 个文件"
                
                # 更新标签和状态
                self.align_files_label.setText(f"已选择 {len(files)} 个对齐文件")
                self.statusBar().showMessage(f"已加载 {len(files)} 个对齐文件")
                
                # 匹配文件
                self._match_batch_files()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载对齐文件时出错: {str(e)}")
    
    def _get_audio_files_in_dir(self, dir_path):
        """获取目录中的所有音频文件"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
        files = []
        
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in audio_extensions:
                files.append(file_path)
                
        return files
    
    def _match_batch_files(self):
        """匹配批量处理文件"""
        # 刷新参考文件组和对齐文件组文件列表
        ref_files = []
        align_files = []
        
        # 获取文件列表
        if hasattr(self, 'ref_dir_path') and self.ref_dir_path:
            if os.path.exists(self.ref_dir_path):
                ref_files = self._get_audio_files_in_dir(self.ref_dir_path)
                self.statusBar().showMessage(f"参考文件组: 找到 {len(ref_files)} 个音频文件")
            else:
                self.statusBar().showMessage("参考文件组路径不存在")
                
        if hasattr(self, 'align_dir_path') and self.align_dir_path:
            if os.path.exists(self.align_dir_path):
                align_files = self._get_audio_files_in_dir(self.align_dir_path)
                self.statusBar().showMessage(f"对齐文件组: 找到 {len(align_files)} 个音频文件")
            else:
                self.statusBar().showMessage("对齐文件组路径不存在")
        
        # 如果两组都有文件，进行匹配
        if ref_files and align_files:
            # 获取ID匹配模式
            id_pattern = self.config_manager.get('id_pattern', r'(\d+)')
            
            # 从文件名中提取ID
            ref_ids = {}
            align_ids = {}
            
            # 记录匹配情况的调试信息
            debug_info = ["文件ID匹配情况:"]
            
            for file_path in ref_files:
                file_name = get_file_basename(file_path)
                # 直接尝试从文件名开头提取数字序列
                digits = ""
                for char in file_name:
                    if char.isdigit():
                        digits += char
                    elif digits:  # 如果已经有数字但遇到非数字，则结束提取
                        break
                
                if digits:
                    file_id = digits
                    ref_ids[file_id] = file_path
                    debug_info.append(f"参考文件: {file_name} → ID: {file_id}")
                else:
                    # 尝试使用正则表达式
                    match = re.search(id_pattern, file_name)
                    if match:
                        file_id = match.group(1)
                        ref_ids[file_id] = file_path
                        debug_info.append(f"参考文件: {file_name} → ID: {file_id} (正则)")
                    else:
                        debug_info.append(f"参考文件: {file_name} → 无法提取ID")
            
            for file_path in align_files:
                file_name = get_file_basename(file_path)
                # 直接尝试从文件名开头提取数字序列
                digits = ""
                for char in file_name:
                    if char.isdigit():
                        digits += char
                    elif digits:  # 如果已经有数字但遇到非数字，则结束提取
                        break
                
                if digits:
                    file_id = digits
                    align_ids[file_id] = file_path
                    debug_info.append(f"对齐文件: {file_name} → ID: {file_id}")
                else:
                    # 尝试使用正则表达式
                    match = re.search(id_pattern, file_name)
                    if match:
                        file_id = match.group(1)
                        align_ids[file_id] = file_path
                        debug_info.append(f"对齐文件: {file_name} → ID: {file_id} (正则)")
                    else:
                        debug_info.append(f"对齐文件: {file_name} → 无法提取ID")
            
            # 找到共同的ID
            common_ids = set(ref_ids.keys()) & set(align_ids.keys())
            debug_info.append(f"共同ID数量: {len(common_ids)}")
            
            # 构建匹配的文件对
            self.batch_files = []
            for file_id in common_ids:
                self.batch_files.append({
                    'id': file_id,
                    'ref_file': ref_ids[file_id],
                    'align_file': align_ids[file_id]
                })
                debug_info.append(f"匹配对: ID {file_id} - {os.path.basename(ref_ids[file_id])} ↔ {os.path.basename(align_ids[file_id])}")
            
            # 更新状态栏
            self.statusBar().showMessage(f"已匹配 {len(self.batch_files)} 对文件")
            
            # 如果没有匹配的文件，显示警告和调试信息
            if not self.batch_files:
                debug_message = "\n".join(debug_info)
                QMessageBox.warning(self, "警告", f"没有找到匹配的文件对，请检查文件命名模式和正则表达式设置\n\n调试信息:\n{debug_message}")
            else:
                # 预先填充表格，方便用户查看匹配结果
                self.batch_results_table.setRowCount(0)
                
                # 临时禁用表格排序，防止添加数据时的排序问题
                self.batch_results_table.setSortingEnabled(False)
                
                for file_pair in self.batch_files:
                    row = self.batch_results_table.rowCount()
                    self.batch_results_table.insertRow(row)
                    # 使用自定义数字排序表格项
                    self.batch_results_table.setItem(row, 0, NumericTableWidgetItem(file_pair['id'], file_pair['id']))
                    self.batch_results_table.setItem(row, 1, QTableWidgetItem(get_file_name(file_pair['ref_file'])))
                    self.batch_results_table.setItem(row, 2, QTableWidgetItem(get_file_name(file_pair['align_file'])))
                    self.batch_results_table.setItem(row, 3, QTableWidgetItem("待分析"))
                    self.batch_results_table.setItem(row, 4, QTableWidgetItem("待分析"))
                
                # 填充完毕后重新启用排序
                self.batch_results_table.setSortingEnabled(True)
                # 默认按ID列升序排序
                self.batch_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
    
    def analyze_alignment(self):
        """分析两个音频文件的对齐情况"""
        if not hasattr(self, 'audio_processor1') or not hasattr(self, 'audio_processor2'):
            QMessageBox.warning(self, "警告", "请先加载音频文件")
            return
            
        if self.audio_processor1.audio_data is None or self.audio_processor2.audio_data is None:
            QMessageBox.warning(self, "警告", "请确保两个音频文件都已加载")
            return
            
        # 获取音频数据
        data1 = self.audio_processor1.get_audio_data()
        data2 = self.audio_processor2.get_audio_data()
        
        try:
            # 分析对齐情况
            result = self.alignment_detector.analyze(data1, data2)
            self.alignment_result = result
            
            # 更新波形图显示
            self.update_waveform(result['offset'])
            
            # 准备结果显示文本 - 简化显示，只显示主要结果
            result_text = ""
            if result['is_aligned']:
                result_text = f"音频已对齐 ✓"
            else:
                result_text = f"音频未对齐 ✗"
            
            # 添加时间偏移信息，统一使用秒为单位
            offset_sec = result['offset']
            result_text += f"\n时间偏移: {offset_sec:.6f} 秒"
            result_text += f"\n阈值设置: {result['threshold_ms']/1000:.6f} 秒"
            
            # 检查音频时长是否一致
            has_duration_mismatch = result.get('has_duration_mismatch', False)
            if has_duration_mismatch:
                duration1 = result.get('duration1', 0)
                duration2 = result.get('duration2', 0)
                duration_diff = result.get('duration_diff', 0)
                result_text += f"\n音频时长不一致: 差异 {duration_diff:.2f} 秒"
                result_text += f"\n音频1时长: {duration1:.2f} 秒"
                result_text += f"\n音频2时长: {duration2:.2f} 秒"
            
            # 更新结果显示
            self.result_text.setText(result_text)
            
            # 根据结果设置结果文本颜色
            if result['is_aligned']:
                has_duration_mismatch = result.get('has_duration_mismatch', False)
                duration_diff = result.get('duration_diff', 0)
                
                if has_duration_mismatch:
                    self.result_text.setStyleSheet("color: purple; font-weight: bold;")
                    
                    # 如果时长差异大于1秒，添加额外警告
                    if duration_diff > 1.0:
                        self.result_text.setStyleSheet("color: purple; font-weight: bold; background-color: #ffdddd;")
                        result_text += f"\n警告: 音频时长差异较大 ({duration_diff:.2f} 秒)"
                else:
                    self.result_text.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.result_text.setStyleSheet("color: red; font-weight: bold;")
            
            # 添加详细分析按钮
            detail_button = QPushButton("查看详细分析")
            detail_button.clicked.connect(lambda: self.show_detail_analysis(result))
            results_layout = self.result_text.parent().layout()
            
            # 移除之前的按钮（如果有）
            for i in range(results_layout.count()):
                widget = results_layout.itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == "查看详细分析":
                    widget.deleteLater()
                    break
            
            results_layout.addWidget(detail_button)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析过程中出错: {str(e)}")
            traceback.print_exc()
    
    def batch_analyze_alignment(self):
        """批量分析音频文件对齐情况"""
        if not self.batch_files:
            QMessageBox.warning(self, "警告", "没有匹配的文件对，请先选择文件夹并确保有匹配的文件")
            return
        
        # 清空结果表格
        self.batch_results_table.setRowCount(0)
        self.batch_results = []
        
        # 重置统计数据
        self.aligned_count = 0
        self.not_aligned_count = 0
        self.error_count = 0
        
        # 重置并设置进度条
        total_files = len(self.batch_files)
        self.batch_progress_bar.setRange(0, total_files)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"%p% - 正在处理: %v/{total_files}")
        
        # 批量处理开始前禁用表格排序，防止边添加数据边排序导致的显示问题
        self.batch_results_table.setSortingEnabled(False)
        
        # 逐对分析文件
        for index, file_pair in enumerate(self.batch_files):
            try:
                # 更新进度条
                self.batch_progress_bar.setValue(index)
                QApplication.processEvents()  # 让UI有机会刷新
                
                # 更新状态栏
                self.statusBar().showMessage(f"正在分析第 {index+1}/{total_files} 对文件...")
                
                # 加载文件
                processor1 = AudioProcessor()
                processor2 = AudioProcessor()
                
                processor1.load_file(file_pair['ref_file'])
                processor2.load_file(file_pair['align_file'])
                
                # 获取音频数据
                data1 = processor1.get_audio_data()
                data2 = processor2.get_audio_data()
                
                # 使用全局的alignment_detector进行分析
                result = self.alignment_detector.analyze(data1, data2)
                
                # 更新统计数据
                if result['is_aligned']:
                    self.aligned_count += 1
                else:
                    self.not_aligned_count += 1
                
                # 添加到结果列表，保存完整的分析结果
                result_item = result.copy()
                result_item.update({
                    'id': file_pair['id'],
                    'ref_file': file_pair['ref_file'],
                    'align_file': file_pair['align_file']
                })
                
                # 提取更多有价值的特征数据，确保保存到结果中
                if 'segment_info' in result:
                    segment_info = result['segment_info']
                    # 添加分段偏移统计数据
                    result_item['segment_offset_std'] = segment_info.get('offset_std_dev', 0)
                    result_item['segment_offset_max_diff'] = segment_info.get('max_offset_diff', 0)
                    result_item['segment_offset_mean'] = segment_info.get('mean_offset', 0)
                    result_item['trend_consistency'] = segment_info.get('trend_consistency', 0)
                
                # 确保峰值相关数据存在
                if 'consistency' in result:
                    result_item['peak_consistency'] = result['consistency']
                if 'num_peaks' in result:
                    result_item['peak_count'] = result['num_peaks']
                if 'num_peaks_used' in result:
                    result_item['used_peak_count'] = result['num_peaks_used']
                
                self.batch_results.append(result_item)
                
                # 添加到表格
                self._add_result_to_table(self.batch_results_table, index, result, file_pair['ref_file'], file_pair['align_file'])
                
                # 更新统计标签
                self._update_stats_labels()
                
            except Exception as e:
                # 出错时添加错误信息
                self.error_count += 1
                
                error_item = {
                    'id': file_pair['id'],
                    'ref_file': file_pair['ref_file'],
                    'align_file': file_pair['align_file'],
                    'offset': None,
                    'offset_ms': None,
                    'is_aligned': False,
                    'error': str(e)
                }
                self.batch_results.append(error_item)
                self._add_result_to_table(self.batch_results_table, index, None, file_pair['ref_file'], file_pair['align_file'])
                
                # 更新统计标签
                self._update_stats_labels()
        
        # 完成后设置进度条满值
        self.batch_progress_bar.setValue(total_files)
        
        # 启用导出按钮
        if self.batch_results:
            self.export_excel_button.setEnabled(True)
            self.save_history_button.setEnabled(True)
        
        # 更新状态栏
        self.statusBar().showMessage(f"批量分析完成，共 {total_files} 对文件，{self.aligned_count} 对对齐，{self.not_aligned_count} 对不对齐，{self.error_count} 对出错")
        
        # 处理完所有数据后重新启用排序功能
        self.batch_results_table.setSortingEnabled(True)
        # 按ID列排序
        self.batch_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
    
    def _update_stats_labels(self):
        """更新统计标签"""
        total = len(self.batch_results)
        self.aligned_count_label.setText(f"对齐文件数: {self.aligned_count}")
        self.not_aligned_count_label.setText(f"不对齐文件数: {self.not_aligned_count}")
        self.total_count_label.setText(f"总文件对数: {total} (错误: {self.error_count})")
    
    def _add_result_to_table(self, table, row, result, ref_file="", align_file=""):
        """将分析结果添加到表格
        
        Args:
            table: 目标表格控件
            row: 行号
            result: 分析结果字典，如果是错误信息则为None
            ref_file: 参考文件路径
            align_file: 对齐文件路径
        """
        # 确保表格有足够的行
        while table.rowCount() <= row:
            table.insertRow(table.rowCount())
            
        # 如果result为None，表示分析出错
        if result is None:
            id_text = str(row + 1)
            if hasattr(self, 'batch_files') and row < len(self.batch_files) and 'id' in self.batch_files[row]:
                id_text = str(self.batch_files[row]['id'])
                
            # 创建表格项
            table.setItem(row, 0, NumericTableWidgetItem(id_text, id_text))
            
            # 确保显示文件名（这里是修复关键）
            ref_basename = os.path.basename(ref_file) if ref_file else "未知"
            align_basename = os.path.basename(align_file) if align_file else "未知"
            
            table.setItem(row, 1, QTableWidgetItem(ref_basename))
            table.setItem(row, 2, QTableWidgetItem(align_basename))
            
            table.setItem(row, 3, QTableWidgetItem("分析出错"))
            status_item = QTableWidgetItem("错误")
            status_item.setBackground(QColor(200, 200, 200))  # 灰色背景表示错误
            table.setItem(row, 4, status_item)
            table.setItem(row, 5, QTableWidgetItem(""))
            return
            
        # 基本信息
        id_text = str(row + 1)
        if 'id' in result:
            id_text = str(result['id'])
        elif hasattr(self, 'batch_files') and row < len(self.batch_files) and 'id' in self.batch_files[row]:
            id_text = str(self.batch_files[row]['id'])
        
        # 处理偏移信息
        offset_sec = result.get('offset', 0)
        offset_ms = result.get('offset_ms', offset_sec * 1000)
        
        # 确保科学计数法偏移值能正常显示
        if abs(offset_sec) < 0.0001:
            offset_text = f"{offset_sec:.8f} 秒 ({offset_ms:.2f} 毫秒)"
        else:
            offset_text = f"{offset_sec:.6f} 秒 ({offset_ms:.2f} 毫秒)"
            
        # 处理对齐状态
        is_aligned = result.get('is_aligned', False)
        status_text = "对齐" if is_aligned else "不对齐"
        
        # 处理渐进性失调标记
        progressive_misalignment = result.get('progressive_misalignment', False)
        if progressive_misalignment:
            status_text += " (渐变失调)"
        
        # 处理音频时长不一致的情况
        has_duration_mismatch = result.get('has_duration_mismatch', False)
        duration_diff = result.get('duration_diff', 0)
        
        # 确保显示文件名（这里是修复关键）
        ref_basename = os.path.basename(ref_file) if ref_file else "未知"
        align_basename = os.path.basename(align_file) if align_file else "未知"
        
        # 从结果中获取文件名（如果存在）
        if 'ref_file' in result and result['ref_file']:
            ref_basename = os.path.basename(result['ref_file'])
        if 'align_file' in result and result['align_file']:
            align_basename = os.path.basename(result['align_file'])
        
        # 创建表格项，使用数字排序项来确保正确排序
        id_item = NumericTableWidgetItem(id_text, id_text)
        ref_item = QTableWidgetItem(ref_basename)
        align_item = QTableWidgetItem(align_basename)
        offset_item = NumericTableWidgetItem(offset_text, offset_sec)  # 使用秒值作为排序依据
        
        # 处理状态项及其颜色
        if has_duration_mismatch:
            status_text += f" (音频时长不一致: {duration_diff:.2f}秒)"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor(128, 0, 128))  # 紫色文字
        else:
            status_item = QTableWidgetItem(status_text)
        
        detail_item = QTableWidgetItem("详细")
        
        # 设置详细分析列的样式
        detail_item.setForeground(QColor(0, 0, 255))  # 蓝色文字
        detail_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐
        
        # 设置状态列的颜色
        if not is_aligned:
            # 使用不同的红色色调区分不对齐原因
            if progressive_misalignment:
                status_item.setBackground(QColor(255, 150, 150))  # 浅红色背景表示渐进性失调
            else:
                status_item.setBackground(QColor(255, 100, 100))  # 深红色背景表示偏移超过阈值
        else:
            # 检查音频时长不一致（大于1秒）的情况
            if has_duration_mismatch and duration_diff > 1.0:
                status_item.setBackground(QColor(255, 100, 100))  # 红色背景表示时长差异大
            else:
                status_item.setBackground(QColor(100, 255, 100))  # 绿色背景表示对齐
        
        # 添加到表格 - 先设置单元格项，再设置背景色
        table.setItem(row, 0, id_item)
        table.setItem(row, 1, ref_item)
        table.setItem(row, 2, align_item)
        table.setItem(row, 3, offset_item)
        table.setItem(row, 4, status_item)
        table.setItem(row, 5, detail_item)
        
        # 保存完整的分析结果到行的用户数据中
        if result:
            # 将分析结果保存到详细信息项的userData中
            detail_item.setData(Qt.ItemDataRole.UserRole, result)
    
    def update_waveform(self, offset=0):
        """更新波形显示"""
        if self.audio_processor1.is_loaded() and self.audio_processor2.is_loaded():
            # 获取波形数据 - 改用get_audio_data确保与分析方法一致
            data1 = self.audio_processor1.get_audio_data()
            data2 = self.audio_processor2.get_audio_data()
            
            # 确保offset是数值类型
            if offset is None:
                offset = 0
                
            # 更新波形视图
            self.waveform_view.plot_waveforms(data1, data2, offset)
    
    def on_waveform_time_selected(self, time_sec):
        """处理波形时间选择事件"""
        # 设置两个播放器的位置
        if hasattr(self.audio_player1, 'seek_to_time'):
            self.audio_player1.seek_to_time(time_sec)
            self.audio_player2.seek_to_time(time_sec)
        elif hasattr(self.audio_player1, 'set_position'):
            self.audio_player1.set_position(time_sec)
            self.audio_player2.set_position(time_sec)
            
    def on_waveform_clicked(self, time_sec):
        """保持兼容性，重定向到on_waveform_time_selected"""
        self.on_waveform_time_selected(time_sec)
    
    def add_recent_file(self, file_path):
        """添加到最近文件列表"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        
        self.recent_files.insert(0, file_path)
        
        # 限制最近文件数量
        if len(self.recent_files) > 10:
            self.recent_files = self.recent_files[:10]
        
        # 更新菜单
        self.update_recent_files_menu()
        
        # 保存到配置
        self.config_manager.set('recent_files', self.recent_files)
        
    def update_recent_files_menu(self):
        """更新最近文件菜单"""
        self.recent_files_menu.clear()
        
        for file_path in self.recent_files:
            action = QAction(get_file_name(file_path), self)
            action.setData(file_path)
            action.triggered.connect(self.open_recent_file)
            self.recent_files_menu.addAction(action)
            
    def open_recent_file(self):
        """打开最近文件"""
        action = self.sender()
        if action:
            file_path = action.data()
            
            # 确定加载到哪个音频处理器
            if not self.audio_processor1.is_loaded():
                try:
                    self.audio_processor1.load_file(file_path)
                    self.file1_label.setText(get_file_name(file_path))
                    self.file1_path_edit.setText(file_path)
                    self.audio_player1.load_file(file_path)
                    self.update_waveform()
                    self.statusBar().showMessage(f"已加载文件1: {get_file_name(file_path)}")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法加载音频文件1: {str(e)}")
            elif not self.audio_processor2.is_loaded():
                try:
                    self.audio_processor2.load_file(file_path)
                    self.file2_label.setText(get_file_name(file_path))
                    self.file2_path_edit.setText(file_path)
                    self.audio_player2.load_file(file_path)
                    self.update_waveform()
                    self.statusBar().showMessage(f"已加载文件2: {get_file_name(file_path)}")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法加载音频文件2: {str(e)}")
            else:
                # 如果两个都已加载，询问用户替换哪个
                msg_box = QMessageBox()
                msg_box.setWindowTitle("选择替换")
                msg_box.setText("两个音频文件都已加载，请选择要替换的文件。")
                msg_box.addButton("替换文件1", QMessageBox.ButtonRole.AcceptRole)
                msg_box.addButton("替换文件2", QMessageBox.ButtonRole.AcceptRole)
                msg_box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                
                choice = msg_box.exec()
                
                if choice == 0:  # 替换文件1
                    try:
                        self.audio_processor1.load_file(file_path)
                        self.file1_label.setText(get_file_name(file_path))
                        self.file1_path_edit.setText(file_path)
                        self.audio_player1.load_file(file_path)
                        self.update_waveform()
                        self.statusBar().showMessage(f"已加载文件1: {get_file_name(file_path)}")
                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"无法加载音频文件1: {str(e)}")
                elif choice == 1:  # 替换文件2
                    try:
                        self.audio_processor2.load_file(file_path)
                        self.file2_label.setText(get_file_name(file_path))
                        self.file2_path_edit.setText(file_path)
                        self.audio_player2.load_file(file_path)
                        self.update_waveform()
                        self.statusBar().showMessage(f"已加载文件2: {get_file_name(file_path)}")
                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"无法加载音频文件2: {str(e)}")
    
    def show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self, self.config_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 更新设置
            settings = dialog.get_settings()
            for key, value in settings.items():
                self.config_manager.set(key, value)
            
            # 更新对齐检测器的阈值
            new_threshold_ms = int(settings['alignment_threshold'] * 1000)
            self.alignment_detector.threshold_ms = new_threshold_ms
            self.alignment_detector.threshold_sec = settings['alignment_threshold']
            
            # 更新人声排除设置
            self.alignment_detector.enable_vocal_exclusion = settings['enable_vocal_exclusion']
            self.alignment_detector.vocal_energy_ratio_threshold = settings['vocal_energy_ratio_threshold']
            self.alignment_detector.vocal_frame_duration = settings['vocal_frame_duration']
            self.alignment_detector.min_non_vocal_segment_length = settings['min_non_vocal_segment_length']
            
            # 显示当前设置
            print(f"阈值已更新: {self.alignment_detector.threshold_sec}秒 ({self.alignment_detector.threshold_ms}毫秒)")
            print(f"人声排除: {'启用' if self.alignment_detector.enable_vocal_exclusion else '禁用'}")
            if self.alignment_detector.enable_vocal_exclusion:
                print(f"  - 人声能量比阈值: {self.alignment_detector.vocal_energy_ratio_threshold}")
                print(f"  - 分析帧长度: {self.alignment_detector.vocal_frame_duration}秒")
                print(f"  - 最小非人声段落长度: {self.alignment_detector.min_non_vocal_segment_length}秒")
            
            # 保存设置
            self.config_manager.save()
            
            # 如果有批量结果，提醒用户
            if hasattr(self, 'batch_results') and self.batch_results:
                vocal_status = "启用" if settings['enable_vocal_exclusion'] else "禁用"
                QMessageBox.information(
                    self, 
                    "设置已更新", 
                    f"阈值已更新为: {settings['alignment_threshold']}秒 ({new_threshold_ms}毫秒)\n"
                    f"人声排除功能: {vocal_status}\n\n"
                    f"请注意，需要重新分析当前批量文件以使用新设置。"
                )
            else:
                vocal_status = "启用" if settings['enable_vocal_exclusion'] else "禁用"
                self.statusBar().showMessage(f"设置已更新，阈值: {settings['alignment_threshold']}秒，人声排除: {vocal_status}")
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于音频对齐检测器",
            """
            <h1>音频对齐检测器</h1>
            <p>版本：1.0.0</p>
            <p>一个用于检测两个音频文件是否对齐的工具。</p>
            <p>提供单文件对齐检测和批量对齐检测功能。</p>
            """
        )
        
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 保存窗口状态
        self.config_manager.set('window_size', [self.width(), self.height()])
        self.config_manager.set('window_pos', [self.x(), self.y()])
        
        # 保存最近文件列表
        self.config_manager.set('recent_files', self.recent_files)
        
        # 保存配置
        self.config_manager.save()
        
        event.accept()

    def open_ref_files_multi(self):
        """打开多个参考文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个参考文件", "", "音频文件 (*.wav *.mp3 *.flac *.aac *.ogg);;所有文件 (*.*)"
        )
        if file_paths:
            try:
                # 创建临时目录
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_ref")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # 设置参考路径为临时目录
                self.ref_dir_path = temp_dir
                
                # 更新标签和状态
                self.ref_files_label.setText(f"已选择 {len(file_paths)} 个参考文件")
                self.statusBar().showMessage(f"已加载 {len(file_paths)} 个参考文件")
                
                # 保存文件列表
                self.ref_files = file_paths
                
                # 匹配文件
                self._match_batch_files_direct()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载参考文件时出错: {str(e)}")
    
    def open_align_files_multi(self):
        """打开多个对齐文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个对齐文件", "", "音频文件 (*.wav *.mp3 *.flac *.aac *.ogg);;所有文件 (*.*)"
        )
        if file_paths:
            try:
                # 创建临时目录
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_align")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # 设置对齐路径为临时目录
                self.align_dir_path = temp_dir
                
                # 更新标签和状态
                self.align_files_label.setText(f"已选择 {len(file_paths)} 个对齐文件")
                self.statusBar().showMessage(f"已加载 {len(file_paths)} 个对齐文件")
                
                # 保存文件列表
                self.align_files = file_paths
                
                # 匹配文件
                self._match_batch_files_direct()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载对齐文件时出错: {str(e)}")
    
    def _match_batch_files_direct(self):
        """直接匹配选择的文件列表"""
        ref_files = getattr(self, 'ref_files', [])
        align_files = getattr(self, 'align_files', [])
        
        # 如果两组都有文件，进行匹配
        if ref_files and align_files:
            # 获取ID匹配模式
            id_pattern = self.config_manager.get('id_pattern', r'(\d+)')
            
            # 从文件名中提取ID
            ref_ids = {}
            align_ids = {}
            
            # 记录匹配情况的调试信息
            debug_info = ["文件ID匹配情况:"]
            
            for file_path in ref_files:
                file_name = get_file_basename(file_path)
                # 直接尝试从文件名开头提取数字序列
                digits = ""
                for char in file_name:
                    if char.isdigit():
                        digits += char
                    elif digits:  # 如果已经有数字但遇到非数字，则结束提取
                        break
                
                if digits:
                    file_id = digits
                    ref_ids[file_id] = file_path
                    debug_info.append(f"参考文件: {file_name} → ID: {file_id}")
                else:
                    # 尝试使用正则表达式
                    match = re.search(id_pattern, file_name)
                    if match:
                        file_id = match.group(1)
                        ref_ids[file_id] = file_path
                        debug_info.append(f"参考文件: {file_name} → ID: {file_id} (正则)")
                    else:
                        debug_info.append(f"参考文件: {file_name} → 无法提取ID")
            
            for file_path in align_files:
                file_name = get_file_basename(file_path)
                # 直接尝试从文件名开头提取数字序列
                digits = ""
                for char in file_name:
                    if char.isdigit():
                        digits += char
                    elif digits:  # 如果已经有数字但遇到非数字，则结束提取
                        break
                
                if digits:
                    file_id = digits
                    align_ids[file_id] = file_path
                    debug_info.append(f"对齐文件: {file_name} → ID: {file_id}")
                else:
                    # 尝试使用正则表达式
                    match = re.search(id_pattern, file_name)
                    if match:
                        file_id = match.group(1)
                        align_ids[file_id] = file_path
                        debug_info.append(f"对齐文件: {file_name} → ID: {file_id} (正则)")
                    else:
                        debug_info.append(f"对齐文件: {file_name} → 无法提取ID")
            
            # 找到共同的ID
            common_ids = set(ref_ids.keys()) & set(align_ids.keys())
            debug_info.append(f"共同ID数量: {len(common_ids)}")
            
            # 构建匹配的文件对
            self.batch_files = []
            for file_id in common_ids:
                self.batch_files.append({
                    'id': file_id,
                    'ref_file': ref_ids[file_id],
                    'align_file': align_ids[file_id]
                })
                debug_info.append(f"匹配对: ID {file_id} - {os.path.basename(ref_ids[file_id])} ↔ {os.path.basename(align_ids[file_id])}")
            
            # 更新状态栏
            self.statusBar().showMessage(f"已匹配 {len(self.batch_files)} 对文件")
            
            # 如果没有匹配的文件，显示警告和调试信息
            if not self.batch_files:
                debug_message = "\n".join(debug_info)
                QMessageBox.warning(self, "警告", f"没有找到匹配的文件对，请检查文件命名模式和正则表达式设置\n\n调试信息:\n{debug_message}")
            else:
                # 预先填充表格，方便用户查看匹配结果
                self.batch_results_table.setRowCount(0)
                
                # 临时禁用表格排序，防止添加数据时的排序问题
                self.batch_results_table.setSortingEnabled(False)
                
                for file_pair in self.batch_files:
                    row = self.batch_results_table.rowCount()
                    self.batch_results_table.insertRow(row)
                    # 使用自定义数字排序表格项
                    self.batch_results_table.setItem(row, 0, NumericTableWidgetItem(file_pair['id'], file_pair['id']))
                    self.batch_results_table.setItem(row, 1, QTableWidgetItem(get_file_name(file_pair['ref_file'])))
                    self.batch_results_table.setItem(row, 2, QTableWidgetItem(get_file_name(file_pair['align_file'])))
                    self.batch_results_table.setItem(row, 3, QTableWidgetItem("待分析"))
                    self.batch_results_table.setItem(row, 4, QTableWidgetItem("待分析"))
                
                # 填充完毕后重新启用排序
                self.batch_results_table.setSortingEnabled(True)
                # 默认按ID列升序排序
                self.batch_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
    
    def on_batch_table_double_clicked(self, row, column):
        """双击批量结果表格行时的处理"""
        if row < 0 or row >= self.batch_results_table.rowCount():
            return
        
        # 从表格中获取ID
        id_item = self.batch_results_table.item(row, 0)
        if not id_item:
            return
            
        # 获取ID文本
        id_text = id_item.text()
        
        # 在批量结果中查找匹配的ID
        result_item = None
        for item in self.batch_results:
            # 从文件名中提取ID
            ref_file = item.get('ref_file', '')
            if ref_file:
                file_id = re.search(r'(\d+)', os.path.basename(ref_file))
                if file_id and file_id.group(1) == id_text:
                    result_item = item
                    break
        
        if not result_item:
            QMessageBox.warning(self, "警告", f"找不到ID为{id_text}的文件信息")
            return
            
        # 尝试加载对应的文件对
        ref_file = result_item.get('ref_file', '')
        align_file = result_item.get('align_file', '')
        
        if not ref_file or not align_file:
            QMessageBox.warning(self, "警告", "找不到对应的文件信息")
            return
            
        # 加载参考文件
        try:
            self.audio_processor1.load_file(ref_file)
            self.file1_path_edit.setText(ref_file)
            self.file1_label.setText(os.path.basename(ref_file))
            
            # 同步更新音频播放器1
            self.audio_player1.load_file(ref_file)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"无法加载参考文件: {str(e)}")
            return
            
        # 加载对齐文件
        try:
            self.audio_processor2.load_file(align_file)
            self.file2_path_edit.setText(align_file)
            self.file2_label.setText(os.path.basename(align_file))
            
            # 同步更新音频播放器2
            self.audio_player2.load_file(align_file)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"无法加载对齐文件: {str(e)}")
            return
        
        # 如果有分析结果，更新波形和结果显示
        if result_item:
            # 使用正确的偏移量字段
            offset = result_item.get('offset', 0)
            # 确保offset不是None
            if offset is None:
                offset = 0
                
            # 更新波形显示
            self.update_waveform(offset)
            # 更新结果显示
            self.update_results_from_batch(result_item)
        
        # 切换到单文件视图
        self.tab_widget.setCurrentIndex(0)
    
    def update_results_from_batch(self, result):
        """从批量分析结果更新单文件结果显示"""
        if not result:
            self.result_text.setText("无法获取分析结果")
            self.result_text.setStyleSheet("color: red;")
            return
            
        # 准备结果显示文本 - 简化显示，只显示主要结果
        result_text = ""
        if result.get('is_aligned', False):
            result_text = f"音频已对齐 ✓"
        else:
            result_text = f"音频未对齐 ✗"
        
        # 添加时间偏移信息，统一使用秒为单位
        offset_sec = result.get('offset', 0)
        result_text += f"\n时间偏移: {offset_sec:.6f} 秒"
        result_text += f"\n阈值设置: {result.get('threshold_ms', 0)/1000:.6f} 秒"
        
        # 更新结果显示
        self.result_text.setText(result_text)
        
        # 根据结果设置结果文本颜色
        if result.get('is_aligned', False):
            self.result_text.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.result_text.setStyleSheet("color: red; font-weight: bold;")
            
        # 添加详细分析按钮
        detail_button = QPushButton("查看详细分析")
        detail_button.clicked.connect(lambda: self.show_detail_analysis(result))
        results_layout = self.result_text.parent().layout()
        
        # 移除之前的按钮（如果有）
        for i in range(results_layout.count()):
            widget = results_layout.itemAt(i).widget()
            if isinstance(widget, QPushButton) and widget.text() == "查看详细分析":
                widget.deleteLater()
                break
        
        results_layout.addWidget(detail_button)
    
    def toggle_play_pause(self):
        """统一控制两个音频播放器的播放和暂停"""
        # 检查是否有加载的音频文件
        if not hasattr(self, 'audio_processor1') or not self.audio_processor1.is_loaded():
            QMessageBox.warning(self, "警告", "请先加载音频文件1")
            return
            
        if not hasattr(self, 'audio_processor2') or not self.audio_processor2.is_loaded():
            QMessageBox.warning(self, "警告", "请先加载音频文件2")
            return
            
        # 确保音频播放器已加载正确的文件
        if not self.audio_player1.is_loaded():
            try:
                self.audio_player1.load_file(self.audio_processor1.file_path)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法加载音频文件1: {str(e)}")
                return
                
        if not self.audio_player2.is_loaded():
            try:
                self.audio_player2.load_file(self.audio_processor2.file_path)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法加载音频文件2: {str(e)}")
                return
            
        # 根据当前状态切换播放/暂停
        if self.play_pause_button.text() == "播放":
            # 开始播放两个音频
            self.audio_player1.play()
            self.audio_player2.play()
            self.play_pause_button.setText("暂停")
        else:
            # 暂停两个音频
            self.audio_player1.pause()
            self.audio_player2.pause()
            self.play_pause_button.setText("播放")
    
    def on_batch_table_cell_clicked(self, row, column):
        """处理批量结果表格单元格点击事件"""
        # 检查是否点击了详细分析列（第6列）
        if column == 5:
            # 获取行的分析数据
            if row >= 0 and row < self.batch_results_table.rowCount():
                item = self.batch_results_table.item(row, 5)
                if item:
                    # 从单元格的userData中获取分析结果
                    analysis_data = item.data(Qt.ItemDataRole.UserRole)
                    if analysis_data:
                        # 显示详细分析对话框
                        dialog = DetailAnalysisDialog(self, analysis_data)
                        dialog.exec()
    
    def on_batch_table_cell_entered(self, row, column):
        """处理批量结果表格单元格鼠标悬停事件"""
        if column == 5 and row >= 0 and row < self.batch_results_table.rowCount():
            item = self.batch_results_table.item(row, column)
            if item and item.text() == "详细":
                # 设置鼠标形状为手型
                self.batch_results_table.setCursor(Qt.CursorShape.PointingHandCursor)
                # 设置提示文本
                item.setToolTip("点击查看详细分析结果")
            else:
                self.batch_results_table.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.batch_results_table.setCursor(Qt.CursorShape.ArrowCursor)
    
    def show_detail_analysis(self, analysis_data):
        """显示详细分析对话框"""
        dialog = DetailAnalysisDialog(self, analysis_data)
        dialog.exec()

    def on_volume_changed(self, value):
        """处理音量滑块值变化事件"""
        # 更新音频播放器的音量
        self.audio_player1.set_volume(value)
        self.audio_player2.set_volume(value)
        self.volume_value_label.setText(f"{value}%")

    def export_to_excel(self):
        """导出分析结果到Excel表格"""
        if not self.batch_results:
            QMessageBox.warning(self, "警告", "没有可导出的分析结果，请先进行批量分析")
            return
        
        try:
            # 尝试导入pandas，如果未安装则提示用户安装
            try:
                import pandas as pd
            except ImportError:
                QMessageBox.critical(
                    self, 
                    "缺少依赖", 
                    "导出Excel需要安装pandas和openpyxl库。\n"
                    "请使用以下命令安装：\n"
                    "pip install pandas openpyxl"
                )
                return
            
            # 让用户选择保存文件的位置
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出Excel表格", "", "Excel文件 (*.xlsx);;所有文件 (*.*)"
            )
            
            if not file_path:
                return  # 用户取消了保存操作
                
            # 如果文件路径没有扩展名，添加.xlsx扩展名
            if not os.path.splitext(file_path)[1]:
                file_path += ".xlsx"
            
            # 开始导出处理
            self.statusBar().showMessage(f"正在导出Excel表格到 {file_path}...")
            
            # 准备数据
            data = []
            for result in self.batch_results:
                # 提取需要的字段
                row_data = {
                    'ID': result.get('id', ''),
                    '参考文件': os.path.basename(result.get('ref_file', '')),
                    '对齐文件': os.path.basename(result.get('align_file', '')),
                    '对齐状态': '对齐' if result.get('is_aligned', False) else '不对齐',
                    '时间偏移(秒)': result.get('offset', 0),
                    '时间偏移(毫秒)': result.get('offset_ms', 0),
                    '阈值(毫秒)': result.get('threshold_ms', 0),
                    '渐进性失调': '是' if result.get('progressive_misalignment', False) else '否',
                    '相关性强度': result.get('correlation_strength', 0),
                    '低可信度': '是' if result.get('low_confidence', False) else '否',
                    '分析方法': result.get('peak_method', ''),
                }
                
                # 添加更多详细特征数据
                if 'peak_consistency' in result:
                    row_data['峰值一致性'] = result['peak_consistency']
                if 'peak_count' in result:
                    row_data['检测峰值数'] = result['peak_count']
                if 'used_peak_count' in result:
                    row_data['使用峰值数'] = result['used_peak_count']
                if 'segment_offset_std' in result:
                    row_data['分段偏移标准差'] = result['segment_offset_std']
                if 'segment_offset_max_diff' in result:
                    row_data['分段偏移最大差异'] = result['segment_offset_max_diff']
                if 'segment_offset_mean' in result:
                    row_data['分段偏移均值'] = result['segment_offset_mean']
                if 'trend_consistency' in result:
                    row_data['趋势一致性'] = result['trend_consistency']
                if 'is_highly_consistent' in result:
                    row_data['高度一致'] = '是' if result['is_highly_consistent'] else '否'
                
                # 如果有错误，添加错误信息
                if 'error' in result:
                    row_data['错误信息'] = result['error']
                
                data.append(row_data)
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 添加分析统计摘要
            summary_data = {
                'ID': ['统计信息'],
                '总文件对数': [len(self.batch_results)],
                '对齐文件数': [self.aligned_count],
                '不对齐文件数': [self.not_aligned_count],
                '错误文件数': [self.error_count],
                '分析时间': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                '阈值设置(毫秒)': [self.alignment_detector.threshold_ms]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # 创建ExcelWriter对象
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 写入主数据到第一个工作表
                df.to_excel(writer, sheet_name='分析结果', index=False)
                
                # 写入摘要到第二个工作表
                summary_df.to_excel(writer, sheet_name='分析摘要', index=False)
                
                # 自动调整列宽
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for i, col in enumerate(df.columns):
                        # 设置列宽为列标题长度和内容的最大长度
                        max_len = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        )
                        # 添加一些额外空间
                        worksheet.column_dimensions[chr(65 + i)].width = max_len + 4
            
            # 导出成功后更新状态栏
            self.statusBar().showMessage(f"Excel表格已成功导出到 {file_path}")
            
            # 询问用户是否打开已导出的文件
            reply = QMessageBox.question(
                self,
                "导出成功",
                f"Excel表格已成功导出到:\n{file_path}\n\n是否打开此文件？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # 打开Excel文件
                import sys
                import subprocess
                
                if sys.platform == 'win32':
                    os.startfile(file_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(('open', file_path))
                else:  # linux
                    subprocess.call(('xdg-open', file_path))
                    
        except Exception as e:
            # 导出过程中发生错误
            QMessageBox.critical(self, "导出错误", f"导出Excel表格时出错:\n{str(e)}")
            traceback.print_exc()
            self.statusBar().showMessage("导出Excel表格失败")
    
    def on_waveform_time_selected(self, time_sec):
        """处理波形时间选择事件"""
        # 设置两个播放器的位置
        if hasattr(self.audio_player1, 'seek_to_time'):
            self.audio_player1.seek_to_time(time_sec)
            self.audio_player2.seek_to_time(time_sec)
        elif hasattr(self.audio_player1, 'set_position'):
            self.audio_player1.set_position(time_sec)
            self.audio_player2.set_position(time_sec)
            
    def on_waveform_clicked(self, time_sec):
        """保持兼容性，重定向到on_waveform_time_selected"""
        self.on_waveform_time_selected(time_sec) 

    def load_settings(self):
        """从配置中加载设置"""
        self.recent_files = self.config_manager.get('recent_files', [])
        
        # 创建音频处理器
        self.audio_processor1 = AudioProcessor()
        self.audio_processor2 = AudioProcessor()

    # ==== 反馈标记功能相关方法 ====
    
    def import_batch_results(self):
        """从批量检测导入结果到反馈标记"""
        if not self.batch_results:
            QMessageBox.warning(self, "警告", "没有可导入的批量分析结果，请先执行批量对齐检测")
            return
        
        # 询问是否覆盖现有数据
        if self.feedback_results:
            reply = QMessageBox.question(
                self, 
                "确认导入", 
                "导入将覆盖现有反馈数据，确定要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # 清空现有数据
        self.feedback_results = []
        self.feedback_table.setRowCount(0)
        
        # 导入数据并填充表格
        for result in self.batch_results:
            feedback_item = {
                'id': result.get('id', ''),
                'ref_file': result.get('ref_file', ''),
                'align_file': result.get('align_file', ''),
                'offset_seconds': result.get('offset', 0),
                'offset_ms': result.get('offset_ms', 0),
                'system_status': '对齐' if result.get('is_aligned', False) else '不对齐',
                'actual_status': '',  # 初始为空
                'feedback': '',       # 初始为空
                'remark': '',         # 初始为空
                'analysis_data': result.copy()  # 保存完整分析数据
            }
            self.feedback_results.append(feedback_item)
            
            # 添加到表格
            self._add_feedback_to_table(len(self.feedback_results) - 1)
            
        # 移除自动切换到反馈标记标签页的代码
        # self.tab_widget.setCurrentIndex(2)  # 不再自动切换标签页
        
        # 更新状态栏
        self.statusBar().showMessage(f"成功导入 {len(self.feedback_results)} 条数据到反馈标记")
    
    def _add_feedback_to_table(self, row_index):
        """将反馈数据添加到表格
        
        Args:
            row_index: 反馈数据的索引
        """
        if row_index < 0 or row_index >= len(self.feedback_results):
            return
            
        feedback = self.feedback_results[row_index]
        
        # 计算要插入的行
        current_row = self.feedback_table.rowCount()
        if current_row <= row_index:
            self.feedback_table.insertRow(row_index)
        
        # 创建表格项
        id_item = QTableWidgetItem(str(feedback['id']))
        ref_file_item = QTableWidgetItem(os.path.basename(feedback['ref_file']))
        align_file_item = QTableWidgetItem(os.path.basename(feedback['align_file']))
        offset_item = QTableWidgetItem(f"{feedback['offset_seconds']:.6f}")
        system_status_item = QTableWidgetItem(feedback['system_status'])
        actual_status_item = QTableWidgetItem(feedback['actual_status'])
        feedback_item = QTableWidgetItem(feedback['feedback'])
        remark_item = QTableWidgetItem(feedback['remark'])
        
        # 设置状态列的颜色
        if feedback['system_status'] == '对齐':
            system_status_item.setBackground(QColor(100, 255, 100))  # 绿色背景表示对齐
        else:
            system_status_item.setBackground(QColor(255, 100, 100))  # 红色背景表示不对齐
            
        # 设置单元格项
        self.feedback_table.setItem(row_index, 0, id_item)
        self.feedback_table.setItem(row_index, 1, ref_file_item)
        self.feedback_table.setItem(row_index, 2, align_file_item)
        self.feedback_table.setItem(row_index, 3, offset_item)
        self.feedback_table.setItem(row_index, 4, system_status_item)
        self.feedback_table.setItem(row_index, 5, actual_status_item)
        self.feedback_table.setItem(row_index, 6, feedback_item)
        self.feedback_table.setItem(row_index, 7, remark_item)
    
    def on_feedback_table_cell_clicked(self, row, column):
        """处理反馈表格单元格点击事件"""
        # 保存当前选中行
        self.selected_feedback_row = row
        
        if row >= 0 and row < len(self.feedback_results):
            feedback = self.feedback_results[row]
            
            # 更新反馈UI控件状态
            # 设置实际情况单选按钮
            if feedback['actual_status'] == '对齐':
                self.actual_status_aligned.setChecked(True)
            elif feedback['actual_status'] == '不对齐':
                self.actual_status_not_aligned.setChecked(True)
            elif feedback['actual_status'] == '不确定':
                self.actual_status_unsure.setChecked(True)
            else:
                # 如果没有选择，清除所有选择
                self.actual_status_group.setExclusive(False)
                self.actual_status_aligned.setChecked(False)
                self.actual_status_not_aligned.setChecked(False)
                self.actual_status_unsure.setChecked(False)
                self.actual_status_group.setExclusive(True)
            
            # 设置反馈单选按钮
            if feedback['feedback'] == '系统判断正确':
                self.feedback_correct.setChecked(True)
            elif feedback['feedback'] == '系统判断错误':
                self.feedback_incorrect.setChecked(True)
            elif feedback['feedback'] == '不确定':
                self.feedback_unsure.setChecked(True)
            else:
                # 如果没有选择，清除所有选择
                self.feedback_group.setExclusive(False)
                self.feedback_correct.setChecked(False)
                self.feedback_incorrect.setChecked(False)
                self.feedback_unsure.setChecked(False)
                self.feedback_group.setExclusive(True)
                
            # 设置备注文本
            self.remark_text.setPlainText(feedback['remark'])
            
            # 更新状态栏
            self.statusBar().showMessage(f"已选择 ID {feedback['id']} 的数据项")
    
    def apply_feedback(self):
        """应用反馈到当前选中的行"""
        if self.selected_feedback_row < 0 or self.selected_feedback_row >= len(self.feedback_results):
            QMessageBox.warning(self, "警告", "请先选择要添加反馈的数据行")
            return
        
        # 获取用户输入的反馈信息
        actual_status = ""
        if self.actual_status_aligned.isChecked():
            actual_status = "对齐"
        elif self.actual_status_not_aligned.isChecked():
            actual_status = "不对齐"
        elif self.actual_status_unsure.isChecked():
            actual_status = "不确定"
            
        feedback = ""
        if self.feedback_correct.isChecked():
            feedback = "系统判断正确"
        elif self.feedback_incorrect.isChecked():
            feedback = "系统判断错误"
        elif self.feedback_unsure.isChecked():
            feedback = "不确定"
            
        remark = self.remark_text.toPlainText()
        
        # 更新数据
        self.feedback_results[self.selected_feedback_row]['actual_status'] = actual_status
        self.feedback_results[self.selected_feedback_row]['feedback'] = feedback
        self.feedback_results[self.selected_feedback_row]['remark'] = remark
        
        # 更新表格显示
        self.feedback_table.item(self.selected_feedback_row, 5).setText(actual_status)
        self.feedback_table.item(self.selected_feedback_row, 6).setText(feedback)
        self.feedback_table.item(self.selected_feedback_row, 7).setText(remark)
        
        # 根据实际情况设置背景色
        actual_status_item = self.feedback_table.item(self.selected_feedback_row, 5)
        if actual_status == "对齐":
            actual_status_item.setBackground(QColor(100, 255, 100))  # 绿色
        elif actual_status == "不对齐":
            actual_status_item.setBackground(QColor(255, 100, 100))  # 红色
        else:
            actual_status_item.setBackground(QColor(200, 200, 200))  # 灰色
            
        # 根据反馈选项设置背景色
        feedback_item = self.feedback_table.item(self.selected_feedback_row, 6)
        if feedback == "系统判断正确":
            feedback_item.setBackground(QColor(100, 255, 100))  # 绿色
        elif feedback == "系统判断错误":
            feedback_item.setBackground(QColor(255, 100, 100))  # 红色
        else:
            feedback_item.setBackground(QColor(200, 200, 200))  # 灰色
            
        # 更新状态栏
        self.statusBar().showMessage(f"已应用反馈到 ID {self.feedback_results[self.selected_feedback_row]['id']} 的数据项")
    
    def save_feedback_data(self):
        """保存反馈数据到CSV文件"""
        if not self.feedback_results:
            QMessageBox.warning(self, "警告", "没有反馈数据可以保存")
            return
            
        # 选择保存文件
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存反馈数据",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 确保文件有.csv扩展名
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
                
            # 添加所有基础字段和分析特征字段
            fieldnames = [
                'id', 'ref_file', 'align_file', 'offset_seconds', 'offset_ms',
                'system_status', 'actual_status', 'feedback', 'remark',
                # 以下是分析数据的特征字段
                'progressive_misalignment', 'beat_consistency', 'correlation_strength',
                'peak_consistency', 'low_confidence', 'segment_offset_std',
                'segment_offset_max_diff', 'segment_offset_mean', 'trend_consistency',
                'peak_count', 'used_peak_count', 'peak_ratio', 'is_highly_consistent',
                'offset_corr_product', 'offset_consistency_ratio', 'std_mean_ratio',
                'corr_std_ratio', 'peak_method', 'peak_method_code'
            ]
                
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for item in self.feedback_results:
                    # 创建一个新字典，包含基础字段
                    row = {field: item.get(field, '') for field in fieldnames[:9]}  # 基础字段
                    
                    # 添加分析数据的特征字段
                    analysis_data = item.get('analysis_data', {})
                    if analysis_data:
                        # 添加基本分析特征
                        row['progressive_misalignment'] = 1 if analysis_data.get('progressive_misalignment', False) else 0
                        row['beat_consistency'] = analysis_data.get('beat_consistency', 0)
                        row['correlation_strength'] = analysis_data.get('correlation_strength', 0)
                        row['peak_consistency'] = analysis_data.get('peak_consistency', 0)
                        row['low_confidence'] = 1 if analysis_data.get('low_confidence', False) else 0
                        
                        # 分段信息
                        segment_info = analysis_data.get('segment_info', {})
                        row['segment_offset_std'] = segment_info.get('offset_std_dev', 0)
                        row['segment_offset_max_diff'] = segment_info.get('max_offset_diff', 0)
                        row['segment_offset_mean'] = segment_info.get('mean_offset', 0)
                        row['trend_consistency'] = segment_info.get('trend_consistency', 0)
                        
                        # 峰值信息
                        row['peak_count'] = analysis_data.get('num_peaks', 0)
                        row['used_peak_count'] = analysis_data.get('num_peaks_used', 0)
                        
                        # 计算峰值比例
                        peak_count = analysis_data.get('num_peaks', 0)
                        used_peak_count = analysis_data.get('num_peaks_used', 0)
                        if peak_count > 0:
                            row['peak_ratio'] = used_peak_count / peak_count
                        else:
                            row['peak_ratio'] = 0
                            
                        # 一致性特征
                        row['is_highly_consistent'] = 1 if analysis_data.get('is_highly_consistent', False) else 0
                        
                        # 组合特征
                        offset_ms_abs = abs(item.get('offset_ms', 0))
                        corr_strength = analysis_data.get('correlation_strength', 0)
                        row['offset_corr_product'] = offset_ms_abs * corr_strength
                        
                        # 偏移量与一致性的比值
                        consistency = analysis_data.get('consistency', 0.001)
                        if consistency > 0:
                            row['offset_consistency_ratio'] = offset_ms_abs / (consistency * 1000)
                        else:
                            row['offset_consistency_ratio'] = 0
                            
                        # 标准差与均值的比值
                        std_dev = segment_info.get('offset_std_dev', 0)
                        mean_offset = abs(segment_info.get('mean_offset', 0.001))
                        if mean_offset > 0.001:
                            row['std_mean_ratio'] = std_dev / mean_offset
                        else:
                            row['std_mean_ratio'] = 0
                            
                        # 相关性强度与分段偏移标准差的比值
                        if std_dev > 0.0001:
                            row['corr_std_ratio'] = corr_strength / std_dev
                        else:
                            row['corr_std_ratio'] = 0
                            
                        # 峰值方法编码
                        peak_method = analysis_data.get('peak_method', '')
                        row['peak_method'] = peak_method
                        
                        method_code = 0
                        if peak_method == 'single':
                            method_code = 1
                        elif peak_method == 'single_fallback':
                            method_code = 2
                        elif peak_method == 'best_of_few':
                            method_code = 3
                        elif peak_method == 'multi_peak_avg':
                            method_code = 4
                        row['peak_method_code'] = method_code
                        
                    writer.writerow(row)
                    
            QMessageBox.information(self, "成功", f"反馈数据已保存到：{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存反馈数据时出错：{str(e)}")
    
    def load_feedback_data(self):
        """从CSV文件加载反馈数据"""
        # 选择文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载反馈数据",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 询问是否覆盖现有数据
            if self.feedback_results:
                reply = QMessageBox.question(
                    self, 
                    "确认加载", 
                    "加载将覆盖现有反馈数据，确定要继续吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # 清空现有数据
            self.feedback_results = []
            self.feedback_table.setRowCount(0)
            
            # 读取CSV文件，尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
            success = False
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', newline='', encoding=encoding) as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # 转换数值字段
                            try:
                                offset_seconds = float(row.get('offset_seconds', 0))
                                offset_ms = float(row.get('offset_ms', 0))
                            except (ValueError, TypeError):
                                offset_seconds = 0
                                offset_ms = 0
                                
                            # 创建反馈项
                            feedback_item = {
                                'id': row.get('id', ''),
                                'ref_file': row.get('ref_file', ''),
                                'align_file': row.get('align_file', ''),
                                'offset_seconds': offset_seconds,
                                'offset_ms': offset_ms,
                                'system_status': row.get('system_status', ''),
                                'actual_status': row.get('actual_status', ''),
                                'feedback': row.get('feedback', ''),
                                'remark': row.get('remark', ''),
                                'analysis_data': None  # 初始化为None，下面将重建分析数据
                            }
                            
                            # 检查是否存在分析特征字段，如果存在则重建分析数据结构
                            analysis_fields = [
                                'progressive_misalignment', 'beat_consistency', 'correlation_strength',
                                'peak_consistency', 'low_confidence', 'segment_offset_std',
                                'segment_offset_max_diff', 'segment_offset_mean', 'trend_consistency',
                                'peak_count', 'used_peak_count', 'peak_ratio', 'is_highly_consistent',
                                'offset_corr_product', 'offset_consistency_ratio', 'std_mean_ratio',
                                'corr_std_ratio', 'peak_method', 'peak_method_code'
                            ]
                            
                            # 检查是否存在至少一个分析字段
                            has_analysis_data = any(field in row for field in analysis_fields)
                            
                            if has_analysis_data:
                                # 重建分析数据结构
                                analysis_data = {}
                                
                                # 转换各个特征字段
                                # 布尔型特征
                                for bool_field in ['progressive_misalignment', 'low_confidence', 'is_highly_consistent']:
                                    if bool_field in row and row[bool_field]:
                                        try:
                                            analysis_data[bool_field] = bool(int(row[bool_field]))
                                        except (ValueError, TypeError):
                                            pass
                                
                                # 数值型特征
                                for float_field in ['beat_consistency', 'correlation_strength', 'peak_consistency']:
                                    if float_field in row and row[float_field]:
                                        try:
                                            analysis_data[float_field] = float(row[float_field])
                                        except (ValueError, TypeError):
                                            pass
                                
                                # 分段信息
                                segment_info = {}
                                if 'segment_offset_std' in row and row['segment_offset_std']:
                                    try:
                                        segment_info['offset_std_dev'] = float(row['segment_offset_std'])
                                    except (ValueError, TypeError):
                                        pass
                                        
                                if 'segment_offset_max_diff' in row and row['segment_offset_max_diff']:
                                    try:
                                        segment_info['max_offset_diff'] = float(row['segment_offset_max_diff'])
                                    except (ValueError, TypeError):
                                        pass
                                        
                                if 'segment_offset_mean' in row and row['segment_offset_mean']:
                                    try:
                                        segment_info['mean_offset'] = float(row['segment_offset_mean'])
                                    except (ValueError, TypeError):
                                        pass
                                        
                                if 'trend_consistency' in row and row['trend_consistency']:
                                    try:
                                        segment_info['trend_consistency'] = float(row['trend_consistency'])
                                    except (ValueError, TypeError):
                                        pass
                                        
                                if segment_info:
                                    analysis_data['segment_info'] = segment_info
                                
                                # 峰值信息
                                if 'peak_count' in row and row['peak_count']:
                                    try:
                                        analysis_data['num_peaks'] = int(float(row['peak_count']))
                                    except (ValueError, TypeError):
                                        pass
                                        
                                if 'used_peak_count' in row and row['used_peak_count']:
                                    try:
                                        analysis_data['num_peaks_used'] = int(float(row['used_peak_count']))
                                    except (ValueError, TypeError):
                                        pass
                                
                                # 一般一致性数据
                                if 'offset_consistency_ratio' in row and row['offset_consistency_ratio']:
                                    try:
                                        # 尝试反向计算一致性值
                                        offset_ms_abs = abs(offset_ms)
                                        if offset_ms_abs > 0:
                                            ratio = float(row['offset_consistency_ratio'])
                                            if ratio > 0:
                                                consistency = offset_ms_abs / (ratio * 1000)
                                                analysis_data['consistency'] = consistency
                                    except (ValueError, TypeError, ZeroDivisionError):
                                        pass
                                
                                # 峰值方法
                                if 'peak_method' in row and row['peak_method']:
                                    analysis_data['peak_method'] = row['peak_method']
                                elif 'peak_method_code' in row and row['peak_method_code']:
                                    try:
                                        code = int(float(row['peak_method_code']))
                                        method = ''
                                        if code == 1:
                                            method = 'single'
                                        elif code == 2:
                                            method = 'single_fallback'
                                        elif code == 3:
                                            method = 'best_of_few'
                                        elif code == 4:
                                            method = 'multi_peak_avg'
                                        if method:
                                            analysis_data['peak_method'] = method
                                    except (ValueError, TypeError):
                                        pass
                                
                                # 设置重建的分析数据
                                if analysis_data:
                                    feedback_item['analysis_data'] = analysis_data
                            
                            self.feedback_results.append(feedback_item)
                    
                    success = True
                    break
                except UnicodeDecodeError:
                    # 如果是编码错误，尝试下一种编码
                    continue
                except Exception as e:
                    # 如果不是编码错误，直接抛出
                    if "decode" not in str(e).lower():
                        raise
            
            if not success:
                raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
                
            # 更新表格
            for i in range(len(self.feedback_results)):
                self._add_feedback_to_table(i)
                
            # 切换到反馈标记标签页
            self.tab_widget.setCurrentIndex(2)  # 假设反馈标记是第3个标签页(索引为2)
            
            QMessageBox.information(self, "成功", f"已从 {file_path} 加载 {len(self.feedback_results)} 条反馈数据")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载反馈数据时出错：{str(e)}")

    # ==== 机器学习功能相关方法 ====
    
    def on_data_source_changed(self, index):
        """处理数据源变更"""
        # 如果选择了外部CSV文件，则启用外部数据路径选择控件
        is_external = index == 1  # 1表示"外部CSV文件"选项
        self.external_data_path.setEnabled(is_external)
        self.external_data_button.setEnabled(is_external)
    
    def select_external_data_file(self):
        """选择外部数据文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择训练数据CSV文件",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if file_paths:
            # 将多个文件路径合并为一个字符串，以分号分隔
            self.external_data_path.setText(";".join(file_paths))
    
    def train_model(self):
        """训练机器学习模型"""
        # 根据数据源获取训练数据
        data_source = self.data_source_combo.currentIndex()
        
        if data_source == 0:  # 反馈标记数据
            if not self.feedback_results:
                QMessageBox.warning(self, "警告", "没有可用的反馈标记数据，请先在反馈标记标签页中导入或创建数据")
                return
                
            # 检查数据是否有足够的反馈
            valid_data = [item for item in self.feedback_results if item['actual_status']]
            if len(valid_data) < 10:  # 至少需要10条带有实际标记的数据
                QMessageBox.warning(self, "警告", f"反馈标记数据中只有 {len(valid_data)} 条有效数据(需要有实际对齐情况标记)，建议至少有10条有效数据")
                reply = QMessageBox.question(
                    self, 
                    "继续训练", 
                    "数据量较少可能导致模型性能不佳，是否仍要继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                    
            # 准备训练数据
            training_data = self._prepare_training_data_from_feedback(valid_data)
            
        else:  # 外部CSV文件
            file_paths = self.external_data_path.text().strip()
            if not file_paths:
                QMessageBox.warning(self, "警告", "请选择有效的外部CSV训练数据文件")
                return
            
            # 分割文件路径
            file_path_list = file_paths.split(";")
            
            # 检查所有文件是否存在
            invalid_files = [path for path in file_path_list if not os.path.isfile(path)]
            if invalid_files:
                QMessageBox.warning(self, "警告", f"以下文件不存在或无效:\n{chr(10).join(invalid_files)}")
                return
                
            try:
                # 存储所有有效数据
                all_training_data = None
                
                # 遍历每个文件并处理
                for file_path in file_path_list:
                    try:
                        # 检查CSV文件是否包含最基本的列
                        try:
                            # 先读取CSV看看包含哪些列
                            df = self._read_csv_with_multiple_encodings(file_path)
                            
                            # 检查是否有id列和actual_status列（基本要求）
                            if 'id' in df.columns and 'actual_status' in df.columns:
                                # 尝试转换为反馈数据格式，然后用同样的方法处理
                                print(f"处理文件 {file_path}，检测到包含基本列，尝试转换为反馈数据格式...")
                                
                                # 将CSV数据转换为反馈数据格式
                                converted_feedback_data = self._convert_csv_to_feedback_format(df)
                                
                                if converted_feedback_data and len(converted_feedback_data) > 0:
                                    # 使用与反馈数据相同的处理方法
                                    file_training_data = self._prepare_training_data_from_feedback(converted_feedback_data)
                                    valid_data_count = len(file_training_data['features']) if file_training_data and 'features' in file_training_data else 0
                                    
                                    # 显示处理结果
                                    if valid_data_count > 0:
                                        print(f"成功从CSV转换为反馈数据格式，有效数据: {valid_data_count}条")
                                    else:
                                        # 如果转换后没有有效数据，回退到常规CSV处理
                                        print("从CSV转换为反馈数据格式后没有有效数据，回退到常规CSV处理方式")
                                        file_training_data = self._load_training_data_from_csv(file_path)
                                else:
                                    # 如果转换失败，回退到常规CSV处理
                                    print("从CSV转换为反馈数据格式失败，回退到常规CSV处理方式")
                                    file_training_data = self._load_training_data_from_csv(file_path)
                            else:
                                # 如果CSV不包含基本列，使用常规CSV处理
                                print("CSV文件不包含必要的列，使用常规CSV处理方式")
                                file_training_data = self._load_training_data_from_csv(file_path)
                                
                        except Exception as e:
                            print(f"尝试检查CSV文件 {file_path} 格式时出错: {str(e)}，使用常规CSV处理方式")
                            # 如果检查过程出错，回退到常规CSV处理
                            file_training_data = self._load_training_data_from_csv(file_path)
                        
                        # 合并数据
                        if all_training_data is None:
                            all_training_data = file_training_data.copy()
                        else:
                            # 特征名称可能不完全一致，取并集并填充缺失值
                            set1 = set(all_training_data['feature_names'])
                            set2 = set(file_training_data['feature_names'])
                            missing_in_first = set2 - set1
                            missing_in_second = set1 - set2
                            
                            # 存在差异，但我们允许自动处理
                            if missing_in_first or missing_in_second:
                                print(f"文件 {file_path} 的特征与已加载文件有差异:")
                                if missing_in_first:
                                    print(f"- 第一个文件缺少特征: {', '.join(missing_in_first)}")
                                if missing_in_second:
                                    print(f"- 第二个文件缺少特征: {', '.join(missing_in_second)}")
                                print("正在进行特征合并和填充...")
                                
                                # 创建合并后的特征名称列表（取并集）
                                merged_feature_names = list(all_training_data['feature_names'])
                                
                                # 添加第一个文件缺少的特征
                                for feature in missing_in_first:
                                    merged_feature_names.append(feature)
                                
                                # 为第一个文件中所有样本添加缺失特征（填充0）
                                if missing_in_first:
                                    for i in range(len(all_training_data['features'])):
                                        for _ in missing_in_first:
                                            all_training_data['features'][i].append(0)
                                
                                # 处理第二个文件的特征
                                # 创建特征名称到索引的映射
                                feature2_map = {name: i for i, name in enumerate(file_training_data['feature_names'])}
                                merged_features = []
                                
                                # 调整第二个文件的每个样本
                                for sample in file_training_data['features']:
                                    new_sample = []
                                    # 遍历合并后的所有特征名称
                                    for feature_name in merged_feature_names:
                                        # 如果特征在第二个文件中存在，使用其值
                                        if feature_name in feature2_map:
                                            idx = feature2_map[feature_name]
                                            if idx < len(sample):
                                                new_sample.append(sample[idx])
                                            else:
                                                new_sample.append(0)  # 安全措施，索引超出范围
                                        else:
                                            # 如果特征在第二个文件中不存在，填充0
                                            new_sample.append(0)
                                    merged_features.append(new_sample)
                                
                                # 更新特征数据
                                all_training_data['feature_names'] = merged_feature_names
                                all_training_data['features'].extend(merged_features)
                                all_training_data['labels'].extend(file_training_data['labels'])
                                
                                # 通知用户已自动处理
                                QMessageBox.information(self, "特征差异已处理", 
                                    f"文件 {file_path} 与之前加载的文件特征结构有差异，已自动处理:\n" +
                                    (f"- 第一个文件缺少特征: {', '.join(missing_in_first)}\n" if missing_in_first else "") +
                                    (f"- 第二个文件缺少特征: {', '.join(missing_in_second)}\n" if missing_in_second else "") +
                                    "已自动使用0填充缺失特征。"
                                )
                            else:
                                # 特征集合相同，但可能顺序不同
                                if list(all_training_data['feature_names']) != list(file_training_data['feature_names']):
                                    # 调整第二个文件的特征顺序
                                    feature2_map = {name: i for i, name in enumerate(file_training_data['feature_names'])}
                                    reordered_features = []
                                    
                                    for sample in file_training_data['features']:
                                        new_sample = []
                                        for feature_name in all_training_data['feature_names']:
                                            if feature_name in feature2_map:
                                                idx = feature2_map[feature_name]
                                                if idx < len(sample):
                                                    new_sample.append(sample[idx])
                                                else:
                                                    new_sample.append(0)
                                            else:
                                                new_sample.append(0)
                                        reordered_features.append(new_sample)
                                    
                                    all_training_data['features'].extend(reordered_features)
                                else:
                                    # 特征完全相同，直接合并
                                    all_training_data['features'].extend(file_training_data['features'])
                                
                                # 合并标签
                                all_training_data['labels'].extend(file_training_data['labels'])
                        
                    except Exception as e:
                        QMessageBox.warning(self, "警告", 
                            f"处理文件 {file_path} 时出错: {str(e)}\n已跳过此文件")
                        continue
                
                # 合并完成后的检查
                if all_training_data is None or 'features' not in all_training_data or 'labels' not in all_training_data:
                    QMessageBox.warning(self, "警告", "从CSV文件加载的训练数据无效")
                    return
                
                # 使用合并后的数据
                training_data = all_training_data
                    
                # 检查features和labels是否存在且非空
                if not training_data['features'] or not training_data['labels']:
                    QMessageBox.warning(self, "警告", "从CSV文件加载的训练数据为空")
                    return
                    
                # 获取实际的有效数据数量
                valid_data_count = len(training_data['features'])
                
                # 计算对齐和不对齐的数据数量
                aligned_count = sum(1 for label in training_data['labels'] if label == 1)
                not_aligned_count = sum(1 for label in training_data['labels'] if label == 0)
                
                if valid_data_count < 10:  # 至少需要10条数据
                    QMessageBox.warning(self, "警告", 
                        f"CSV文件中只有 {valid_data_count} 条有效数据，建议至少有10条\n"
                        f"其中对齐数据: {aligned_count}条，不对齐数据: {not_aligned_count}条")
                    reply = QMessageBox.question(
                        self, 
                        "继续训练", 
                        "数据量较少可能导致模型性能不佳，是否仍要继续？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                        QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                else:
                    # 显示加载的数据数量
                    QMessageBox.information(self, "数据加载成功", 
                        f"从 {len(file_path_list)} 个CSV文件中加载了 {valid_data_count} 条有效数据\n"
                        f"其中对齐数据: {aligned_count}条，不对齐数据: {not_aligned_count}条")
            except Exception as e:
                error_msg = "加载外部CSV文件时出错：" + str(e)
                error_msg += "\n\n请确保CSV文件包含必要的列（offset_seconds和actual_status）"
                error_msg += "\n并且actual_status列包含可识别的值，如\"对齐\"和\"不对齐\"。"
                traceback_info = traceback.format_exc()
                
                QMessageBox.critical(self, "数据加载错误", error_msg)
                print(error_msg)
                print(traceback_info)  # 在控制台输出完整错误堆栈
                return
        
        # 设置训练参数
        test_size = self.test_size_spin.value()
        random_seed = self.random_seed_spin.value()
        
        try:
            # 执行模型训练
            self.statusBar().showMessage("正在训练模型...")
            QApplication.processEvents()  # 刷新UI
            
            X = training_data['features']
            y = training_data['labels']
            feature_names = training_data['feature_names']
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )
            
            # 创建并训练随机森林模型
            model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
            model.fit(X_train, y_train)
            
            # 评估模型
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            # 在测试集上做预测
            y_pred = model.predict(X_test)
            
            # 计算特征重要性
            importances = model.feature_importances_
            feature_importance = {feature_names[i]: importances[i] for i in range(len(feature_names))}
            
            # 对特征重要性排序
            self.feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            
            # 保存模型
            self.ml_model = model
            self.model_trained = True
            self.model_features = feature_names
            
            # 更新模型信息文本
            report = classification_report(y_test, y_pred, output_dict=True)
            self._update_model_info(train_accuracy, test_accuracy, len(X_train), len(X_test), report)
            
            # 启用模型应用按钮
            self.apply_model_button.setEnabled(True)
            self.save_model_button.setEnabled(True)
            
            # 更新状态栏
            self.statusBar().showMessage(f"模型训练完成，测试集准确率: {test_accuracy:.2%}")
            
        except Exception as e:
            QMessageBox.critical(self, "训练失败", f"模型训练过程中发生错误：{str(e)}\n{traceback.format_exc()}")
            self.statusBar().showMessage("模型训练失败")
    
    def _update_model_info(self, train_accuracy, test_accuracy, train_size, test_size, report):
        """更新模型信息文本区域"""
        # 格式化信息文本
        info_text = "模型训练信息:\n"
        info_text += f"训练集大小: {train_size}条, 测试集大小: {test_size}条\n"
        info_text += f"训练集准确率: {train_accuracy:.2%}\n"
        info_text += f"测试集准确率: {test_accuracy:.2%}\n\n"
        
        # 添加分类报告
        info_text += "分类性能报告:\n"
        info_text += f"精确率(对齐类): {report['1']['precision']:.2%}\n"
        info_text += f"召回率(对齐类): {report['1']['recall']:.2%}\n"
        info_text += f"F1分数(对齐类): {report['1']['f1-score']:.2%}\n\n"
        
        info_text += f"精确率(不对齐类): {report['0']['precision']:.2%}\n"
        info_text += f"召回率(不对齐类): {report['0']['recall']:.2%}\n"
        info_text += f"F1分数(不对齐类): {report['0']['f1-score']:.2%}\n\n"
        
        # 添加特征重要性
        info_text += "特征重要性排名:\n"
        for feature, importance in self.feature_importance.items():
            info_text += f"{feature}: {importance:.4f}\n"
        
        # 设置文本
        self.model_info_text.setText(info_text)
    
    def _prepare_training_data_from_feedback(self, feedback_data):
        """从反馈数据准备训练数据
        
        Args:
            feedback_data: 含有实际标记的反馈数据列表
            
        Returns:
            dict: 包含特征和标签的训练数据
        """
        features = []
        labels = []
        feature_names = ['offset_seconds', 'offset_ms_abs']
        
        for item in feedback_data:
            # 只使用有明确实际情况标注的数据
            if item['actual_status'] not in ['对齐', '不对齐']:
                continue
                
            # 提取基本特征
            offset_seconds = item['offset_seconds']
            offset_ms_abs = abs(item['offset_ms'])
            
            # 创建特征向量
            feature_vector = [offset_seconds, offset_ms_abs]
            
            # 如果有分析数据，可以添加更多特征
            if item.get('analysis_data'):
                analysis = item['analysis_data']
                
                # 添加是否有渐进性失调特征
                if 'progressive_misalignment' in analysis:
                    progressive = 1 if analysis.get('progressive_misalignment', False) else 0
                    feature_vector.append(progressive)
                    if 'progressive_misalignment' not in feature_names:
                        feature_names.append('progressive_misalignment')
                
                # 添加节拍偏移一致性特征
                if 'beat_consistency' in analysis:
                    beat_consistency = analysis.get('beat_consistency', 0)
                    feature_vector.append(beat_consistency)
                    if 'beat_consistency' not in feature_names:
                        feature_names.append('beat_consistency')
                
                # === 新增特征 ===
                # 添加相关性强度特征
                if 'correlation_strength' in analysis:
                    corr_strength = analysis.get('correlation_strength', 0)
                    feature_vector.append(corr_strength)
                    if 'correlation_strength' not in feature_names:
                        feature_names.append('correlation_strength')
                
                # 添加峰值一致性特征
                if 'peak_consistency' in analysis:
                    peak_consistency = analysis.get('peak_consistency', 0)
                    feature_vector.append(peak_consistency)
                    if 'peak_consistency' not in feature_names:
                        feature_names.append('peak_consistency')
                
                # 添加低可信度标记特征
                if 'low_confidence' in analysis:
                    low_confidence = 1 if analysis.get('low_confidence', False) else 0
                    feature_vector.append(low_confidence)
                    if 'low_confidence' not in feature_names:
                        feature_names.append('low_confidence')
                
                # 添加分段偏移标准差特征
                if 'segment_info' in analysis and 'offset_std_dev' in analysis['segment_info']:
                    segment_std = analysis['segment_info'].get('offset_std_dev', 0)
                    feature_vector.append(segment_std)
                    if 'segment_offset_std' not in feature_names:
                        feature_names.append('segment_offset_std')
                
                # 添加分段偏移最大差异特征
                if 'segment_info' in analysis and 'max_offset_diff' in analysis['segment_info']:
                    segment_max_diff = analysis['segment_info'].get('max_offset_diff', 0)
                    feature_vector.append(segment_max_diff)
                    if 'segment_offset_max_diff' not in feature_names:
                        feature_names.append('segment_offset_max_diff')
                
                # 添加分段偏移均值特征
                if 'segment_info' in analysis and 'mean_offset' in analysis['segment_info']:
                    segment_mean_offset = analysis['segment_info'].get('mean_offset', 0)
                    feature_vector.append(segment_mean_offset)
                    if 'segment_offset_mean' not in feature_names:
                        feature_names.append('segment_offset_mean')
                
                # 添加趋势一致性特征
                if 'segment_info' in analysis and 'trend_consistency' in analysis['segment_info']:
                    trend_consistency = analysis['segment_info'].get('trend_consistency', 0)
                    feature_vector.append(trend_consistency)
                    if 'trend_consistency' not in feature_names:
                        feature_names.append('trend_consistency')
                
                # 添加检测到的峰值数量特征
                if 'num_peaks' in analysis:
                    peak_count = analysis.get('num_peaks', 0)
                    feature_vector.append(peak_count)
                    if 'peak_count' not in feature_names:
                        feature_names.append('peak_count')
                
                # 添加用于计算的峰值数量特征
                if 'num_peaks_used' in analysis:
                    used_peak_count = analysis.get('num_peaks_used', 0)
                    feature_vector.append(used_peak_count)
                    if 'used_peak_count' not in feature_names:
                        feature_names.append('used_peak_count')
                
                # 添加峰值比例特征（用于计算的峰值/检测到的峰值）
                if 'num_peaks' in analysis and 'num_peaks_used' in analysis:
                    peak_count = analysis.get('num_peaks', 0)
                    if peak_count > 0:
                        used_peak_count = analysis.get('num_peaks_used', 0)
                        peak_ratio = used_peak_count / peak_count
                        feature_vector.append(peak_ratio)
                        if 'peak_ratio' not in feature_names:
                            feature_names.append('peak_ratio')
                
                # 添加高度一致特征
                if 'is_highly_consistent' in analysis:
                    highly_consistent = 1 if analysis.get('is_highly_consistent', False) else 0
                    feature_vector.append(highly_consistent)
                    if 'is_highly_consistent' not in feature_names:
                        feature_names.append('is_highly_consistent')
                
                # 创建复合特征 - 偏移量与相关性强度的组合
                if 'correlation_strength' in analysis:
                    offset_corr_product = offset_ms_abs * analysis.get('correlation_strength', 0)
                    feature_vector.append(offset_corr_product)
                    if 'offset_corr_product' not in feature_names:
                        feature_names.append('offset_corr_product')
                
                # 创建复合特征 - 偏移量与一致性的比值
                if 'consistency' in analysis and analysis.get('consistency', 0) > 0:
                    offset_consistency_ratio = offset_ms_abs / (analysis.get('consistency', 0.001) * 1000)
                    feature_vector.append(offset_consistency_ratio)
                    if 'offset_consistency_ratio' not in feature_names:
                        feature_names.append('offset_consistency_ratio')
                
                # 创建复合特征 - 分段偏移标准差与均值的比值
                if 'segment_info' in analysis:
                    segment_info = analysis['segment_info']
                    std_dev = segment_info.get('offset_std_dev', 0)
                    mean_offset = abs(segment_info.get('mean_offset', 0))
                    if mean_offset > 0.001:  # 避免除以接近零的值
                        std_mean_ratio = std_dev / mean_offset
                        feature_vector.append(std_mean_ratio)
                        if 'std_mean_ratio' not in feature_names:
                            feature_names.append('std_mean_ratio')
                
                # 创建复合特征 - 相关性强度与分段偏移标准差的比值
                if 'correlation_strength' in analysis and 'segment_info' in analysis:
                    corr_strength = analysis.get('correlation_strength', 0)
                    std_dev = analysis['segment_info'].get('offset_std_dev', 0)
                    if std_dev > 0.0001:  # 避免除以接近零的值
                        corr_std_ratio = corr_strength / std_dev
                        feature_vector.append(corr_std_ratio)
                        if 'corr_std_ratio' not in feature_names:
                            feature_names.append('corr_std_ratio')
                
                # 分析方法的数值编码
                if 'peak_method' in analysis:
                    method = analysis.get('peak_method', '')
                    method_code = 0
                    if method == 'single':
                        method_code = 1
                    elif method == 'single_fallback':
                        method_code = 2
                    elif method == 'best_of_few':
                        method_code = 3
                    elif method == 'multi_peak_avg':
                        method_code = 4
                    feature_vector.append(method_code)
                    if 'peak_method_code' not in feature_names:
                        feature_names.append('peak_method_code')
            
            # 添加到特征和标签列表
            features.append(feature_vector)
            labels.append(1 if item['actual_status'] == '对齐' else 0)  # 1表示对齐，0表示不对齐
        
        # 确保所有特征向量长度一致
        max_length = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_length:
                # 使用0填充缺失特征
                padded = f + [0] * (max_length - len(f))
                padded_features.append(padded)
            else:
                padded_features.append(f)
        
        return {
            'features': padded_features,
            'labels': labels,
            'feature_names': feature_names
        }
    
    def _load_training_data_from_csv(self, csv_path):
        """从CSV文件加载训练数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            dict: 包含特征和标签的训练数据
        """
        # 读取CSV文件，尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
        df = None
        successful_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                successful_encoding = encoding
                print(f"成功使用编码 {encoding} 读取CSV文件")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                    raise
        
        if df is None:
            raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
        
        # 显示CSV文件的基本信息
        print(f"CSV文件行数: {len(df)}, 列数: {len(df.columns)}")
        print(f"CSV文件列名: {list(df.columns)}")
        
        # 检查必要的列是否存在
        required_columns = ['offset_seconds', 'actual_status']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        # 打印CSV文件中的actual_status的唯一值，用于调试
        try:
            actual_status_values = df['actual_status'].unique()
            print(f"CSV文件中的actual_status唯一值: {actual_status_values}")
            
            # 计算各种状态的数量
            status_counts = df['actual_status'].value_counts()
            print(f"CSV文件中各状态的数量:\n{status_counts}")
        except Exception as e:
            print(f"分析actual_status值时出错: {str(e)}")
        
        # 准备特征和标签
        features = []
        labels = []
        
        # 确定CSV中可用的基础特征列
        base_feature_columns = ['offset_seconds']
        
        # 基础特征 - 毫秒偏移绝对值
        if 'offset_ms' in df.columns:
            df['offset_ms_abs'] = df['offset_ms'].abs()
            base_feature_columns.append('offset_ms_abs')
        
        # 检查已保存的CSV文件中是否包含分析特征
        analysis_features = [
            'progressive_misalignment', 'beat_consistency', 'correlation_strength',
            'peak_consistency', 'low_confidence', 'segment_offset_std',
            'segment_offset_max_diff', 'segment_offset_mean', 'trend_consistency',
            'peak_count', 'used_peak_count', 'is_highly_consistent',
            'peak_method_code'
        ]
        
        available_analysis_features = []
        for feature in analysis_features:
            if feature in df.columns:
                available_analysis_features.append(feature)
        
        # 构建复合特征（与prepare_training_data_from_feedback方法中一致）
        # 首先检查是否有足够的基础特征来构建复合特征
        if 'offset_ms_abs' in df.columns and 'peak_count' in df.columns and 'used_peak_count' in df.columns:
            # 峰值比例特征
            df['peak_ratio'] = df.apply(
                lambda row: row['used_peak_count'] / row['peak_count'] if row['peak_count'] > 0 else 0, 
                axis=1
            )
            
        if 'offset_ms_abs' in df.columns and 'correlation_strength' in df.columns:
            # 偏移量与相关性强度的乘积
            df['offset_corr_product'] = df['offset_ms_abs'] * df['correlation_strength']
            
        if 'offset_ms_abs' in df.columns and 'peak_consistency' in df.columns:
            # 偏移量与一致性的比值
            df['offset_consistency_ratio'] = df.apply(
                lambda row: row['offset_ms_abs'] / (row['peak_consistency'] * 1000) 
                           if row['peak_consistency'] > 0.001 else 0,
                axis=1
            )
            
        if 'segment_offset_std' in df.columns and 'segment_offset_mean' in df.columns:
            # 分段偏移标准差与均值的比值
            df['std_mean_ratio'] = df.apply(
                lambda row: row['segment_offset_std'] / abs(row['segment_offset_mean'])
                           if abs(row['segment_offset_mean']) > 0.001 else 0,
                axis=1
            )
            
        if 'correlation_strength' in df.columns and 'segment_offset_std' in df.columns:
            # 相关性强度与分段偏移标准差的比值
            df['corr_std_ratio'] = df.apply(
                lambda row: row['correlation_strength'] / row['segment_offset_std']
                           if row['segment_offset_std'] > 0.0001 else 0,
                axis=1
            )
        
        # 获取所有特征列
        all_feature_columns = (
            base_feature_columns + 
            available_analysis_features + 
            [col for col in ['peak_ratio', 'offset_corr_product', 'offset_consistency_ratio', 
                            'std_mean_ratio', 'corr_std_ratio'] if col in df.columns]
        )
        
        feature_names = all_feature_columns.copy()
        print(f"将使用的特征列: {feature_names}")
        
        # 定义对齐和不对齐的可能值（支持多种表达方式）
        align_values = ['对齐', '已对齐', '是', '1', 'True', 'true', '正确', '一致', 'Y', 'y', 'Yes', 'yes']
        not_align_values = ['不对齐', '未对齐', '否', '0', 'False', 'false', '错误', '不一致', 'N', 'n', 'No', 'no']
        
        # 记录识别到的数据数量
        total_rows = 0
        valid_rows = 0
        aligned_count = 0
        not_aligned_count = 0
        unknown_status_samples = []
        
        # 处理每一行数据
        for idx, row in df.iterrows():
            total_rows += 1
            try:
                actual_status = str(row['actual_status']).strip()
            except:
                actual_status = ""
            
            # 确定标签（更灵活的匹配）
            is_aligned = None
            
            if actual_status in align_values:
                is_aligned = True
            elif actual_status in not_align_values:
                is_aligned = False
            else:
                # 尝试更模糊的匹配
                status_lower = actual_status.lower()
                if '对齐' in status_lower or '一致' in status_lower or 'align' in status_lower or 'true' in status_lower or 'yes' in status_lower:
                    is_aligned = True
                elif '不对齐' in status_lower or '不一致' in status_lower or 'not align' in status_lower or 'false' in status_lower or 'no' in status_lower:
                    is_aligned = False
            
            # 跳过无法识别标记的数据
            if is_aligned is None:
                if len(unknown_status_samples) < 5:  # 只记录前5个无法识别的样本
                    unknown_status_samples.append(f"行{idx+1}: '{actual_status}'")
                continue
                
            valid_rows += 1
            if is_aligned:
                aligned_count += 1
            else:
                not_aligned_count += 1
            
            # 构建特征向量
            feature_vector = []
            for col in all_feature_columns:
                if col in df.columns:
                    try:
                        # 尝试将特征值转换为浮点数
                        feature_vector.append(float(row[col]))
                    except (ValueError, TypeError):
                        feature_vector.append(0)  # 如果转换失败，使用默认值0
                else:
                    feature_vector.append(0)  # 填充缺失值
            
            # 添加到特征和标签列表
            features.append(feature_vector)
            labels.append(1 if is_aligned else 0)  # 1表示对齐，0表示不对齐
        
        print(f"CSV文件总行数: {total_rows}, 有效数据行数: {valid_rows} (对齐: {aligned_count}, 不对齐: {not_aligned_count})")
        if unknown_status_samples:
            print(f"无法识别的状态值示例: {', '.join(unknown_status_samples)}")
        
        if valid_rows == 0:
            raise ValueError(f"CSV文件中没有有效的训练数据。请确保'actual_status'列包含有效的对齐状态标记，如'对齐'或'不对齐'。")
        
        # 对特征进行标准化处理与反馈数据保持一致
        # 确保所有特征向量长度一致
        max_length = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_length:
                # 使用0填充缺失特征
                padded = f + [0] * (max_length - len(f))
                padded_features.append(padded)
            else:
                padded_features.append(f)
        
        return {
            'features': padded_features,
            'labels': labels,
            'feature_names': feature_names
        }
    
    def save_model(self):
        """保存训练好的模型"""
        if not self.ml_model or not self.model_trained:
            QMessageBox.warning(self, "警告", "没有训练好的模型可以保存")
            return
        
        # 选择保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存机器学习模型",
            "",
            "模型文件 (*.joblib);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 确保文件扩展名
            if not file_path.lower().endswith('.joblib'):
                file_path += '.joblib'
            
            # 创建要保存的模型数据
            model_data = {
                'model': self.ml_model,
                'feature_names': self.model_features,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存模型
            joblib.dump(model_data, file_path)
            
            QMessageBox.information(self, "成功", f"模型已保存到：{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存模型时出错：{str(e)}")
    
    def load_model(self):
        """加载保存的模型"""
        # 选择模型文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载机器学习模型",
            "",
            "模型文件 (*.joblib);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 加载模型
            model_data = joblib.load(file_path)
            
            # 提取模型和相关信息
            self.ml_model = model_data.get('model')
            self.model_features = model_data.get('feature_names', [])
            self.feature_importance = model_data.get('feature_importance', {})
            self.model_trained = True
            
            # 启用模型应用按钮
            self.apply_model_button.setEnabled(True)
            self.save_model_button.setEnabled(True)
            
            # 更新模型信息
            timestamp = model_data.get('timestamp', '未知')
            info_text = f"已加载模型 ({timestamp})\n\n"
            info_text += "模型特征:\n"
            for feature in self.model_features:
                info_text += f"- {feature}\n"
            
            info_text += "\n特征重要性:\n"
            for feature, importance in self.feature_importance.items():
                info_text += f"{feature}: {importance:.4f}\n"
                
            self.model_info_text.setText(info_text)
            
            QMessageBox.information(self, "成功", f"已加载模型文件：{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载模型时出错：{str(e)}")
            self.ml_model = None
            self.model_trained = False
            self.apply_model_button.setEnabled(False)
    
    def apply_ml_model(self):
        """使用机器学习模型进行对齐检测"""
        if not self.ml_model or not self.model_trained:
            QMessageBox.warning(self, "警告", "没有训练好的模型可用，请先训练或加载模型")
            return
        
        # 检查是否有批量处理的文件
        if not self.batch_files:
            QMessageBox.warning(self, "警告", "没有批量文件可以处理，请先在批量对齐检测标签页选择文件")
            return
        
        # 确认执行
        reply = QMessageBox.question(
            self, 
            "确认使用模型", 
            "将使用机器学习模型对当前批量文件进行对齐检测，这将覆盖现有的批量检测结果。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return
        
        try:
            # 移除自动切换到批量检测标签页的代码
            # self.tab_widget.setCurrentIndex(1)  # 不再自动切换标签页
            
            # 清空结果表格
            self.batch_results_table.setRowCount(0)
            self.batch_results = []
            
            # 重置统计数据
            self.aligned_count = 0
            self.not_aligned_count = 0
            self.error_count = 0
            
            # 设置进度条
            total_files = len(self.batch_files)
            self.batch_progress_bar.setRange(0, total_files)
            self.batch_progress_bar.setValue(0)
            self.batch_progress_bar.setFormat(f"%p% - ML模型处理: %v/{total_files}")
            
            # 批量处理开始前禁用表格排序
            self.batch_results_table.setSortingEnabled(False)
            
            # 逐对分析文件
            for index, file_pair in enumerate(self.batch_files):
                try:
                    # 更新进度条
                    self.batch_progress_bar.setValue(index)
                    QApplication.processEvents()  # 让UI有机会刷新
                    
                    # 更新状态栏
                    self.statusBar().showMessage(f"机器学习模型正在分析第 {index+1}/{total_files} 对文件...")
                    
                    # 加载文件
                    processor1 = AudioProcessor()
                    processor2 = AudioProcessor()
                    
                    processor1.load_file(file_pair['ref_file'])
                    processor2.load_file(file_pair['align_file'])
                    
                    # 获取音频数据
                    data1 = processor1.get_audio_data()
                    data2 = processor2.get_audio_data()
                    
                    # 先使用普通方法获取分析结果
                    result = self.alignment_detector.analyze(data1, data2)
                    
                    # 准备特征数据
                    features = []
                    for feature in self.model_features:
                        if feature == 'offset_seconds':
                            features.append(result.get('offset', 0))
                        elif feature == 'offset_ms_abs':
                            offset_ms_abs = abs(result.get('offset_ms', 0))
                            features.append(offset_ms_abs)
                        elif feature == 'progressive_misalignment':
                            features.append(1 if result.get('progressive_misalignment', False) else 0)
                        elif feature == 'beat_consistency':
                            features.append(result.get('beat_consistency', 0))
                        # === 新增特征 ===
                        elif feature == 'correlation_strength':
                            features.append(result.get('correlation_strength', 0))
                        elif feature == 'peak_consistency':
                            features.append(result.get('peak_consistency', 0))
                        elif feature == 'low_confidence':
                            features.append(1 if result.get('low_confidence', False) else 0)
                        elif feature == 'segment_offset_std':
                            if 'segment_info' in result:
                                features.append(result['segment_info'].get('offset_std_dev', 0))
                            else:
                                features.append(0)
                        elif feature == 'segment_offset_max_diff':
                            if 'segment_info' in result:
                                features.append(result['segment_info'].get('max_offset_diff', 0))
                            else:
                                features.append(0)
                        elif feature == 'segment_offset_mean':
                            if 'segment_info' in result:
                                features.append(result['segment_info'].get('mean_offset', 0))
                            else:
                                features.append(0)
                        elif feature == 'trend_consistency':
                            if 'segment_info' in result:
                                features.append(result['segment_info'].get('trend_consistency', 0))
                            else:
                                features.append(0)
                        elif feature == 'peak_count':
                            features.append(result.get('num_peaks', 0))
                        elif feature == 'used_peak_count':
                            features.append(result.get('num_peaks_used', 0))
                        elif feature == 'peak_ratio':
                            peak_count = result.get('num_peaks', 0)
                            if peak_count > 0:
                                used_peak_count = result.get('num_peaks_used', 0)
                                features.append(used_peak_count / peak_count)
                            else:
                                features.append(0)
                        elif feature == 'is_highly_consistent':
                            features.append(1 if result.get('is_highly_consistent', False) else 0)
                        elif feature == 'offset_corr_product':
                            offset_ms_abs = abs(result.get('offset_ms', 0))
                            corr_strength = result.get('correlation_strength', 0)
                            features.append(offset_ms_abs * corr_strength)
                        elif feature == 'offset_consistency_ratio':
                            offset_ms_abs = abs(result.get('offset_ms', 0))
                            consistency = result.get('consistency', 0.001)
                            if consistency > 0:
                                features.append(offset_ms_abs / (consistency * 1000))
                            else:
                                features.append(0)
                        elif feature == 'std_mean_ratio':
                            if 'segment_info' in result:
                                segment_info = result['segment_info']
                                std_dev = segment_info.get('offset_std_dev', 0)
                                mean_offset = abs(segment_info.get('mean_offset', 0))
                                if mean_offset > 0.001:
                                    features.append(std_dev / mean_offset)
                                else:
                                    features.append(0)
                            else:
                                features.append(0)
                        elif feature == 'corr_std_ratio':
                            corr_strength = result.get('correlation_strength', 0)
                            if 'segment_info' in result:
                                std_dev = result['segment_info'].get('offset_std_dev', 0)
                                if std_dev > 0.0001:
                                    features.append(corr_strength / std_dev)
                                else:
                                    features.append(0)
                            else:
                                features.append(0)
                        elif feature == 'peak_method_code':
                            method = result.get('peak_method', '')
                            method_code = 0
                            if method == 'single':
                                method_code = 1
                            elif method == 'single_fallback':
                                method_code = 2
                            elif method == 'best_of_few':
                                method_code = 3
                            elif method == 'multi_peak_avg':
                                method_code = 4
                            features.append(method_code)
                        else:
                            # 对于未知特征，使用0填充
                            features.append(0)
                    
                    # 使用模型预测
                    prediction = self.ml_model.predict([features])[0]
                    
                    # 使用模型的预测覆盖原始结果
                    result['is_aligned'] = bool(prediction)
                    result['ml_prediction'] = bool(prediction)
                    
                    # 更新统计数据
                    if result['is_aligned']:
                        self.aligned_count += 1
                    else:
                        self.not_aligned_count += 1
                    
                    # 添加到结果列表
                    result_item = result.copy()
                    result_item.update({
                        'id': file_pair['id'],
                        'ref_file': file_pair['ref_file'],
                        'align_file': file_pair['align_file']
                    })
                    self.batch_results.append(result_item)
                    
                    # 添加到表格
                    self._add_result_to_table(self.batch_results_table, index, result, file_pair['ref_file'], file_pair['align_file'])
                    
                    # 更新统计标签
                    self._update_stats_labels()
                    
                except Exception as e:
                    # 出错时添加错误信息
                    self.error_count += 1
                    
                    error_item = {
                        'id': file_pair['id'],
                        'ref_file': file_pair['ref_file'],
                        'align_file': file_pair['align_file'],
                        'offset': None,
                        'offset_ms': None,
                        'is_aligned': False,
                        'error': str(e)
                    }
                    self.batch_results.append(error_item)
                    self._add_result_to_table(self.batch_results_table, index, None, file_pair['ref_file'], file_pair['align_file'])
                    
                    # 更新统计标签
                    self._update_stats_labels()
                
            # 完成后设置进度条满值
            self.batch_progress_bar.setValue(total_files)
            
            # 启用导出按钮
            if self.batch_results:
                self.export_excel_button.setEnabled(True)
                self.save_history_button.setEnabled(True)
            
            # 更新状态栏
            self.statusBar().showMessage(f"机器学习模型批量分析完成，共 {total_files} 对文件，{self.aligned_count} 对对齐，{self.not_aligned_count} 对不对齐，{self.error_count} 对出错")
            
            # 处理完所有数据后重新启用排序功能
            self.batch_results_table.setSortingEnabled(True)
            # 按ID列排序
            self.batch_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
            
        except Exception as e:
            QMessageBox.critical(self, "处理失败", f"使用机器学习模型进行批量分析时出错：{str(e)}\n{traceback.format_exc()}")
            self.statusBar().showMessage("机器学习模型批量分析失败")

    # ==== 数据提取功能相关方法 ====
    
    def select_extract_file(self):
        """选择要提取数据的文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            "",
            "CSV文件 (*.csv);;Excel文件 (*.xlsx *.xls);;所有文件 (*.*)"
        )
        
        if file_path:
            self.extract_file_path.setText(file_path)
            # 自动加载列
            self.load_file_columns()
    
    def load_file_columns(self):
        """加载文件的列名"""
        file_path = self.extract_file_path.text().strip()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择数据文件")
            return
            
        try:
            # 读取文件，尝试多种编码
            if file_path.lower().endswith('.csv'):
                # 尝试多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                            raise
                
                if df is None:
                    raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
            else:
                df = pd.read_excel(file_path)
                
            # 清空并更新列选择下拉框
            self.id_column_combo.clear()
            self.status_column_combo.clear()
            
            # 添加列名
            for col in df.columns:
                self.id_column_combo.addItem(str(col))
                self.status_column_combo.addItem(str(col))
                
            # 自动选择可能的ID列和状态列
            for i, col in enumerate(df.columns):
                col_lower = str(col).lower()
                if 'id' in col_lower:
                    self.id_column_combo.setCurrentIndex(i)
                if '对齐' in col_lower or 'align' in col_lower or 'status' in col_lower:
                    self.status_column_combo.setCurrentIndex(i)
                    
            self.statusBar().showMessage(f"成功加载文件列: {len(df.columns)}列, {len(df)}行")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载文件列失败: {str(e)}")
    
    def _convert_status(self, status, align_sources, not_align_sources, target_align, target_not_align):
        """转换状态值，支持多个源文本映射到同一目标值
        
        Args:
            status: 原始状态文本
            align_sources: 表示"对齐"的源文本列表
            not_align_sources: 表示"不对齐"的源文本列表
            target_align: "对齐"的目标文本
            target_not_align: "不对齐"的目标文本
            
        Returns:
            str: 转换后的状态文本
        """
        status_str = str(status).strip()
        
        if status_str in align_sources:
            return target_align
        elif status_str in not_align_sources:
            return target_not_align
        else:
            return status_str
            
    def preview_extract_data(self):
        """预览数据转换结果"""
        file_path = self.extract_file_path.text().strip()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择数据文件")
            return
            
        # 获取列选择
        id_column = self.id_column_combo.currentText()
        status_column = self.status_column_combo.currentText()
        
        if not id_column or not status_column:
            QMessageBox.warning(self, "警告", "请选择ID列和对齐状态列")
            return
            
        try:
            # 读取文件，尝试多种编码
            if file_path.lower().endswith('.csv'):
                # 尝试多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                            raise
                
                if df is None:
                    raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
            else:
                df = pd.read_excel(file_path)
                
            # 获取转换规则，支持逗号分隔多个源文本
            align_text = self.align_text_edit.text()
            not_align_text = self.not_align_text_edit.text()
            target_align_text = self.target_align_text_edit.text()
            target_not_align_text = self.target_not_align_text_edit.text()
            
            # 解析源文本列表
            align_sources = [s.strip() for s in align_text.split(',')]
            not_align_sources = [s.strip() for s in not_align_text.split(',')]
            
            # 只保留需要的列
            df = df[[id_column, status_column]].copy()
            
            # 创建转换后的状态列
            df['转换后状态'] = df[status_column].apply(
                lambda x: self._convert_status(
                    x, 
                    align_sources, 
                    not_align_sources, 
                    target_align_text, 
                    target_not_align_text
                )
            )
            
            # 更新预览表格
            self.preview_table.setRowCount(0)  # 清空表格
            self.preview_table.setRowCount(min(100, len(df)))  # 最多显示100行
            
            for i, (_, row) in enumerate(df.head(100).iterrows()):
                id_item = QTableWidgetItem(str(row[id_column]))
                original_status_item = QTableWidgetItem(str(row[status_column]))
                converted_status_item = QTableWidgetItem(str(row['转换后状态']))
                
                self.preview_table.setItem(i, 0, id_item)
                self.preview_table.setItem(i, 1, original_status_item)
                self.preview_table.setItem(i, 2, converted_status_item)
                
                # 设置背景色
                if row['转换后状态'] == target_align_text:
                    converted_status_item.setBackground(QColor(100, 255, 100))  # 绿色
                elif row['转换后状态'] == target_not_align_text:
                    converted_status_item.setBackground(QColor(255, 100, 100))  # 红色
            
            # 启用提取按钮
            self.extract_data_button.setEnabled(True)
            
            # 更新状态栏
            self.statusBar().showMessage(f"预览数据成功: 共{len(df)}行, 显示前{min(100, len(df))}行")
            
        except Exception as e:
            QMessageBox.critical(self, "预览失败", f"预览数据转换失败: {str(e)}\n{traceback.format_exc()}")
    
    def extract_and_save_data(self):
        """提取并保存转换后的数据"""
        file_path = self.extract_file_path.text().strip()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择数据文件")
            return
            
        # 选择保存文件
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存提取的数据",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if not save_path:
            return
            
        # 确保文件有.csv扩展名
        if not save_path.lower().endswith('.csv'):
            save_path += '.csv'
            
        # 获取列选择
        id_column = self.id_column_combo.currentText()
        status_column = self.status_column_combo.currentText()
        
        try:
            # 读取文件，尝试多种编码
            if file_path.lower().endswith('.csv'):
                # 尝试多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                            raise
                
                if df is None:
                    raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
            else:
                df = pd.read_excel(file_path)
                
            # 获取转换规则，支持逗号分隔多个源文本
            align_text = self.align_text_edit.text()
            not_align_text = self.not_align_text_edit.text()
            target_align_text = self.target_align_text_edit.text()
            target_not_align_text = self.target_not_align_text_edit.text()
            
            # 解析源文本列表
            align_sources = [s.strip() for s in align_text.split(',')]
            not_align_sources = [s.strip() for s in not_align_text.split(',')]
            
            # 只保留需要的列
            result_df = df[[id_column]].copy()
            
            # 创建actual_status列
            result_df['actual_status'] = df[status_column].apply(
                lambda x: self._convert_status(
                    x, 
                    align_sources, 
                    not_align_sources, 
                    target_align_text, 
                    target_not_align_text
                )
            )
            
            # 重命名ID列为id
            result_df.rename(columns={id_column: 'id'}, inplace=True)
            
            # 保存为CSV
            result_df.to_csv(save_path, index=False, encoding='utf-8')
            
            QMessageBox.information(
                self, 
                "保存成功", 
                f"成功提取和转换数据，共{len(result_df)}行，已保存到：{save_path}\n\n"
                f"对齐规则: {', '.join(align_sources)} → {target_align_text}\n"
                f"不对齐规则: {', '.join(not_align_sources)} → {target_not_align_text}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"提取和保存数据失败: {str(e)}\n{traceback.format_exc()}")

    def import_reference_data(self):
        """导入参考数据并自动应用反馈"""
        # 检查是否有批量分析结果
        if not self.feedback_results:
            QMessageBox.warning(self, "警告", "反馈标记中没有数据，请先从批量检测导入结果")
            return
            
        # 选择参考数据文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考数据CSV",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 读取参考数据
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
            reference_df = None
            successful_encoding = None
            
            for encoding in encodings:
                try:
                    reference_df = pd.read_csv(file_path, encoding=encoding)
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                        raise
            
            if reference_df is None:
                raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
            
            # 检查必要的列
            required_columns = ['id', 'actual_status']
            for col in required_columns:
                if col not in reference_df.columns:
                    QMessageBox.warning(self, "警告", f"参考数据文件缺少必要的列: {col}")
                    return
            
            # 转换参考数据的ID列类型为字符串，并且只保留数字部分
            reference_df['id_numeric'] = reference_df['id'].astype(str).apply(
                lambda x: ''.join(filter(str.isdigit, x))  # 只保留数字部分
            )
            
            # 创建ID到实际状态的映射 (使用处理后的数字ID)
            reference_map = dict(zip(reference_df['id_numeric'], reference_df['actual_status']))
            
            # 跟踪匹配和应用情况
            matched_count = 0
            correct_count = 0
            incorrect_count = 0
            unmatched_ids = []
            
            # 对每个反馈结果应用参考数据
            for i, item in enumerate(self.feedback_results):
                # 提取ID的数字部分
                item_id_numeric = ''.join(filter(str.isdigit, str(item['id'])))
                
                # 如果在参考数据中找到匹配项
                if item_id_numeric in reference_map:
                    matched_count += 1
                    actual_status = reference_map[item_id_numeric]
                    
                    # 更新实际情况
                    item['actual_status'] = actual_status
                    self.feedback_table.item(i, 5).setText(str(actual_status))
                    
                    # 设置实际情况背景色
                    actual_status_item = self.feedback_table.item(i, 5)
                    if str(actual_status) == "对齐":
                        actual_status_item.setBackground(QColor(100, 255, 100))  # 绿色
                    elif str(actual_status) == "不对齐":
                        actual_status_item.setBackground(QColor(255, 100, 100))  # 红色
                    else:
                        actual_status_item.setBackground(QColor(200, 200, 200))  # 灰色
                    
                    # 比较系统判定和实际情况
                    system_status = item['system_status']
                    is_system_correct = (system_status == "对齐" and str(actual_status) == "对齐") or \
                                       (system_status == "不对齐" and str(actual_status) == "不对齐")
                    
                    # 设置反馈
                    if is_system_correct:
                        feedback = "系统判断正确"
                        correct_count += 1
                    else:
                        feedback = "系统判断错误"
                        incorrect_count += 1
                    
                    item['feedback'] = feedback
                    self.feedback_table.item(i, 6).setText(feedback)
                    
                    # 设置反馈背景色
                    feedback_item = self.feedback_table.item(i, 6)
                    if feedback == "系统判断正确":
                        feedback_item.setBackground(QColor(100, 255, 100))  # 绿色
                    else:
                        feedback_item.setBackground(QColor(255, 100, 100))  # 红色
                else:
                    unmatched_ids.append(str(item['id']))
            
            # 显示匹配和应用结果
            if matched_count > 0:
                accuracy = (correct_count / matched_count) * 100
                unmatched_msg = ""
                if len(unmatched_ids) > 0:
                    # 最多显示前10个未匹配的ID
                    display_ids = unmatched_ids[:10]
                    if len(unmatched_ids) > 10:
                        display_ids.append("...")
                    unmatched_msg = f"\n\n未匹配ID: {', '.join(display_ids)}"
                    
                QMessageBox.information(
                    self, 
                    "应用完成", 
                    f"成功匹配和应用了{matched_count}条参考数据\n"
                    f"系统判断正确: {correct_count}条 ({accuracy:.2f}%)\n"
                    f"系统判断错误: {incorrect_count}条 ({100-accuracy:.2f}%)"
                    f"{unmatched_msg}"
                )
                self.statusBar().showMessage(
                    f"已应用参考数据: {matched_count}条匹配, "
                    f"{correct_count}条判断正确 ({accuracy:.2f}%), "
                    f"{incorrect_count}条判断错误"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "无匹配", 
                    "未找到与反馈数据匹配的参考数据，请检查ID是否一致"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入参考数据并应用反馈时出错: {str(e)}\n{traceback.format_exc()}")

    # ==== 历史记录功能相关方法 ====
    
    def save_current_history(self):
        """保存当前批量分析结果到历史记录"""
        if not self.batch_results:
            QMessageBox.warning(self, "警告", "没有可保存的批量分析结果")
            return
            
        # 创建保存对话框，让用户输入描述
        description, ok = QInputDialog.getText(
            self, 
            "保存历史记录", 
            "请输入此次检测的描述:",
            QLineEdit.EchoMode.Normal,
            f"批量检测 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        if not ok or not description:
            return
            
        try:
            # 创建历史记录数据
            history_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': description,
                'file_count': len(self.batch_results),
                'aligned_count': self.aligned_count,
                'not_aligned_count': self.not_aligned_count,
                'error_count': self.error_count,
                'results': self.batch_results
            }
            
            # 生成文件名
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"history_{timestamp_str}.json"
            filepath = os.path.join(self.history_directory, filename)
            
            # 保存为JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理不可序列化的对象
                json.dump(history_data, f, ensure_ascii=False, indent=2, default=self._json_serializer)
                
            QMessageBox.information(
                self, 
                "保存成功", 
                f"历史记录已保存: {description}\n"
                f"文件数: {len(self.batch_results)}, "
                f"对齐: {self.aligned_count}, "
                f"不对齐: {self.not_aligned_count}"
            )
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存历史记录时出错: {str(e)}\n{traceback.format_exc()}")
    
    def _json_serializer(self, obj):
        """JSON序列化器，处理不可序列化的对象"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        # 对于其他不可序列化对象，转为字符串
        try:
            return str(obj)
        except:
            return "非序列化对象"
    
    def view_history_records(self):
        """查看历史记录"""
        # 加载历史记录文件列表
        history_files = self._load_history_files()
        
        if not history_files:
            QMessageBox.information(self, "提示", "没有找到历史记录")
            return
            
        # 创建历史记录对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("历史检测记录")
        dialog.resize(600, 400)
        
        # 设置对话框布局
        layout = QVBoxLayout(dialog)
        
        # 创建历史记录表格
        history_table = QTableWidget()
        history_table.setColumnCount(5)  # 时间, 描述, 文件数, 对齐数, 不对齐数
        history_table.setHorizontalHeaderLabels(["时间", "描述", "文件数", "对齐数", "不对齐数"])
        history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # 填充表格数据
        history_table.setRowCount(len(history_files))
        for i, history in enumerate(history_files):
            time_item = QTableWidgetItem(history['timestamp'])
            desc_item = QTableWidgetItem(history['description'])
            count_item = QTableWidgetItem(str(history['file_count']))
            aligned_item = QTableWidgetItem(str(history['aligned_count']))
            not_aligned_item = QTableWidgetItem(str(history['not_aligned_count']))
            
            history_table.setItem(i, 0, time_item)
            history_table.setItem(i, 1, desc_item)
            history_table.setItem(i, 2, count_item)
            history_table.setItem(i, 3, aligned_item)
            history_table.setItem(i, 4, not_aligned_item)
        
        layout.addWidget(history_table)
        
        # 添加按钮区域
        button_layout = QHBoxLayout()
        load_button = QPushButton("加载选中记录")
        delete_button = QPushButton("删除选中记录")
        close_button = QPushButton("关闭")
        
        button_layout.addWidget(load_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # 按钮事件处理
        load_button.clicked.connect(lambda: self._load_selected_history(dialog, history_table, history_files))
        delete_button.clicked.connect(lambda: self._delete_selected_history(dialog, history_table, history_files))
        close_button.clicked.connect(dialog.accept)
        
        # 双击加载历史记录
        history_table.cellDoubleClicked.connect(lambda row, col: self._load_selected_history(dialog, history_table, history_files))
        
        # 显示对话框
        dialog.exec()
    
    def _load_history_files(self):
        """加载历史记录文件"""
        history_files = []
        
        try:
            # 遍历历史记录目录中的所有.json文件
            for filename in os.listdir(self.history_directory):
                if filename.endswith('.json') and filename.startswith('history_'):
                    filepath = os.path.join(self.history_directory, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            history_data = json.load(f)
                            # 添加文件路径信息
                            history_data['filepath'] = filepath
                            history_files.append(history_data)
                        except:
                            # 忽略无法解析的文件
                            pass
            
            # 按时间戳排序，最新的排在前面
            history_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载历史记录时出错: {str(e)}")
            return []
            
        return history_files
    
    def _load_selected_history(self, dialog, table, history_files):
        """加载选中的历史记录"""
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(dialog, "警告", "请先选择一条历史记录")
            return
            
        row_index = selected_rows[0].row()
        if row_index < 0 or row_index >= len(history_files):
            return
            
        history = history_files[row_index]
        
        # 询问用户确认
        reply = QMessageBox.question(
            dialog,
            "确认加载",
            f"确定要加载选中的历史记录吗？\n"
            f"描述: {history['description']}\n"
            f"时间: {history['timestamp']}\n"
            f"这将替换当前的批量检测结果。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        try:
            # 加载历史记录数据
            self.batch_results = history['results']
            self.aligned_count = history.get('aligned_count', 0)
            self.not_aligned_count = history.get('not_aligned_count', 0)
            self.error_count = history.get('error_count', 0)
            
            # 更新表格和统计信息
            self.batch_results_table.setRowCount(0)  # 清空表格
            for i, result in enumerate(self.batch_results):
                ref_file = result.get('ref_file', '')
                align_file = result.get('align_file', '')
                self._add_result_to_table(self.batch_results_table, i, result, ref_file, align_file)
                
            # 更新统计标签
            self._update_stats_labels()
            
            # 启用相关按钮
            self.export_excel_button.setEnabled(True)
            self.save_history_button.setEnabled(True)
            
            # 移除自动切换到批量检测标签页的代码
            # self.tab_widget.setCurrentIndex(1)  # 不再自动切换标签页

            # 更新状态栏
            self.statusBar().showMessage(
                f"已加载历史记录: {history['description']} ({history['timestamp']}), "
                f"共 {len(self.batch_results)} 对文件"
            )
            
            # 关闭对话框
            dialog.accept()
            
        except Exception as e:
            QMessageBox.critical(dialog, "加载失败", f"加载历史记录时出错: {str(e)}\n{traceback.format_exc()}")
    
    def _delete_selected_history(self, dialog, table, history_files):
        """删除选中的历史记录"""
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(dialog, "警告", "请先选择一条历史记录")
            return
            
        row_index = selected_rows[0].row()
        if row_index < 0 or row_index >= len(history_files):
            return
            
        history = history_files[row_index]
        
        # 询问用户确认
        reply = QMessageBox.question(
            dialog,
            "确认删除",
            f"确定要删除选中的历史记录吗？\n"
            f"描述: {history['description']}\n"
            f"时间: {history['timestamp']}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        try:
            # 删除历史记录文件
            filepath = history.get('filepath')
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                
                # 从列表中移除
                history_files.pop(row_index)
                
                # 更新表格
                table.removeRow(row_index)
                
                QMessageBox.information(dialog, "成功", "历史记录已删除")
            else:
                QMessageBox.warning(dialog, "警告", "找不到历史记录文件")
                
        except Exception as e:
            QMessageBox.critical(dialog, "删除失败", f"删除历史记录时出错: {str(e)}")
    
    def _read_csv_with_multiple_encodings(self, csv_path):
        """尝试多种编码读取CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            pandas.DataFrame: 读取的CSV数据
        """
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"成功使用编码 {encoding} 读取CSV文件")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if "utf-8" not in str(e).lower():  # 如果不是编码错误，直接抛出
                    raise
        
        raise ValueError(f"无法使用常见编码读取CSV文件，请尝试其他编码或检查文件格式")
    
    def _convert_csv_to_feedback_format(self, df):
        """将CSV数据转换为反馈数据格式，以便使用相同的处理逻辑
        
        Args:
            df: 包含CSV数据的DataFrame
            
        Returns:
            list: 转换后的反馈数据列表
        """
        converted_data = []
        
        # 定义对齐和不对齐的可能值
        align_values = ['对齐', '已对齐', '是', '1', 'True', 'true', '正确', '一致', 'Y', 'y', 'Yes', 'yes']
        not_align_values = ['不对齐', '未对齐', '否', '0', 'False', 'false', '错误', '不一致', 'N', 'n', 'No', 'no']
        
        for _, row in df.iterrows():
            try:
                # 确定标签
                actual_status = str(row.get('actual_status', '')).strip()
                status_value = None
                
                if actual_status in align_values:
                    status_value = "对齐"
                elif actual_status in not_align_values:
                    status_value = "不对齐"
                else:
                    # 尝试更模糊的匹配
                    status_lower = actual_status.lower()
                    if '对齐' in status_lower or '一致' in status_lower or 'align' in status_lower or 'true' in status_lower or 'yes' in status_lower:
                        status_value = "对齐"
                    elif '不对齐' in status_lower or '不一致' in status_lower or 'not align' in status_lower or 'false' in status_lower or 'no' in status_lower:
                        status_value = "不对齐"
                
                # 跳过无法识别标记的数据
                if status_value is None:
                    continue
                
                # 创建基本反馈项
                feedback_item = {
                    'id': str(row.get('id', '')),
                    'actual_status': status_value,
                    'offset_seconds': float(row.get('offset_seconds', 0)),
                    'offset_ms': float(row.get('offset_ms', 0)) if 'offset_ms' in row else 0
                }
                
                # 如果CSV包含分析数据字段，构建analysis_data结构
                analysis_data = {}
                segment_info = {}
                
                # 检查CSV是否包含各种分析字段
                # 布尔型特征
                for bool_field, csv_field in [
                    ('progressive_misalignment', 'progressive_misalignment'),
                    ('low_confidence', 'low_confidence'),
                    ('is_highly_consistent', 'is_highly_consistent')
                ]:
                    if csv_field in row and not pd.isna(row[csv_field]):
                        try:
                            analysis_data[bool_field] = bool(int(float(row[csv_field])))
                        except (ValueError, TypeError):
                            pass
                
                # 数值型特征
                for float_field, csv_field in [
                    ('beat_consistency', 'beat_consistency'),
                    ('correlation_strength', 'correlation_strength'),
                    ('peak_consistency', 'peak_consistency'),
                    ('consistency', 'consistency')
                ]:
                    if csv_field in row and not pd.isna(row[csv_field]):
                        try:
                            analysis_data[float_field] = float(row[csv_field])
                        except (ValueError, TypeError):
                            pass
                
                # 分段信息
                for info_field, csv_field in [
                    ('offset_std_dev', 'segment_offset_std'),
                    ('max_offset_diff', 'segment_offset_max_diff'),
                    ('mean_offset', 'segment_offset_mean'),
                    ('trend_consistency', 'trend_consistency')
                ]:
                    if csv_field in row and not pd.isna(row[csv_field]):
                        try:
                            segment_info[info_field] = float(row[csv_field])
                        except (ValueError, TypeError):
                            pass
                
                if segment_info:
                    analysis_data['segment_info'] = segment_info
                
                # 峰值信息
                for count_field, csv_field in [
                    ('num_peaks', 'peak_count'),
                    ('num_peaks_used', 'used_peak_count')
                ]:
                    if csv_field in row and not pd.isna(row[csv_field]):
                        try:
                            analysis_data[count_field] = int(float(row[csv_field]))
                        except (ValueError, TypeError):
                            pass
                
                # 峰值方法
                if 'peak_method' in row and not pd.isna(row['peak_method']):
                    analysis_data['peak_method'] = str(row['peak_method'])
                elif 'peak_method_code' in row and not pd.isna(row['peak_method_code']):
                    try:
                        code = int(float(row['peak_method_code']))
                        method = ''
                        if code == 1:
                            method = 'single'
                        elif code == 2:
                            method = 'single_fallback'
                        elif code == 3:
                            method = 'best_of_few'
                        elif code == 4:
                            method = 'multi_peak_avg'
                        if method:
                            analysis_data['peak_method'] = method
                    except (ValueError, TypeError):
                        pass
                
                # 添加分析数据（如果有）
                if analysis_data:
                    feedback_item['analysis_data'] = analysis_data
                
                converted_data.append(feedback_item)
                
            except Exception as e:
                print(f"转换CSV数据行时出错: {str(e)}")
                continue
        
        print(f"从CSV转换了 {len(converted_data)} 条反馈数据")
        return converted_data
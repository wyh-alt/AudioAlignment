"""
波形可视化组件 - 用于显示和比较音频波形
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class MplCanvas(FigureCanvas):
    """Matplotlib画布类，用于绘制波形图"""
    
    def __init__(self, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class WaveformView(QWidget):
    """波形可视化组件，用于显示和比较音频波形"""
    
    # 自定义信号
    timeClicked = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # 数据
        self.waveform1_data = None
        self.waveform2_data = None
        self.beats1 = None
        self.beats2 = None
        self.offset_seconds = None
        self.threshold_seconds = 0.02
        self.is_aligned = False
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 创建Matplotlib画布
        self.canvas = MplCanvas(width=10, height=4, dpi=100)
        layout.addWidget(self.canvas)
        
        # 移除边距
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # 连接鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def on_canvas_click(self, event):
        """处理画布上的鼠标点击事件"""
        if event.xdata is not None:
            self.timeClicked.emit(event.xdata)
    
    def set_visualization_data(self, data):
        """
        设置可视化数据
        
        Args:
            data: 包含波形和节奏点数据的字典
        """
        self.waveform1_data = data.get("waveform1")
        self.waveform2_data = data.get("waveform2")
        self.beats1 = data.get("beats1", [])
        self.beats2 = data.get("beats2", [])
        self.offset_seconds = data.get("offset_seconds")
        self.threshold_seconds = data.get("threshold_seconds", 0.02)
        self.is_aligned = data.get("is_aligned", False)
        
        self.plot_waveforms()
    
    def plot_waveforms(self, data1=None, data2=None, offset=0):
        """
        绘制两个音频的波形
        
        Args:
            data1: 第一个音频数据
            data2: 第二个音频数据
            offset: 两个音频之间的时间偏移（秒）
        """
        # 清除画布
        self.canvas.figure.clear()
        
        # 如果没有数据，不绘制
        if not data1 and not data2:
            self.canvas.draw()
            return
            
        # 创建子图
        ax = self.canvas.figure.add_subplot(111)
        
        # 绘制第一个波形
        if data1:
            time1 = np.linspace(0, data1['duration'], len(data1['data']))
            ax.plot(time1, data1['data'], color='blue', alpha=0.7, label='音频1')
        
        # 绘制第二个波形（考虑偏移）
        if data2:
            time2 = np.linspace(0, data2['duration'], len(data2['data'])) + offset
            ax.plot(time2, data2['data'], color='red', alpha=0.7, label='音频2')
        
        # 添加图例和标签
        ax.legend(loc='upper right')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('振幅')
        ax.grid(True)
        
        # 设置坐标轴范围
        if data1 and data2:
            max_duration = max(data1['duration'], data2['duration'] + abs(offset))
            ax.set_xlim(0, max_duration)
        elif data1:
            ax.set_xlim(0, data1['duration'])
        elif data2:
            ax.set_xlim(0, data2['duration'] + abs(offset))
        
        # 添加偏移信息
        if data1 and data2:
            offset_text = f"偏移: {offset:.6f}秒"
            ax.text(0.02, 0.95, offset_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # 绘制
        self.canvas.draw()
    
    def clear(self):
        """清除波形图"""
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'axes'):
            self.canvas.axes.clear()
            self.canvas.draw()
            
        self.waveform1_data = None
        self.waveform2_data = None
        self.beats1 = None
        self.beats2 = None
        self.offset_seconds = None 
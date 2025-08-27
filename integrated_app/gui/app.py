import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from integrated_app.gui.main_window import MainWidget


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("音频对齐工具")
    # 设置应用图标
    icon_path = Path(__file__).resolve().parent.parent / 'assets' / 'icon.svg'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    w = MainWidget()
    w.setWindowTitle("音频对齐工具")
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()



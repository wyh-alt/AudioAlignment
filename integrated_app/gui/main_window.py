import os
import joblib
from typing import Optional, List

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QGroupBox, QProgressBar, QTextEdit, QCheckBox,
    QDoubleSpinBox
)

from ..orchestrator import run_pipeline
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView


class DropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.setText(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list, list, str, str)
    error_signal = pyqtSignal(str)
    record_signal = pyqtSignal(dict)

    def __init__(self, vocal_dir: str, inst_dir: str, output_dir: str,
                 naming_template: str, model_path: Optional[str], use_ml: bool,
                 threshold_ms: float):
        super().__init__()
        self.vocal_dir = vocal_dir
        self.inst_dir = inst_dir
        self.output_dir = output_dir
        self.naming_template = naming_template
        self.model_path = model_path
        self.use_ml = use_ml
        self.threshold_ms = threshold_ms

    def run(self):
        try:
            ml_model = None
            feature_names: Optional[List[str]] = None
            if self.use_ml and self.model_path and os.path.exists(self.model_path):
                self.log_signal.emit(f"加载模型: {self.model_path}")
                model_data = joblib.load(self.model_path)
                ml_model = model_data.get('model')
                feature_names = model_data.get('feature_names', [])
            else:
                if self.use_ml:
                    self.log_signal.emit("未找到模型文件，将不使用模型进行首轮检测")

            def log_cb(msg: str):
                self.log_signal.emit(msg)

            def progress_cb(value: int):
                self.progress_signal.emit(int(max(0, min(100, value))))

            def record_cb(_rec: dict):
                # 实时推送记录到UI线程以更新表格
                self.record_signal.emit(_rec)

            self.log_signal.emit("开始流程：首轮检测 → 自动对齐 → 二次检测 → 导出Excel")

            all_records, success_records, excel_all, excel_success = run_pipeline(
                vocal_dir=self.vocal_dir,
                inst_dir=self.inst_dir,
                output_dir=self.output_dir,
                naming_template=self.naming_template,
                ml_model=ml_model,
                ml_feature_names=feature_names,
                first_pass_use_ml=self.use_ml and ml_model is not None,
                detector_threshold_ms=self.threshold_ms,
                log_cb=log_cb,
                progress_cb=progress_cb,
                record_cb=record_cb,
            )

            self.progress_signal.emit(100)
            self.finished_signal.emit(all_records, success_records, excel_all, excel_success)
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.worker: Optional[PipelineWorker] = None

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 路径与设置
        path_group = QGroupBox("输入/输出")
        grid = QGridLayout(path_group)

        self.vocal_edit = DropLineEdit()
        self.inst_edit = DropLineEdit()
        self.output_edit = DropLineEdit()
        btn_vocal = QPushButton("浏览…")
        btn_inst = QPushButton("浏览…")
        btn_out = QPushButton("浏览…")

        grid.addWidget(QLabel("原唱目录"), 0, 0)
        grid.addWidget(self.vocal_edit, 0, 1)
        grid.addWidget(btn_vocal, 0, 2)

        grid.addWidget(QLabel("伴奏目录"), 1, 0)
        grid.addWidget(self.inst_edit, 1, 1)
        grid.addWidget(btn_inst, 1, 2)

        grid.addWidget(QLabel("成品目录"), 2, 0)
        grid.addWidget(self.output_edit, 2, 1)
        grid.addWidget(btn_out, 2, 2)

        # 命名模板
        self.tpl_edit = QLineEdit("{id}-原唱_已改")
        grid.addWidget(QLabel("命名模板"), 3, 0)
        grid.addWidget(self.tpl_edit, 3, 1, 1, 2)

        # 模型与阈值
        self.use_model_chk = QCheckBox("首轮使用模型")
        self.model_edit = DropLineEdit()
        self.model_edit.setPlaceholderText("选择 .joblib 模型文件，可拖拽")
        btn_model = QPushButton("选择模型…")

        grid.addWidget(self.use_model_chk, 4, 0)
        grid.addWidget(self.model_edit, 4, 1)
        grid.addWidget(btn_model, 4, 2)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 500.0)
        self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setValue(20.0)
        grid.addWidget(QLabel("检测阈值(ms)"), 5, 0)
        grid.addWidget(self.threshold_spin, 5, 1)

        main_layout.addWidget(path_group)

        # 控制区
        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始一键执行")
        self.start_btn.setDefault(True)
        ctrl_layout.addWidget(self.start_btn)
        main_layout.addLayout(ctrl_layout)

        # 进度与日志
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)  # 限制日志区域高度
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.log)

        # 结果预览表（去掉偏移列）
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["编号", "参考文件", "需对齐文件", "状态", "输出文件"])
        # 列宽策略：编号按内容，其余随窗口自适应
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(200)  # 确保表格有足够显示空间
        main_layout.addWidget(self.table)

        # 结果提示
        self.result_label = QLabel("")
        main_layout.addWidget(self.result_label)

        # 连接信号
        btn_vocal.clicked.connect(lambda: self._choose_dir(self.vocal_edit))
        btn_inst.clicked.connect(lambda: self._choose_dir(self.inst_edit))
        btn_out.clicked.connect(lambda: self._choose_dir(self.output_edit))
        btn_model.clicked.connect(self._choose_model)
        self.start_btn.clicked.connect(self._start_pipeline)

    def _choose_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if path:
            edit.setText(path)

    def _choose_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.joblib);;所有文件 (*.*)")
        if file_path:
            self.model_edit.setText(file_path)

    def _append_log(self, text: str):
        self.log.append(text)

    def _set_progress(self, value: int):
        self.progress.setValue(value)

    def _on_finished(self, all_records: list, success_records: list, excel_all: str, excel_success: str):
        self.start_btn.setEnabled(True)
        self._append_log(f"完成。成功对齐文件数: {len(success_records)} / 全量: {len(all_records)}")
        self.result_label.setText(f"全量结果: {excel_all} | 成功结果: {excel_success}")
        # 填充预览表
        self.table.setRowCount(0)
        for rec in all_records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            idv = str(rec.get('id', ''))
            ref = str(rec.get('ref_file', ''))
            aln = str(rec.get('align_file', ''))
            status = str(rec.get('status', ''))
            out = str(rec.get('output_file', ''))
            for col, val in enumerate([idv, ref, aln, status, out]):
                self.table.setItem(row, col, QTableWidgetItem(val))

    def _on_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self._append_log(f"错误: {msg}")

    def _start_pipeline(self):
        vocal = self.vocal_edit.text().strip()
        inst = self.inst_edit.text().strip()
        out = self.output_edit.text().strip()
        tpl = self.tpl_edit.text().strip() or "{id}-原唱_已改"
        use_ml = self.use_model_chk.isChecked()
        model_path = self.model_edit.text().strip() if use_ml else None
        threshold = float(self.threshold_spin.value())

        if not (vocal and os.path.isdir(vocal)):
            self._append_log("请设置有效的原唱目录")
            return
        if not (inst and os.path.isdir(inst)):
            self._append_log("请设置有效的伴奏目录")
            return
        if not out:
            self._append_log("请设置成品输出目录")
            return

        self.progress.setValue(0)
        self.log.clear()
        self.result_label.setText("")
        self.table.setRowCount(0)  # 清除上一次的处理结果
        self.start_btn.setEnabled(False)

        self.worker = PipelineWorker(vocal, inst, out, tpl, model_path, use_ml, threshold)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self._set_progress)
        self.worker.record_signal.connect(self._append_record_row)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _append_record_row(self, rec: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        idv = str(rec.get('id', ''))
        ref = str(rec.get('ref_file', ''))
        aln = str(rec.get('align_file', ''))
        status = str(rec.get('status', ''))
        out = str(rec.get('output_file', ''))
        for col, val in enumerate([idv, ref, aln, status, out]):
            self.table.setItem(row, col, QTableWidgetItem(val))



import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import pandas as pd

# 修正可导入路径：添加两个子项目根目录到sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DETECTOR_ROOT = PROJECT_ROOT / 'audio_algnment_checker2-master'
ALIGNER_ROOT = PROJECT_ROOT / '原唱伴奏对齐'
for _p in [str(DETECTOR_ROOT), str(ALIGNER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_alignment.core.audio_processor import AudioProcessor as DetectorAudioProcessor
from audio_alignment.core.alignment_detector import AlignmentDetector
from audio_processor import AudioProcessor as AlignerAudioProcessor

from .feature_builder import build_feature_vector_from_analysis


def extract_id_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    # 规则：优先用 '-' 前的部分，否则取连续数字
    if '-' in base:
        candidate = base.split('-')[0].strip()
        if candidate:
            return candidate
    match = re.search(r"(\d+)", base)
    return match.group(1) if match else os.path.splitext(base)[0]


def find_audio_files(directory: str) -> List[str]:
    exts = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    return files


def match_pairs(vocal_dir: str, inst_dir: str) -> List[Tuple[str, str, str]]:
    vocal_files = find_audio_files(vocal_dir)
    inst_files = find_audio_files(inst_dir)

    id_to_vocal: Dict[str, str] = {}
    id_to_inst: Dict[str, str] = {}

    for f in vocal_files:
        id_ = extract_id_from_filename(f)
        if id_ not in id_to_vocal:
            id_to_vocal[id_] = f

    for f in inst_files:
        id_ = extract_id_from_filename(f)
        if id_ not in id_to_inst:
            id_to_inst[id_] = f

    pairs: List[Tuple[str, str, str]] = []
    for id_, vfile in id_to_vocal.items():
        if id_ in id_to_inst:
            pairs.append((id_, vfile, id_to_inst[id_]))
    return pairs


def analyze_pair(detector: AlignmentDetector, ref_file: str, align_file: str) -> Dict:
    p1 = DetectorAudioProcessor()
    p2 = DetectorAudioProcessor()
    p1.load_file(ref_file)
    p2.load_file(align_file)
    data1 = p1.get_audio_data()
    data2 = p2.get_audio_data()
    # 使用改进分析逻辑
    result = detector.analyze(data1, data2)
    # 标准化字段名
    result.setdefault('offset_seconds', result.get('offset', 0.0))
    result.setdefault('offset_ms', result.get('offset_ms', result.get('offset_seconds', 0.0) * 1000.0))
    return result


def predict_with_model(analysis_result: Dict, model, feature_names: List[str]) -> int:
    vector = build_feature_vector_from_analysis(analysis_result, feature_names)
    # RandomForest on single sample
    y = model.predict([vector])
    try:
        return int(y[0])
    except Exception:
        return 0


def format_output_filename(template_str: str, file_id: str, vocal_filename: str) -> str:
    name_wo_ext, ext = os.path.splitext(os.path.basename(vocal_filename))
    try:
        return template_str.format(id=file_id, name=name_wo_ext) + ext
    except Exception:
        return f"{file_id}-原唱_已改{ext}"


def align_one(vocal_file: str, inst_file: str, output_path: str) -> Tuple[float, float, bool]:
    aligner = AlignerAudioProcessor()
    time_offset, confidence, aligned = aligner.align_audio(vocal_file, inst_file, output_path)
    return time_offset, confidence, aligned


def run_pipeline(
    vocal_dir: str,
    inst_dir: str,
    output_dir: str,
    naming_template: str = "{id}-原唱_已改",
    ml_model: Optional[object] = None,
    ml_feature_names: Optional[List[str]] = None,
    first_pass_use_ml: bool = True,
    detector_threshold_ms: float = 20.0,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    record_cb: Optional[Callable[[Dict], None]] = None,
) -> Tuple[List[Dict], List[Dict], str, str]:
    os.makedirs(output_dir, exist_ok=True)

    detector = AlignmentDetector(threshold_ms=detector_threshold_ms)

    pairs = match_pairs(vocal_dir, inst_dir)
    total = max(1, len(pairs))
    first_done = 0
    second_done = 0
    if log_cb:
        log_cb(f"匹配到 {len(pairs)} 对文件")
    if progress_cb:
        progress_cb(0)

    first_pass_records: List[Dict] = []
    all_records: List[Dict] = []
    not_aligned_pairs: List[Tuple[str, str, str]] = []

    for file_id, vocal_file, inst_file in pairs:
        analysis = analyze_pair(detector, inst_file, vocal_file)
        system_is_aligned = bool(analysis.get('is_aligned', False))

        predicted_is_aligned = system_is_aligned
        if first_pass_use_ml and ml_model and ml_feature_names:
            label = predict_with_model(analysis, ml_model, ml_feature_names)
            # 约定：1 表示对齐，0 表示不对齐
            predicted_is_aligned = (label == 1)

        base_item = {
            'id': file_id,
            'ref_file': os.path.basename(inst_file),
            'align_file': os.path.basename(vocal_file),
            'offset_seconds': float(analysis.get('offset_seconds', 0.0)),
            'offset_ms': float(analysis.get('offset_ms', 0.0)),
            'system_status': '对齐' if system_is_aligned else '不对齐',
            'ml_status': '对齐' if predicted_is_aligned else '不对齐' if first_pass_use_ml else '',
            'analysis_data': analysis,
        }
        first_pass_records.append(base_item)

        if not predicted_is_aligned:
            not_aligned_pairs.append((file_id, vocal_file, inst_file))
        else:
            # 无需对齐，直接计入全量结果
            record = dict(base_item)
            record['status'] = '无需对齐'
            record['output_file'] = ''
            all_records.append(record)
            if record_cb:
                record_cb(record)
        # 首轮进度（前50%）
        first_done += 1
        if progress_cb:
            progress_cb(int(first_done / total * 50))

    # 自动对齐并进行二次检测（不使用ML）
    final_success_records: List[Dict] = []
    for file_id, vocal_file, inst_file in not_aligned_pairs:
        out_name = format_output_filename(naming_template, file_id, vocal_file)
        out_path = os.path.join(output_dir, out_name)

        _, _, align_ok = align_one(vocal_file, inst_file, out_path)
        if not align_ok:
            # 对齐失败
            record = {
                'id': file_id,
                'ref_file': os.path.basename(inst_file),
                'align_file': os.path.basename(vocal_file),
                'status': '对齐失败',
                'output_file': '',
            }
            all_records.append(record)
            if record_cb:
                record_cb(record)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            continue

        # 二次检测
        second = analyze_pair(detector, inst_file, out_path)
        second_ok = bool(second.get('is_aligned', False))
        if second_ok:
            rec = {
                'id': file_id,
                'ref_file': os.path.basename(inst_file),
                'aligned_vocal': os.path.basename(out_path),
                'offset_seconds': float(second.get('offset_seconds', 0.0)),
                'offset_ms': float(second.get('offset_ms', 0.0)),
                'analysis_data': second,
            }
            final_success_records.append(rec)
            record = {
                'id': file_id,
                'ref_file': os.path.basename(inst_file),
                'align_file': os.path.basename(os.path.basename(vocal_file)),
                'status': '对齐成功',
                'output_file': os.path.basename(out_path),
                'offset_seconds': rec['offset_seconds'],
                'offset_ms': rec['offset_ms'],
            }
            all_records.append(record)
            if record_cb:
                record_cb(record)
        else:
            # 二次检测未通过
            record = {
                'id': file_id,
                'ref_file': os.path.basename(inst_file),
                'align_file': os.path.basename(vocal_file),
                'status': '二次检测未通过',
                'output_file': '',
                'offset_seconds': float(second.get('offset_seconds', 0.0)),
                'offset_ms': float(second.get('offset_ms', 0.0)),
            }
            all_records.append(record)
            if record_cb:
                record_cb(record)
            try:
                os.remove(out_path)
            except Exception:
                pass
        # 二轮进度（后50%）
        second_done += 1
        if progress_cb:
            total_second = max(1, len(not_aligned_pairs))
            progress_cb(50 + int(second_done / total_second * 50))

    # 生成Excel
    excel_success = os.path.join(output_dir, '对齐成功结果.xlsx')
    excel_all = os.path.join(output_dir, '对齐全量结果.xlsx')

    def _autofit_columns(wb, ws_name: str):
        try:
            ws = wb[ws_name]
            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        val = str(cell.value) if cell.value is not None else ''
                    except Exception:
                        val = ''
                    max_length = max(max_length, len(val))
                ws.column_dimensions[col_letter].width = min(max(10, max_length + 2), 60)
        except Exception:
            pass

    def _write_excel(filepath: str, records: List[Dict], columns: List[str], headers_cn: List[str]):
        df = pd.DataFrame(records, columns=columns)
        df.columns = headers_cn
        # 使用openpyxl并自动列宽
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            wb = writer.book
            _autofit_columns(wb, 'Sheet1')

    # 写入成功结果（中文表头）
    success_cols = ['id','ref_file','aligned_vocal','offset_seconds','offset_ms']
    success_headers = ['编号','参考文件','对齐后原唱','偏移(秒)','偏移(毫秒)']
    if final_success_records:
        _write_excel(excel_success, final_success_records, success_cols, success_headers)
    else:
        _write_excel(excel_success, [], success_cols, success_headers)

    # 写入全量结果（中文表头）
    all_cols = ['id','ref_file','align_file','status','output_file','offset_seconds','offset_ms']
    all_headers = ['编号','参考文件','需对齐文件','状态','输出文件','偏移(秒)','偏移(毫秒)']
    if all_records:
        _write_excel(excel_all, all_records, all_cols, all_headers)
    else:
        _write_excel(excel_all, [], all_cols, all_headers)

    return all_records, final_success_records, excel_all, excel_success



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


def build_result_text(record: Dict) -> str:
    """
    将偏移量等信息转成"处理结果"文本，用于写入 Excel。
    示例：+20.56ms / -105ms / +20.56ms 结尾延长80ms 等。
    """
    status = record.get('status', '')
    analysis = record.get('analysis_data') or {}

    # 优先使用记录里的 offset_ms，其次从分析结果里取
    offset_ms = record.get('offset_ms', analysis.get('offset_ms'))
    # 对于自动对齐成功的结果，我们还会记录实际"处理偏移量"
    applied_ms = record.get('applied_offset_ms')
    # 结尾补静音相关信息（用于"仅结尾延长"场景的判断）
    tail_extend_ms = record.get('tail_extend_ms', analysis.get('tail_extend_ms'))
    need_tail_extend = record.get('need_tail_extend', analysis.get('need_tail_extend'))

    # 特殊状态优先处理
    if status == '对齐失败':
        return '对齐失败'
    if status == '二次检测未通过':
        # 按要求：不显示残余偏移值，只给出结果文案
        return '二次检测未通过'
    if status == '无需对齐':
        # 无需对齐状态，不显示偏移量信息
        return '无需对齐'

    parts: List[str] = []

    # 基础偏移量描述（保留正负号区分方向）
    # 优先使用"实际处理偏移量"，否则使用检测得到的偏移量
    ms_value = None
    if applied_ms is not None:
        ms_value = float(applied_ms)
    elif offset_ms is not None:
        ms_value = float(offset_ms)

    # 特殊场景：仅结尾延长、未做任何偏移处理的情况
    # 这类情况在"处理结果/处理偏移量"中固定显示为 0ms
    has_tail_extend = bool(need_tail_extend) or (tail_extend_ms not in (None, 0, 0.0))
    if has_tail_extend:
        # applied_ms / offset_ms 都视为"没有实质偏移处理"时，显示 0ms
        applied_mag = abs(float(applied_ms)) if applied_ms is not None else 0.0
        offset_mag = abs(float(offset_ms)) if offset_ms is not None else 0.0
        # 对于"对齐成功"状态，如果只有结尾延长，优先显示 0ms
        if status == '对齐成功' and applied_mag < 1.0:
            return "0ms"
        # 对于其他状态，如果偏移量都很小，也显示 0ms
        if applied_mag < 1.0 and offset_mag < 1.0:
            return "0ms"

    if ms_value is not None:
        mag = abs(ms_value)
        if mag >= 1e-3:  # 忽略极小数值
            if mag < 100:
                parts.append(f"{ms_value:+.2f}ms")
            else:
                parts.append(f"{ms_value:+.0f}ms")

    if parts:
        return ' '.join(parts)

    # 回退：对于成功的对齐状态，返回"0ms"表示无偏移；其他状态返回状态文本
    if status == '对齐成功':
        return '0ms'
    return status or ''


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


def align_one(vocal_file: str, inst_file: str, output_path: str, align_duration: bool = True, skip_content_align: bool = False) -> Tuple[float, float, bool]:
    aligner = AlignerAudioProcessor()
    time_offset, confidence, aligned = aligner.align_audio(
        vocal_file, inst_file, output_path,
        align_duration=align_duration,
        skip_content_align=skip_content_align
    )
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
    align_duration: bool = False,
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
    # 存放需要进入自动对齐流程的文件对，同时携带首轮分析结果和"内容是否已对齐"标志
    # 元组格式: (file_id, vocal_file, inst_file, base_item, content_already_aligned)
    not_aligned_pairs: List[Tuple[str, str, str, Dict, bool]] = []

    for file_id, vocal_file, inst_file in pairs:
        analysis = analyze_pair(detector, inst_file, vocal_file)
        system_is_aligned = bool(analysis.get('is_aligned', False))
        has_duration_mismatch = bool(analysis.get('has_duration_mismatch', False))

        # 首轮"是否对齐"的判断（系统 + 可选模型）
        predicted_is_aligned = system_is_aligned
        if first_pass_use_ml and ml_model and ml_feature_names:
            label = predict_with_model(analysis, ml_model, ml_feature_names)
            # 约定：1 表示对齐，0 表示不对齐
            predicted_is_aligned = (label == 1)

        # 时长差（秒），用于判断是否需要"结尾补静音/裁剪"
        # duration1: 参考伴奏时长，duration2: 原唱时长
        duration1_sec = float(analysis.get('duration1', 0.0))
        duration2_sec = float(analysis.get('duration2', 0.0))
        # 保留有符号差值：>0 表示伴奏更长，需要"延长"原唱尾部；<0 表示原唱更长，需要"裁剪"原唱尾部
        raw_duration_diff_sec = duration1_sec - duration2_sec
        duration_diff_abs_sec = abs(raw_duration_diff_sec)
        # 使用与阈值相同的级别来判断是否需要尾部长度调整：> detector_threshold_ms 视为需要处理
        tail_threshold_sec = detector_threshold_ms / 1000.0
        # 仅当启用"总时长对齐"选项时，才考虑时长差异
        need_tail_extend = align_duration and (duration_diff_abs_sec > tail_threshold_sec)
        # 记录预计需要调整的毫秒数（带正负号，用于区分延长/裁剪）
        tail_extend_ms = raw_duration_diff_sec * 1000.0 if need_tail_extend else 0.0

        base_item = {
            'id': file_id,
            'ref_file': os.path.basename(inst_file),
            'align_file': os.path.basename(vocal_file),
            'offset_seconds': float(analysis.get('offset_seconds', 0.0)),
            'offset_ms': float(analysis.get('offset_ms', 0.0)),
            'system_status': '对齐' if system_is_aligned else '不对齐',
            'ml_status': '对齐' if predicted_is_aligned else '不对齐' if first_pass_use_ml else '',
            'need_tail_extend': need_tail_extend,
            'tail_extend_ms': tail_extend_ms,
            'analysis_data': analysis,
        }
        first_pass_records.append(base_item)

        # 只有在"偏移在阈值内 且 时长差小于阈值"时，才真正视作"无需对齐"
        # 一旦需要补尾（结尾静音延长），也要进入自动对齐流程
        can_skip_align = predicted_is_aligned and (not need_tail_extend)

        if not can_skip_align:
            # 将首轮分析结果一并带入，方便后续生成"处理结果"文本
            # 同时记录"内容是否已对齐"标志，用于决定是否跳过内容对齐（仅做时长调整）
            content_already_aligned = predicted_is_aligned
            not_aligned_pairs.append((file_id, vocal_file, inst_file, base_item, content_already_aligned))
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
    for file_id, vocal_file, inst_file, base_item, content_already_aligned in not_aligned_pairs:
        out_name = format_output_filename(naming_template, file_id, vocal_file)
        out_path = os.path.join(output_dir, out_name)

        # align_one 返回的是实际应用在原唱上的偏移量（秒）
        # 如果内容已对齐（仅需时长调整），则跳过内容对齐计算
        time_offset, _, align_ok = align_one(
            vocal_file, inst_file, out_path,
            align_duration=align_duration,
            skip_content_align=content_already_aligned
        )
        # 从首轮分析中取出"结尾补尾"的相关信息，后续无论成功/失败都写入记录
        first_need_tail = bool(base_item.get('need_tail_extend', False))
        first_tail_ms = float(base_item.get('tail_extend_ms', 0.0)) if base_item.get('tail_extend_ms') is not None else 0.0

        if not align_ok:
            # 对齐失败
            record = {
                'id': file_id,
                'ref_file': os.path.basename(inst_file),
                'align_file': os.path.basename(vocal_file),
                'status': '对齐失败',
                'output_file': '',
                'need_tail_extend': first_need_tail,
                'tail_extend_ms': first_tail_ms,
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

        # 记录实际处理的偏移量（用于结果展示）
        applied_offset_seconds = float(time_offset)
        applied_offset_ms = applied_offset_seconds * 1000.0

        # 二次检测（检测对齐后的残余偏移）
        second = analyze_pair(detector, inst_file, out_path)
        second_ok = bool(second.get('is_aligned', False))
        if second_ok:
            # 判断是否实际上"无需对齐"：处理偏移量和时长延长都在阈值内
            # 使用与检测阈值相同的标准（detector_threshold_ms）
            actual_no_change = (
                abs(applied_offset_ms) < detector_threshold_ms and
                abs(first_tail_ms) < detector_threshold_ms
            )
            
            if actual_no_change:
                # 实际无需对齐，删除输出文件，标记为"无需对齐"
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                record = {
                    'id': file_id,
                    'ref_file': os.path.basename(inst_file),
                    'align_file': os.path.basename(vocal_file),
                    'status': '无需对齐',
                    'output_file': '',
                    'offset_seconds': base_item.get('offset_seconds', 0.0),
                    'offset_ms': base_item.get('offset_ms', 0.0),
                    'need_tail_extend': False,
                    'tail_extend_ms': 0.0,
                    'analysis_data': base_item.get('analysis_data'),
                }
                all_records.append(record)
                if record_cb:
                    record_cb(record)
            else:
                # 真正的对齐成功
                rec = {
                    'id': file_id,
                    'ref_file': os.path.basename(inst_file),
                    'aligned_vocal': os.path.basename(out_path),
                    'offset_seconds': float(second.get('offset_seconds', 0.0)),
                    'offset_ms': float(second.get('offset_ms', 0.0)),
                     # 实际处理偏移量
                    'applied_offset_seconds': applied_offset_seconds,
                    'applied_offset_ms': applied_offset_ms,
                    # 记录首轮检测到的"是否需要补尾"以及补尾毫秒数，用于"处理结果"展示
                    'need_tail_extend': first_need_tail,
                    'tail_extend_ms': first_tail_ms,
                    'analysis_data': second,
                }
                final_success_records.append(rec)
                record = {
                    'id': file_id,
                    'ref_file': os.path.basename(inst_file),
                    'align_file': os.path.basename(os.path.basename(vocal_file)),
                    'status': '对齐成功',
                    'output_file': os.path.basename(out_path),
                    # 记录残余偏移和实际处理偏移，方便后续展示与排查
                    'offset_seconds': rec['offset_seconds'],
                    'offset_ms': rec['offset_ms'],
                    'applied_offset_seconds': applied_offset_seconds,
                    'applied_offset_ms': applied_offset_ms,
                    'need_tail_extend': first_need_tail,
                    'tail_extend_ms': first_tail_ms,
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
                'need_tail_extend': first_need_tail,
                'tail_extend_ms': first_tail_ms,
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

    # 生成Excel前，为每条记录生成"处理结果"文本
    for rec in final_success_records:
        rec['result'] = build_result_text(rec)
    for rec in all_records:
        rec['result'] = build_result_text(rec)

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
        # 将尾部延长毫秒数转换为形如"427ms"的文本
        # 对于"二次检测未通过"等状态，不显示结尾延长信息（置为空）
        if 'tail_extend_ms' in df.columns:
            tail_series = pd.to_numeric(df['tail_extend_ms'], errors='coerce')
            # 在全量结果表中可以通过 status 字段判断是否需要隐藏
            if 'status' in df.columns:
                hide_mask = df['status'] == '二次检测未通过'
                tail_series[hide_mask] = pd.NA
            # 四舍五入到整数毫秒
            tail_series = tail_series.round(0)
            # 数值行转换为 "+xxxms"/"-xxxms"，0 显示为 "0ms"，缺失行保持为空字符串
            def _fmt_tail(v: float) -> str:
                try:
                    iv = int(v)
                except Exception:
                    return ''
                if iv == 0:
                    return '0ms'
                return f"{iv:+d}ms"

            tail_series_str = tail_series.where(
                tail_series.isna(),
                tail_series.map(_fmt_tail)
            ).fillna('')
            df['tail_extend_ms'] = tail_series_str
        df.columns = headers_cn
        # 使用openpyxl并自动列宽
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            wb = writer.book
            _autofit_columns(wb, 'Sheet1')

    # 写入成功结果（中文表头，"处理偏移量"单独一列，"结尾延长"单独一列）
    success_cols = ['id', 'ref_file', 'aligned_vocal', 'result', 'tail_extend_ms']
    success_headers = ['编号', '参考文件', '对齐后原唱', '处理偏移量', '结尾延长']
    if final_success_records:
        _write_excel(excel_success, final_success_records, success_cols, success_headers)
    else:
        _write_excel(excel_success, [], success_cols, success_headers)

    # 写入全量结果（中文表头，"处理偏移量"单独一列，"结尾延长"单独一列）
    all_cols = ['id', 'ref_file', 'align_file', 'status', 'output_file', 'result', 'tail_extend_ms']
    all_headers = ['编号', '参考文件', '需对齐文件', '状态', '输出文件', '处理偏移量', '结尾延长']
    if all_records:
        _write_excel(excel_all, all_records, all_cols, all_headers)
    else:
        _write_excel(excel_all, [], all_cols, all_headers)

    return all_records, final_success_records, excel_all, excel_success

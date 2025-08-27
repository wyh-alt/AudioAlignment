import math


def build_feature_vector_from_analysis(analysis_result: dict, model_feature_names: list) -> list:


    feature_map = {}

    # 基础特征
    offset_seconds = float(analysis_result.get('offset', analysis_result.get('offset_seconds', 0.0)))
    offset_ms = float(analysis_result.get('offset_ms', offset_seconds * 1000.0))
    feature_map['offset_seconds'] = offset_seconds
    feature_map['offset_ms_abs'] = abs(offset_ms)

    # analysis_result 可能包含的详细分析字段
    feature_map['progressive_misalignment'] = 1 if analysis_result.get('progressive_misalignment', False) else 0
    feature_map['beat_consistency'] = float(analysis_result.get('beat_consistency', 0.0))
    feature_map['correlation_strength'] = float(analysis_result.get('correlation_strength', 0.0))
    feature_map['peak_consistency'] = float(analysis_result.get('peak_consistency', 0.0))
    feature_map['low_confidence'] = 1 if analysis_result.get('low_confidence', False) else 0
    feature_map['is_highly_consistent'] = 1 if analysis_result.get('is_highly_consistent', False) else 0

    # 峰值信息
    if 'num_peaks' in analysis_result and 'num_peaks_used' in analysis_result:
        num_peaks = int(analysis_result.get('num_peaks') or 0)
        num_peaks_used = int(analysis_result.get('num_peaks_used') or 0)
        feature_map['peak_count'] = num_peaks
        feature_map['used_peak_count'] = num_peaks_used
        feature_map['peak_ratio'] = (num_peaks_used / num_peaks) if num_peaks > 0 else 0.0

    # 复合特征
    if 'correlation_strength' in analysis_result:
        feature_map['offset_corr_product'] = abs(offset_ms) * float(analysis_result.get('correlation_strength') or 0.0)

    if 'consistency' in analysis_result and float(analysis_result.get('consistency') or 0.0) > 0:
        consistency = float(analysis_result.get('consistency'))
        feature_map['offset_consistency_ratio'] = abs(offset_ms) / (consistency * 1000.0)

    # 分段信息
    segment_info = analysis_result.get('segment_info') or {}
    if isinstance(segment_info, dict) and segment_info:
        std_dev = float(segment_info.get('offset_std_dev', 0.0))
        mean_offset = float(segment_info.get('mean_offset', 0.0))
        max_diff = float(segment_info.get('max_offset_diff', 0.0))
        trend_consistency = float(segment_info.get('trend_consistency', 0.0))

        feature_map['segment_offset_std'] = std_dev
        feature_map['segment_offset_max_diff'] = max_diff
        feature_map['segment_offset_mean'] = mean_offset
        feature_map['trend_consistency'] = trend_consistency

        if abs(mean_offset) > 1e-3:
            feature_map['std_mean_ratio'] = std_dev / abs(mean_offset)
        else:
            feature_map['std_mean_ratio'] = 0.0

        if std_dev > 1e-4:
            corr_strength = float(analysis_result.get('correlation_strength', 0.0))
            feature_map['corr_std_ratio'] = corr_strength / std_dev
        else:
            feature_map['corr_std_ratio'] = 0.0

    # 按模型需要的特征顺序组装，缺失用0填充
    vector = []
    for name in model_feature_names:
        value = feature_map.get(name, 0)
        # 统一布尔、数值为float
        try:
            value = float(value)
        except Exception:
            value = 0.0
        # 处理nan/inf
        if math.isnan(value) or math.isinf(value):
            value = 0.0
        vector.append(value)

    return vector



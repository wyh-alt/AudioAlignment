"""
对齐检测模块 - 负责检测两个音频文件的对齐情况
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from .audio_processor import AudioProcessor


class AlignmentDetector:
    """音频对齐检测类，用于检测两个音频是否对齐"""
    
    def __init__(self, threshold_ms: float = 20.0):
        """
        初始化对齐检测器
        
        Args:
            threshold_ms: 对齐阈值（毫秒），默认为20毫秒
        """
        # 将阈值从毫秒转换为秒
        self.threshold_sec = threshold_ms / 1000.0
        self.threshold_ms = threshold_ms
        print(f"AlignmentDetector初始化 - 阈值设置: {self.threshold_sec}秒 ({self.threshold_ms}毫秒)")
        self.audio1 = None
        self.audio2 = None
        self.correlation_result = None
        self.offset_sec = None
        self.is_aligned = False
        self.beat_offsets = None
        # 添加人声检测的默认参数
        self.enable_vocal_exclusion = True  # 默认启用人声排除
        self.vocal_energy_ratio_threshold = 8.0  # 能量比阈值
        self.vocal_frame_duration = 0.2  # 分析帧长度(秒)
        self.min_non_vocal_segment_length = 1.0  # 最小非人声段落长度(秒)
    
    def set_audio_processors(self, audio1: AudioProcessor, audio2: AudioProcessor):
        """
        设置要比较的两个音频处理器
        
        Args:
            audio1: 第一个音频处理器
            audio2: 第二个音频处理器
        """
        self.audio1 = audio1
        self.audio2 = audio2
    
    def detect_alignment(self) -> Dict:
        """
        检测两个音频文件是否对齐
        
        Returns:
            Dict: 包含对齐信息的字典
        """
        if self.audio1 is None or self.audio2 is None:
            raise Exception("请先设置音频处理器")
        
        if self.audio1.audio_data is None or self.audio2.audio_data is None:
            raise Exception("音频数据未加载")
        
        # 确保两个音频有相同的采样率
        if self.audio1.sample_rate != self.audio2.sample_rate:
            raise Exception("两个音频的采样率必须相同")
        
        # 计算互相关
        correlation = self._calculate_cross_correlation()
        self.correlation_result = correlation
        
        # 找到最大互相关的位置
        max_corr_idx = np.argmax(np.abs(correlation))
        
        # 计算偏移量（单位：秒）
        sample_rate = self.audio1.sample_rate
        offset_samples = max_corr_idx - len(self.audio1.audio_data)
        self.offset_sec = offset_samples / sample_rate
        
        # 根据阈值判断是否对齐
        self.is_aligned = abs(self.offset_sec) < self.threshold_sec
        
        # 检测节奏点偏移
        self.beat_offsets = self._analyze_beat_alignment()
        
        # 检查两个音频的时长是否一致
        duration1 = self.audio1.duration
        duration2 = self.audio2.duration
        duration_diff = abs(duration1 - duration2)
        has_duration_mismatch = duration_diff > 0.1  # 如果时长差异大于0.1秒，则认为不一致
        
        # 返回结果
        result = {
            "is_aligned": self.is_aligned,
            "offset_seconds": self.offset_sec,
            "offset_ms": self.offset_sec * 1000,
            "threshold_ms": self.threshold_ms,
            "beat_offsets": self.beat_offsets,
            "duration1": duration1,
            "duration2": duration2,
            "duration_diff": duration_diff,
            "has_duration_mismatch": has_duration_mismatch
        }
        
        return result
    
    def _calculate_cross_correlation(self) -> np.ndarray:
        """
        计算两个音频的互相关
        
        Returns:
            np.ndarray: 互相关结果
        """
        # 归一化音频数据
        audio1_norm = self.audio1.audio_data / np.sqrt(np.sum(self.audio1.audio_data**2))
        audio2_norm = self.audio2.audio_data / np.sqrt(np.sum(self.audio2.audio_data**2))
        
        # 计算互相关
        correlation = signal.correlate(audio2_norm, audio1_norm, mode='full')
        
        return correlation
    
    def _analyze_beat_alignment(self) -> List[Dict]:
        """
        分析节奏点的对齐情况
        
        Returns:
            List[Dict]: 节奏点偏移信息列表
        """
        # 检测两个音频的节奏点
        beats1 = self.audio1.detect_beats()
        beats2 = self.audio2.detect_beats()
        
        # 如果检测到的节奏点太少，则返回空列表
        if len(beats1) < 3 or len(beats2) < 3:
            return []
        
        # 计算节奏点的偏移
        beat_offsets = []
        min_beats = min(len(beats1), len(beats2))
        
        # 考虑之前计算的整体偏移
        adjusted_beats2 = beats2 + self.offset_sec
        
        # 分析每个节奏点的对齐情况
        for i in range(min_beats):
            offset = beats1[i] - adjusted_beats2[i]
            is_aligned = abs(offset) < self.threshold_sec
            
            beat_offsets.append({
                "beat_index": i,
                "time_audio1": beats1[i],
                "time_audio2": adjusted_beats2[i],
                "offset_seconds": offset,
                "offset_ms": offset * 1000,
                "is_aligned": is_aligned
            })
        
        return beat_offsets
    
    def get_visualization_data(self) -> Dict:
        """
        获取用于可视化的数据
        
        Returns:
            Dict: 包含可视化数据的字典
        """
        if self.audio1 is None or self.audio2 is None:
            raise Exception("请先设置音频处理器")
        
        waveform1 = self.audio1.get_waveform_data()
        waveform2 = self.audio2.get_waveform_data()
        
        # 考虑偏移，调整第二个波形的时间轴
        if self.offset_sec is not None:
            waveform2["time"] = waveform2["time"] + self.offset_sec
        
        return {
            "waveform1": waveform1,
            "waveform2": waveform2,
            "beats1": self.audio1.detect_beats() if self.audio1.audio_data is not None else [],
            "beats2": self.audio2.detect_beats() if self.audio2.audio_data is not None else [],
            "offset_seconds": self.offset_sec,
            "is_aligned": self.is_aligned,
            "threshold_seconds": self.threshold_sec
        }

    def _find_multiple_peaks(self, correlation, num_peaks=5, min_distance=1000):
        """
        在互相关结果中找出多个局部最大值
        
        Args:
            correlation: 互相关结果
            num_peaks: 要寻找的峰值数量
            min_distance: 峰值之间的最小距离（样本数）
            
        Returns:
            list: 峰值对应的索引列表
        """
        # 复制相关性数组，以便在检测过程中进行修改
        corr = np.abs(correlation).copy()
        peaks = []
        
        # 寻找多个峰值
        for _ in range(num_peaks):
            if np.max(corr) <= 0:
                break
                
            # 找出当前最大值
            peak_idx = np.argmax(corr)
            peak_value = corr[peak_idx]
            
            # 如果第一个峰值就很小，则可能不存在明显的相关性，直接返回
            if len(peaks) == 0 and peak_value < 0.1:
                break
                
            # 如果当前峰值相对于第一个峰值太小，则停止寻找
            if len(peaks) > 0 and peak_value < 0.3 * corr[peaks[0]]:
                break
                
            peaks.append(peak_idx)
            
            # 将当前峰值周围区域置零，避免重复检测
            start_idx = max(0, peak_idx - min_distance)
            end_idx = min(len(corr), peak_idx + min_distance)
            corr[start_idx:end_idx] = 0
            
        return peaks

    def analyze(self, data1, data2):
        """
        分析两个音频数据的对齐情况
        
        Args:
            data1: 第一个音频数据
            data2: 第二个音频数据
            
        Returns:
            Dict: 包含对齐信息的字典
        """
        # 计算互相关
        correlation = self._calculate_cross_correlation_data(data1, data2)
        
        # 采样率和信号长度
        sample_rate = data1['sample_rate']
        len_data1 = len(data1['data'])
        zero_offset_idx = len_data1 - 1
        
        # 检查两个音频的时长是否一致
        duration1 = data1.get('duration', len(data1['data']) / sample_rate)
        duration2 = data2.get('duration', len(data2['data']) / sample_rate)
        duration_diff = abs(duration1 - duration2)
        has_duration_mismatch = duration_diff > 0.1  # 如果时长差异大于0.1秒，则认为不一致
        
        # 计算最大合理偏移（默认为30秒，通常录音不会相差太多）
        max_reasonable_offset = 30.0  # 秒
        min_distance_samples = int(0.1 * sample_rate)  # 峰值之间至少间隔0.1秒
        
        # 找出多个峰值（例如取前5个局部最大值）
        peak_indices = self._find_multiple_peaks(
            correlation, 
            num_peaks=7,  # 寻找7个峰值
            min_distance=min_distance_samples
        )
        
        # 先执行分段分析以检测渐进性失调和偏移一致性
        segment_result = self.analyze_with_segments(data1, data2)
        progressive_misalignment = segment_result.get('progressive_misalignment', False)
        
        # 提取分段分析的关键指标
        segment_mean_offset = segment_result.get('mean_offset', 0)
        segment_std_dev = segment_result.get('offset_std_dev', 0)
        segment_max_diff = segment_result.get('max_offset_diff', 0)
        
        # 检查分段偏移是否非常一致且接近零 - 这表明可能是高度对齐的
        is_highly_consistent = segment_std_dev < 0.001 and abs(segment_mean_offset) < 0.001
        
        # 如果没有找到任何峰值，使用传统方法
        if not peak_indices:
            # 找到最大互相关的位置
            max_corr_idx = np.argmax(np.abs(correlation))
            max_corr_value = np.abs(correlation[max_corr_idx])
            
            # 计算偏移
            offset_samples = max_corr_idx - zero_offset_idx
            offset_sec = offset_samples / sample_rate
            
            # 限制合理范围
            if abs(offset_sec) > max_reasonable_offset:
                offset_sec = max_reasonable_offset * (1 if offset_sec > 0 else -1)
                
            # 降低单峰值方法的可信度要求
            low_confidence = max_corr_value < 0.2 or abs(offset_sec) >= max_reasonable_offset
            
            # 判断逻辑：单峰值偏移量小于阈值且没有渐进性失调才算对齐
            abs_offset = abs(offset_sec)
            debug_info = (
                f"\n阈值检查:"
                f"\n- 偏移量 (abs): {abs_offset:.8f}秒 ({abs_offset * 1000:.4f}毫秒)"
                f"\n- 阈值: {self.threshold_sec:.8f}秒 ({self.threshold_ms:.4f}毫秒)"
                f"\n- 比较结果: {abs_offset} < {self.threshold_sec} = {abs_offset < self.threshold_sec}"
                f"\n- 存在渐进性失调: {progressive_misalignment}"
                f"\n- 分段偏移均值: {segment_mean_offset:.8f}秒"
                f"\n- 分段偏移标准差: {segment_std_dev:.8f}秒"
                f"\n- 分段偏移最大差异: {segment_max_diff:.8f}秒"
                f"\n- 高度一致的分段: {is_highly_consistent}"
            )
            print(debug_info)
            
            # 对齐判断 - 使用更宽容的标准，同时考虑分段分析结果
            # 如果分段分析显示高度一致且偏移接近0，优先信任分段分析结果
            if is_highly_consistent:
                is_aligned = True
                print("特殊情况：分段分析显示高度一致且偏移接近0，判定为对齐")
            else:
                # 使用分段偏移均值作为主要判断依据
                segment_abs_offset = abs(segment_mean_offset)
                print(f"使用分段偏移均值判断: {segment_abs_offset}秒")
                
                # 获取分段偏移数据
                segment_offsets = segment_result.get('segment_offsets', [])
                last_segment_offset = abs(segment_offsets[-1]) if segment_offsets else 0
                is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                              all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
                
                # 1. 检查总体偏移是否在阈值范围内
                total_offset_aligned = abs_offset < self.threshold_sec
                
                # 2. 检查分段偏移的稳定性和趋势
                # 计算各段偏移的平均绝对值
                mean_abs_segment_offset = np.mean([abs(offset) for offset in segment_offsets]) if segment_offsets else 0
                
                # 判断是否存在显著的渐进性变化
                has_progressive_trend = False
                if is_monotonic and segment_result.get('trend_consistency', 0) >= 0.75:
                    first_last_diff = abs(segment_offsets[-1] - segment_offsets[0]) if segment_offsets else 0
                    if (last_segment_offset > self.threshold_sec and 
                        first_last_diff > self.threshold_sec * 0.5):
                        has_progressive_trend = True
                        print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
                
                # 3. 最终判断逻辑
                if has_progressive_trend:
                    # 如果存在显著的渐进性变化，判定为不对齐
                    is_aligned = False
                    print("由于存在显著的渐进性变化，判定为不对齐")
                elif mean_abs_segment_offset < self.threshold_sec and total_offset_aligned:
                    # 如果平均绝对偏移和总体偏移都在阈值范围内，判定为对齐
                    is_aligned = True
                    print(f"平均绝对偏移({mean_abs_segment_offset:.6f}秒)和总体偏移({abs_offset:.6f}秒)都在阈值范围内，判定为对齐")
                else:
                    # 其他情况，使用严格的阈值判断
                    is_aligned = segment_abs_offset < self.threshold_sec and not progressive_misalignment
                    print(f"使用严格阈值判断：分段偏移均值 {segment_abs_offset:.6f}秒 {'<' if segment_abs_offset < self.threshold_sec else '>='} {self.threshold_sec:.6f}秒")
            
            # 结果
            result = {
                "is_aligned": is_aligned,  # 根据偏移量和渐进性失调判断对齐状态
                "offset": offset_sec,
                "offset_ms": offset_sec * 1000,
                "threshold_ms": self.threshold_ms,
                "correlation_strength": float(max_corr_value),
                "low_confidence": low_confidence,
                "peak_method": "single",
                "progressive_misalignment": progressive_misalignment,
                "segment_info": segment_result,
                "debug_info": debug_info,
                "is_highly_consistent": is_highly_consistent,
                "duration1": duration1,
                "duration2": duration2,
                "duration_diff": duration_diff,
                "has_duration_mismatch": has_duration_mismatch
            }
            return result
        
        # 计算每个峰值对应的偏移和强度
        offsets = []
        for idx in peak_indices:
            offset_samples = idx - zero_offset_idx
            offset_sec = offset_samples / sample_rate
            corr_strength = float(np.abs(correlation[idx]))
            
            # 仅考虑合理范围内的偏移
            if abs(offset_sec) <= max_reasonable_offset:
                offsets.append((offset_sec, corr_strength))
        
        # 如果没有合理的偏移，使用传统方法
        if not offsets:
            # 同上方单峰情况
            max_corr_idx = np.argmax(np.abs(correlation))
            max_corr_value = np.abs(correlation[max_corr_idx])
            offset_samples = max_corr_idx - zero_offset_idx
            offset_sec = offset_samples / sample_rate
            
            # 限制合理范围
            if abs(offset_sec) > max_reasonable_offset:
                offset_sec = max_reasonable_offset * (1 if offset_sec > 0 else -1)
            
            # 降低单峰值方法的可信度要求
            low_confidence = max_corr_value < 0.2 or abs(offset_sec) >= max_reasonable_offset
            
            # 判断逻辑：主要依据偏移量和渐进性失调
            abs_offset = abs(offset_sec)
            debug_info = (
                f"\n阈值检查:"
                f"\n- 偏移量 (abs): {abs_offset:.8f}秒 ({abs_offset * 1000:.4f}毫秒)"
                f"\n- 阈值: {self.threshold_sec:.8f}秒 ({self.threshold_ms:.4f}毫秒)"
                f"\n- 比较结果: {abs_offset} < {self.threshold_sec} = {abs_offset < self.threshold_sec}"
                f"\n- 存在渐进性失调: {progressive_misalignment}"
                f"\n- 分段偏移均值: {segment_mean_offset:.8f}秒"
                f"\n- 分段偏移标准差: {segment_std_dev:.8f}秒"
                f"\n- 分段偏移最大差异: {segment_max_diff:.8f}秒"
                f"\n- 高度一致的分段: {is_highly_consistent}"
            )
            print(debug_info)
            
            # 对齐判断 - 使用更宽容的标准，同时考虑分段分析结果
            # 如果分段分析显示高度一致且偏移接近0，优先信任分段分析结果
            if is_highly_consistent:
                is_aligned = True
                print("特殊情况：分段分析显示高度一致且偏移接近0，判定为对齐")
            else:
                # 使用分段偏移均值作为主要判断依据
                segment_abs_offset = abs(segment_mean_offset)
                print(f"使用分段偏移均值判断: {segment_abs_offset}秒")
                
                # 获取分段偏移数据
                segment_offsets = segment_result.get('segment_offsets', [])
                last_segment_offset = abs(segment_offsets[-1]) if segment_offsets else 0
                is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                              all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
                
                # 1. 检查总体偏移是否在阈值范围内
                total_offset_aligned = abs_offset < self.threshold_sec
                
                # 2. 检查分段偏移的稳定性和趋势
                # 计算各段偏移的平均绝对值
                mean_abs_segment_offset = np.mean([abs(offset) for offset in segment_offsets]) if segment_offsets else 0
                
                # 判断是否存在显著的渐进性变化
                has_progressive_trend = False
                if is_monotonic and segment_result.get('trend_consistency', 0) >= 0.75:
                    first_last_diff = abs(segment_offsets[-1] - segment_offsets[0]) if segment_offsets else 0
                    if (last_segment_offset > self.threshold_sec and 
                        first_last_diff > self.threshold_sec * 0.5):
                        has_progressive_trend = True
                        print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
                
                # 3. 最终判断逻辑
                if has_progressive_trend:
                    # 如果存在显著的渐进性变化，判定为不对齐
                    is_aligned = False
                    print("由于存在显著的渐进性变化，判定为不对齐")
                elif mean_abs_segment_offset < self.threshold_sec and total_offset_aligned:
                    # 如果平均绝对偏移和总体偏移都在阈值范围内，判定为对齐
                    is_aligned = True
                    print(f"平均绝对偏移({mean_abs_segment_offset:.6f}秒)和总体偏移({abs_offset:.6f}秒)都在阈值范围内，判定为对齐")
                else:
                    # 其他情况，使用严格的阈值判断
                    is_aligned = segment_abs_offset < self.threshold_sec and not progressive_misalignment
                    print(f"使用严格阈值判断：分段偏移均值 {segment_abs_offset:.6f}秒 {'<' if segment_abs_offset < self.threshold_sec else '>='} {self.threshold_sec:.6f}秒")
            
            # 结果
            result = {
                "is_aligned": is_aligned,
                "offset": offset_sec,
                "offset_ms": offset_sec * 1000,
                "threshold_ms": self.threshold_ms,
                "correlation_strength": float(max_corr_value),
                "low_confidence": low_confidence,
                "peak_method": "single_fallback",
                "progressive_misalignment": progressive_misalignment,
                "segment_info": segment_result,
                "debug_info": debug_info,
                "is_highly_consistent": is_highly_consistent,
                "duration1": duration1,
                "duration2": duration2,
                "duration_diff": duration_diff,
                "has_duration_mismatch": has_duration_mismatch
            }
            return result
        
        # 如果找到的有效偏移少于3个，使用最强的那个
        if len(offsets) < 3:
            # 按相关性强度排序
            offsets.sort(key=lambda x: x[1], reverse=True)
            best_offset, best_strength = offsets[0]
            
            low_confidence = best_strength < 0.2
            
            # 判断逻辑：主要依据偏移量和渐进性失调
            abs_offset = abs(best_offset)
            debug_info = (
                f"\n阈值检查:"
                f"\n- 偏移量 (abs): {abs_offset:.8f}秒 ({abs_offset * 1000:.4f}毫秒)"
                f"\n- 阈值: {self.threshold_sec:.8f}秒 ({self.threshold_ms:.4f}毫秒)"
                f"\n- 比较结果: {abs_offset} < {self.threshold_sec} = {abs_offset < self.threshold_sec}"
                f"\n- 存在渐进性失调: {progressive_misalignment}"
                f"\n- 分段偏移均值: {segment_mean_offset:.8f}秒"
                f"\n- 分段偏移标准差: {segment_std_dev:.8f}秒"
                f"\n- 分段偏移最大差异: {segment_max_diff:.8f}秒"
                f"\n- 高度一致的分段: {is_highly_consistent}"
            )
            print(debug_info)
            
            # 对齐判断 - 使用更宽容的标准，同时考虑分段分析结果
            # 如果分段分析显示高度一致且偏移接近0，优先信任分段分析结果
            if is_highly_consistent:
                is_aligned = True
                print("特殊情况：分段分析显示高度一致且偏移接近0，判定为对齐")
            else:
                # 使用分段偏移均值作为主要判断依据
                segment_abs_offset = abs(segment_mean_offset)
                print(f"使用分段偏移均值判断: {segment_abs_offset}秒")
                
                # 获取分段偏移数据
                segment_offsets = segment_result.get('segment_offsets', [])
                last_segment_offset = abs(segment_offsets[-1]) if segment_offsets else 0
                is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                              all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
                
                # 1. 检查总体偏移是否在阈值范围内
                total_offset_aligned = abs_offset < self.threshold_sec
                
                # 2. 检查分段偏移的稳定性和趋势
                # 计算各段偏移的平均绝对值
                mean_abs_segment_offset = np.mean([abs(offset) for offset in segment_offsets]) if segment_offsets else 0
                
                # 判断是否存在显著的渐进性变化
                has_progressive_trend = False
                if is_monotonic and segment_result.get('trend_consistency', 0) >= 0.75:
                    first_last_diff = abs(segment_offsets[-1] - segment_offsets[0]) if segment_offsets else 0
                    if (last_segment_offset > self.threshold_sec and 
                        first_last_diff > self.threshold_sec * 0.5):
                        has_progressive_trend = True
                        print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
                
                # 3. 最终判断逻辑
                if has_progressive_trend:
                    # 如果存在显著的渐进性变化，判定为不对齐
                    is_aligned = False
                    print("由于存在显著的渐进性变化，判定为不对齐")
                elif mean_abs_segment_offset < self.threshold_sec and total_offset_aligned:
                    # 如果平均绝对偏移和总体偏移都在阈值范围内，判定为对齐
                    is_aligned = True
                    print(f"平均绝对偏移({mean_abs_segment_offset:.6f}秒)和总体偏移({abs_offset:.6f}秒)都在阈值范围内，判定为对齐")
                else:
                    # 其他情况，使用严格的阈值判断
                    is_aligned = segment_abs_offset < self.threshold_sec and not progressive_misalignment
                    print(f"使用严格阈值判断：分段偏移均值 {segment_abs_offset:.6f}秒 {'<' if segment_abs_offset < self.threshold_sec else '>='} {self.threshold_sec:.6f}秒")
            
            # 结果
            result = {
                "is_aligned": is_aligned,
                "offset": best_offset,
                "offset_ms": best_offset * 1000,
                "threshold_ms": self.threshold_ms,
                "correlation_strength": best_strength,
                "low_confidence": low_confidence,
                "peak_method": "best_of_few",
                "num_peaks": len(offsets),
                "progressive_misalignment": progressive_misalignment,
                "segment_info": segment_result,
                "debug_info": debug_info,
                "is_highly_consistent": is_highly_consistent,
                "duration1": duration1,
                "duration2": duration2,
                "duration_diff": duration_diff,
                "has_duration_mismatch": has_duration_mismatch
            }
            return result
        
        # 如果有足够多的峰值，进行多点处理
        # 按偏移值排序
        offsets.sort(key=lambda x: x[0])
        
        # 去掉最大和最小偏移
        filtered_offsets = offsets[1:-1]
        
        # 计算加权平均偏移
        total_weight = sum(strength for _, strength in filtered_offsets)
        if total_weight > 0:
            weighted_offset = sum(offset * strength for offset, strength in filtered_offsets) / total_weight
        else:
            # 防止除零错误
            weighted_offset = np.mean([offset for offset, _ in filtered_offsets])
        
        # 计算一致性指标（偏移的标准差）
        if len(filtered_offsets) > 1:
            std_dev = np.std([offset for offset, _ in filtered_offsets])
        else:
            std_dev = 0
        
        # 计算平均相关性强度
        avg_strength = total_weight / len(filtered_offsets) if filtered_offsets else 0
        
        # 可信度评估（降低严格程度，更接近原始算法）
        consistent = std_dev < 0.05  # 放宽至50毫秒内的一致性
        low_confidence = avg_strength < 0.1 or (not consistent and std_dev > 0.2)
        
        # 打印调试信息，显示实际阈值和偏移量
        print(f"DEBUG - 阈值: {self.threshold_sec}秒, 偏移量: {weighted_offset}秒")
        print(f"DEBUG - 阈值: {self.threshold_ms}毫秒, 偏移量: {weighted_offset * 1000}毫秒")
        
        # 对齐状态判断 - 确保单位一致并记录详细判断过程
        abs_offset = abs(weighted_offset)
        debug_info = (
            f"\n阈值检查:"
            f"\n- 偏移量 (abs): {abs_offset:.8f}秒 ({abs_offset * 1000:.4f}毫秒)"
            f"\n- 阈值: {self.threshold_sec:.8f}秒 ({self.threshold_ms:.4f}毫秒)"
            f"\n- 比较结果: {abs_offset} < {self.threshold_sec} = {abs_offset < self.threshold_sec}"
            f"\n- 存在渐进性失调: {progressive_misalignment}"
            f"\n- 分段偏移均值: {segment_mean_offset:.8f}秒"
            f"\n- 分段偏移标准差: {segment_std_dev:.8f}秒"
            f"\n- 分段偏移最大差异: {segment_max_diff:.8f}秒"
            f"\n- 高度一致的分段: {is_highly_consistent}"
        )
        print(debug_info)
        
        # 对齐判断 - 使用更宽容的标准，同时考虑分段分析结果
        # 如果分段分析显示高度一致且偏移接近0，优先信任分段分析结果
        if is_highly_consistent:
            is_aligned = True
            print("特殊情况：分段分析显示高度一致且偏移接近0，判定为对齐")
        else:
            # 使用分段偏移均值作为主要判断依据
            segment_abs_offset = abs(segment_mean_offset)
            print(f"使用分段偏移均值判断: {segment_abs_offset}秒")
            
            # 获取分段偏移数据
            segment_offsets = segment_result.get('segment_offsets', [])
            last_segment_offset = abs(segment_offsets[-1]) if segment_offsets else 0
            is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                          all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
            
            # 1. 检查总体偏移是否在阈值范围内
            total_offset_aligned = abs_offset < self.threshold_sec
            
            # 2. 检查分段偏移的稳定性和趋势
            # 计算各段偏移的平均绝对值
            mean_abs_segment_offset = np.mean([abs(offset) for offset in segment_offsets]) if segment_offsets else 0
            
            # 判断是否存在显著的渐进性变化
            has_progressive_trend = False
            if is_monotonic and segment_result.get('trend_consistency', 0) >= 0.75:
                first_last_diff = abs(segment_offsets[-1] - segment_offsets[0]) if segment_offsets else 0
                if (last_segment_offset > self.threshold_sec and 
                    first_last_diff > self.threshold_sec * 0.5):
                    has_progressive_trend = True
                    print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
            
            # 3. 最终判断逻辑
            if has_progressive_trend:
                # 如果存在显著的渐进性变化，判定为不对齐
                is_aligned = False
                print("由于存在显著的渐进性变化，判定为不对齐")
            elif mean_abs_segment_offset < self.threshold_sec and total_offset_aligned:
                # 如果平均绝对偏移和总体偏移都在阈值范围内，判定为对齐
                is_aligned = True
                print(f"平均绝对偏移({mean_abs_segment_offset:.6f}秒)和总体偏移({abs_offset:.6f}秒)都在阈值范围内，判定为对齐")
            else:
                # 其他情况，使用严格的阈值判断
                is_aligned = segment_abs_offset < self.threshold_sec and not progressive_misalignment
                print(f"使用严格阈值判断：分段偏移均值 {segment_abs_offset:.6f}秒 {'<' if segment_abs_offset < self.threshold_sec else '>='} {self.threshold_sec:.6f}秒")
        
        # 结果
        result = {
            "is_aligned": is_aligned,
            "offset": weighted_offset,
            "offset_ms": weighted_offset * 1000,
            "threshold_ms": self.threshold_ms,
            "correlation_strength": float(avg_strength),
            "consistency": float(std_dev),
            "low_confidence": low_confidence,
            "peak_method": "multi_peak_avg",
            "num_peaks": len(filtered_offsets) + 2,  # 加上被过滤掉的最大和最小值
            "num_peaks_used": len(filtered_offsets),
            "progressive_misalignment": progressive_misalignment,
            "segment_info": segment_result,
            "debug_info": debug_info,
            "is_highly_consistent": is_highly_consistent,
            "duration1": duration1,
            "duration2": duration2,
            "duration_diff": duration_diff,
            "has_duration_mismatch": has_duration_mismatch
        }
        
        # 添加ID信息用于识别问题文件
        file_id = None
        try:
            import re
            import os
            if 'ref_file' in globals() and ref_file:
                match = re.search(r'(\d+)', os.path.basename(ref_file))
                if match:
                    file_id = match.group(1)
        except:
            pass
            
        if file_id:
            print(f"ID {file_id} - {debug_info}")
            
        return result

    def _detect_vocal_only_segments(self, audio1_data, audio2_data, sample_rate):
        """
        检测两个音频文件中一个有声而另一个几乎无声的段落，这些通常是仅包含人声的部分
        
        Args:
            audio1_data: 第一个音频数据
            audio2_data: 第二个音频数据
            sample_rate: 采样率
            
        Returns:
            List[Tuple[float, float]]: 人声段落的时间范围列表 [(start1, end1), (start2, end2), ...]
        """
        print(f"开始检测人声段落...")
        
        # 计算帧大小
        frame_size = int(self.vocal_frame_duration * sample_rate)
        
        # 确保两个音频长度一致
        min_length = min(len(audio1_data), len(audio2_data))
        audio1_data = audio1_data[:min_length]
        audio2_data = audio2_data[:min_length]
        
        # 设置参数
        energy_ratio_threshold = self.vocal_energy_ratio_threshold
        
        # 分帧分析
        vocal_segments = []
        current_vocal_start = None
        
        # 使用较大的步长提高效率，但不影响精度
        step_size = frame_size // 2  # 50%重叠
        
        for i in range(0, min_length - frame_size, step_size):
            # 提取当前帧
            frame1 = audio1_data[i:i+frame_size]
            frame2 = audio2_data[i:i+frame_size]
            
            # 计算RMS能量
            energy1 = np.sqrt(np.mean(frame1**2))
            energy2 = np.sqrt(np.mean(frame2**2))
            
            # 避免除零
            energy1 = max(energy1, 1e-10)
            energy2 = max(energy2, 1e-10)
            
            # 计算能量比
            energy_ratio = max(energy1/energy2, energy2/energy1)
            
            # 判断是否一个有声一个无声
            # 同时考虑整体能量水平，避免两个都很低的情况
            min_energy_threshold = 0.01  # 最小有效能量阈值
            is_vocal_only = (energy_ratio > energy_ratio_threshold and 
                             max(energy1, energy2) > min_energy_threshold)
            
            # 调试输出
            if is_vocal_only and i % 20000 == 0:  # 不要打印太多日志
                print(f"  检测到可能的人声段落: 位置={i/sample_rate:.2f}s, 能量比={energy_ratio:.2f}, " +
                      f"能量1={energy1:.6f}, 能量2={energy2:.6f}")
            
            # 标记段落
            if is_vocal_only and current_vocal_start is None:
                current_vocal_start = i / sample_rate  # 转换为秒
            elif not is_vocal_only and current_vocal_start is not None:
                # 记录段落并重置起点
                segment_end = (i + frame_size) / sample_rate
                vocal_segments.append((current_vocal_start, segment_end))
                current_vocal_start = None
        
        # 处理结尾的情况
        if current_vocal_start is not None:
            vocal_segments.append((current_vocal_start, min_length / sample_rate))
        
        # 合并相邻或重叠的段落
        if vocal_segments:
            merged_segments = [vocal_segments[0]]
            for start, end in vocal_segments[1:]:
                prev_start, prev_end = merged_segments[-1]
                # 如果当前段落与上一个段落相邻或重叠（允许0.5秒的间隔）
                if start <= prev_end + 0.5:
                    # 合并段落
                    merged_segments[-1] = (prev_start, max(end, prev_end))
                else:
                    # 添加新段落
                    merged_segments.append((start, end))
            
            vocal_segments = merged_segments
        
        print(f"人声段落检测完成。检测到 {len(vocal_segments)} 个人声段落。")
        # 打印段落信息
        for i, (start, end) in enumerate(vocal_segments):
            duration = end - start
            print(f"  人声段落 #{i+1}: {start:.2f}s - {end:.2f}s (长度: {duration:.2f}s)")
        
        return vocal_segments

    def analyze_with_segments(self, data1, data2, num_segments=5):
        """
        分段分析音频对齐情况，检测渐进性失调
        
        Args:
            data1: 第一个音频数据
            data2: 第二个音频数据
            num_segments: 分段数量
            
        Returns:
            Dict: 包含分段分析结果的字典
        """
        # 将音频分成多段
        sample_rate = data1['sample_rate']
        audio1_data = data1['data']
        audio2_data = data2['data']
        
        # 确保两个数据长度一致以避免索引错误
        min_length = min(len(audio1_data), len(audio2_data))
        audio1_data = audio1_data[:min_length]
        audio2_data = audio2_data[:min_length]
        
        # 检测并排除人声段落
        if self.enable_vocal_exclusion:
            vocal_segments = self._detect_vocal_only_segments(audio1_data, audio2_data, sample_rate)
            
            # 创建有效段落的掩码 (True表示非人声)
            vocal_mask = np.ones(min_length, dtype=bool)
            for start_sec, end_sec in vocal_segments:
                start_idx = max(0, int(start_sec * sample_rate))
                end_idx = min(min_length, int(end_sec * sample_rate))
                vocal_mask[start_idx:end_idx] = False
            
            # 找出足够长的非人声连续段落
            non_vocal_segments = []
            current_start = None
            min_segment_samples = int(self.min_non_vocal_segment_length * sample_rate)
            
            for i in range(len(vocal_mask)):
                if vocal_mask[i] and current_start is None:
                    current_start = i
                elif (not vocal_mask[i] or i == len(vocal_mask)-1) and current_start is not None:
                    segment_length = i - current_start
                    if segment_length >= min_segment_samples:
                        non_vocal_segments.append((current_start, i))
                    current_start = None
            
            # 根据非人声段落长度选择最长的几个段落
            if non_vocal_segments:
                # 按长度排序
                non_vocal_segments.sort(key=lambda x: x[1]-x[0], reverse=True)
                # 选择最长的num_segments个段落，或者全部段落如果不足num_segments个
                selected_segments = non_vocal_segments[:min(num_segments, len(non_vocal_segments))]
                # 按原始顺序重新排序
                selected_segments.sort(key=lambda x: x[0])
                
                print(f"选择了 {len(selected_segments)} 个非人声段落进行分析:")
                for i, (start_idx, end_idx) in enumerate(selected_segments):
                    start_sec = start_idx / sample_rate
                    end_sec = end_idx / sample_rate
                    duration = end_sec - start_sec
                    print(f"  段落 #{i+1}: {start_sec:.2f}s - {end_sec:.2f}s (长度: {duration:.2f}s)")
                
                # 存储段落分析结果
                segment_offsets = []
                segment_strengths = []
                
                # 对每个选定的非人声段落进行分析
                for start_idx, end_idx in selected_segments:
                    segment1 = {'data': audio1_data[start_idx:end_idx], 'sample_rate': sample_rate}
                    segment2 = {'data': audio2_data[start_idx:end_idx], 'sample_rate': sample_rate}
                    
                    # 计算互相关
                    correlation = self._calculate_cross_correlation_data(segment1, segment2)
                    max_corr_idx = np.argmax(np.abs(correlation))
                    max_corr_value = np.abs(correlation[max_corr_idx])
                    
                    # 计算偏移量
                    zero_offset_idx = len(segment1['data']) - 1
                    offset_samples = max_corr_idx - zero_offset_idx
                    offset_sec = offset_samples / sample_rate
                    
                    # 存储结果
                    segment_offsets.append(offset_sec)
                    segment_strengths.append(float(max_corr_value))
                
                # 如果没有足够的有效段落，回退到原始算法
                if len(segment_offsets) < 2:
                    print("警告：有效非人声段落不足，回退到标准分段分析")
                    return self._analyze_with_standard_segments(data1, data2, num_segments)
                
                # 计算统计指标
                mean_offset = np.mean(segment_offsets)
                max_diff = max(segment_offsets) - min(segment_offsets)
                std_dev = np.std(segment_offsets)
                
                # 分析趋势
                # ...与原来的分析相同
                
                # 计算趋势一致性
                offset_diffs = []
                for i in range(1, len(segment_offsets)):
                    diff = segment_offsets[i] - segment_offsets[i-1]
                    offset_diffs.append(diff)
                
                # 计算趋势的一致性
                consistent_diffs = 0
                if len(offset_diffs) >= 3:
                    pos_diffs = sum(1 for diff in offset_diffs if diff > 0.003)
                    neg_diffs = sum(1 for diff in offset_diffs if diff < -0.003)
                    consistent_diffs = max(pos_diffs, neg_diffs)
                
                trend_consistency = consistent_diffs / len(offset_diffs) if offset_diffs else 0
                
                # 判断是否存在明显趋势
                has_consistent_trend = trend_consistency >= 0.75
                
                # 判断是否存在渐进性失调
                progressive_threshold = 0.080
                max_diff_threshold = 0.120
                
                # 检查是否单调
                is_monotonic = False
                if len(segment_offsets) >= 3:
                    is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                                  all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
                
                # 检查首尾差异
                last_segment_offset = abs(segment_offsets[-1])
                first_segment_offset = abs(segment_offsets[0])
                first_last_diff = abs(segment_offsets[-1] - segment_offsets[0])
                
                # 渐进性失调判断
                progressive_misalignment = False
                
                if is_monotonic and trend_consistency >= 0.85:
                    if (last_segment_offset > self.threshold_sec * 2 and 
                        first_last_diff > self.threshold_sec):
                        progressive_misalignment = True
                        print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
                elif not progressive_misalignment:
                    mean_abs_offset = np.mean([abs(offset) for offset in segment_offsets])
                    
                    if mean_abs_offset < self.threshold_sec * 1.2:
                        progressive_misalignment = False
                        print(f"平均绝对偏移在容忍范围内：{mean_abs_offset:.6f}秒")
                    elif std_dev > progressive_threshold and max_diff > max_diff_threshold:
                        progressive_misalignment = True
                        print(f"检测到严重的偏移不稳定：标准差 {std_dev:.6f}秒，最大差异 {max_diff:.6f}秒")
                
                # 打印调试信息
                print(f"排除人声后的分段分析结果:")
                print(f"- 各段偏移: {[f'{offset:.6f}s' for offset in segment_offsets]}")
                print(f"- 偏移均值: {mean_offset:.6f}秒")
                print(f"- 偏移标准差: {std_dev:.6f}秒")
                print(f"- 最大偏移差异: {max_diff:.6f}秒")
                print(f"- 趋势一致性: {trend_consistency:.2f}")
                print(f"- 存在单调趋势: {is_monotonic}")
                print(f"- 判断存在渐进性失调: {progressive_misalignment}")
                
                # 返回结果
                return {
                    "segment_offsets": segment_offsets,
                    "segment_strengths": segment_strengths,
                    "mean_offset": float(mean_offset),
                    "offset_std_dev": float(std_dev),
                    "max_offset_diff": float(max_diff),
                    "progressive_misalignment": progressive_misalignment,
                    "has_consistent_trend": has_consistent_trend,
                    "trend_consistency": float(trend_consistency),
                    "std_dev_check": std_dev > progressive_threshold,
                    "max_diff_check": max_diff > max_diff_threshold,
                    "trend_check": has_consistent_trend and trend_consistency >= 0.85,
                    "small_offset_no_misalignment": abs(mean_offset) < (self.threshold_sec * 1.2) and not progressive_misalignment,
                    "vocal_excluded": True
                }
            else:
                print("没有找到足够长的非人声段落，回退到标准分段分析")
        
        # 如果没有启用人声排除或没有找到足够的非人声段落，使用标准分段分析
        return self._analyze_with_standard_segments(data1, data2, num_segments)
        
    def _analyze_with_standard_segments(self, data1, data2, num_segments=5):
        """
        使用均匀分段的标准分析方法（原有方法）
        
        Args:
            data1: 第一个音频数据
            data2: 第二个音频数据
            num_segments: 分段数量
            
        Returns:
            Dict: 包含分段分析结果的字典
        """
        # 将音频分成多段
        sample_rate = data1['sample_rate']
        audio1_data = data1['data']
        audio2_data = data2['data']
        
        # 确保两个数据长度一致以避免索引错误
        min_length = min(len(audio1_data), len(audio2_data))
        audio1_data = audio1_data[:min_length]
        audio2_data = audio2_data[:min_length]
        
        # 计算每段的长度
        segment_length = min_length // num_segments
        
        # 存储每段的偏移结果
        segment_offsets = []
        segment_strengths = []
        
        # 对每段进行分析
        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, min_length)
            
            # 提取段落数据
            segment1 = {'data': audio1_data[start:end], 'sample_rate': sample_rate}
            segment2 = {'data': audio2_data[start:end], 'sample_rate': sample_rate}
            
            # 计算互相关
            correlation = self._calculate_cross_correlation_data(segment1, segment2)
            max_corr_idx = np.argmax(np.abs(correlation))
            max_corr_value = np.abs(correlation[max_corr_idx])
            
            # 计算偏移量
            zero_offset_idx = len(segment1['data']) - 1
            offset_samples = max_corr_idx - zero_offset_idx
            offset_sec = offset_samples / sample_rate
            
            # 存储结果
            segment_offsets.append(offset_sec)
            segment_strengths.append(float(max_corr_value))
        
        # 计算偏移量统计信息
        mean_offset = np.mean(segment_offsets)
        max_diff = max(segment_offsets) - min(segment_offsets)
        std_dev = np.std(segment_offsets)
        
        # 获取段落偏移之间的趋势 - 检查是否呈现单向递增或递减趋势
        # 计算相邻段落偏移的差值
        offset_diffs = []
        consistent_diffs = 0
        
        for i in range(1, len(segment_offsets)):
            diff = segment_offsets[i] - segment_offsets[i-1]
            offset_diffs.append(diff)
            
        # 计算趋势的一致性 - 检查差值是否都是同一个符号
        if len(offset_diffs) >= 3:
            # 计算有多少差值的符号是一致的
            pos_diffs = sum(1 for diff in offset_diffs if diff > 0.003)  # 正向差异，添加小阈值
            neg_diffs = sum(1 for diff in offset_diffs if diff < -0.003)  # 负向差异，添加小阈值
            consistent_diffs = max(pos_diffs, neg_diffs)
            
        # 趋势的一致性比例
        trend_consistency = consistent_diffs / len(offset_diffs) if offset_diffs else 0
                
        # 判断是否存在明显趋势
        has_consistent_trend = trend_consistency >= 0.75  # 至少75%的变化朝同一个方向
        
        # 判断是否存在渐进性失调
        progressive_threshold = 0.080  # 提高到80毫秒
        max_diff_threshold = 0.120    # 提高到120毫秒
        
        # 检查是否存在明显的递增或递减趋势
        is_monotonic = all(segment_offsets[i] <= segment_offsets[i+1] for i in range(len(segment_offsets)-1)) or \
                      all(segment_offsets[i] >= segment_offsets[i+1] for i in range(len(segment_offsets)-1))
        
        # 检查末端偏移
        last_segment_offset = abs(segment_offsets[-1])
        first_segment_offset = abs(segment_offsets[0])
        
        # 计算首尾段的差异
        first_last_diff = abs(segment_offsets[-1] - segment_offsets[0])
        
        # 简化的渐进性失调判断
        progressive_misalignment = False
        
        # 1. 检查是否存在明显的渐进性变化
        if is_monotonic and trend_consistency >= 0.85:  # 提高趋势一致性要求
            # 如果最后一段超过阈值的2倍，且变化趋势明显
            if (last_segment_offset > self.threshold_sec * 2 and 
                first_last_diff > self.threshold_sec):
                progressive_misalignment = True
                print(f"检测到显著的渐进性变化：末端偏移 {last_segment_offset:.6f}秒，首尾差异 {first_last_diff:.6f}秒")
        
        # 2. 检查偏移的稳定性
        # 只有在没有明显渐进性变化的情况下才考虑稳定性
        elif not progressive_misalignment:
            # 计算各段偏移的平均绝对值
            mean_abs_offset = np.mean([abs(offset) for offset in segment_offsets])
            
            # 如果平均绝对偏移在阈值范围内，即使标准差大也不应判定为失调
            if mean_abs_offset < self.threshold_sec * 1.2:  # 增加容忍度
                progressive_misalignment = False
                print(f"平均绝对偏移在容忍范围内：{mean_abs_offset:.6f}秒")
            # 否则，如果存在严重的不稳定性，判定为失调
            elif std_dev > progressive_threshold and max_diff > max_diff_threshold:
                progressive_misalignment = True
                print(f"检测到严重的偏移不稳定：标准差 {std_dev:.6f}秒，最大差异 {max_diff:.6f}秒")
        
        # 打印调试信息
        print(f"标准分段分析结果:")
        print(f"- 各段偏移: {[f'{offset:.6f}s' for offset in segment_offsets]}")
        print(f"- 偏移均值: {mean_offset:.6f}秒")
        print(f"- 偏移标准差: {std_dev:.6f}秒")
        print(f"- 最大偏移差异: {max_diff:.6f}秒")
        print(f"- 趋势一致性: {trend_consistency:.2f}")
        print(f"- 存在单调趋势: {is_monotonic}")
        print(f"- 判断存在渐进性失调: {progressive_misalignment}")
        
        # 返回结果
        return {
            "segment_offsets": segment_offsets,
            "segment_strengths": segment_strengths,
            "mean_offset": float(mean_offset),
            "offset_std_dev": float(std_dev),
            "max_offset_diff": float(max_diff),
            "progressive_misalignment": progressive_misalignment,
            "has_consistent_trend": has_consistent_trend,
            "trend_consistency": float(trend_consistency),
            "std_dev_check": std_dev > progressive_threshold,
            "max_diff_check": max_diff > max_diff_threshold,
            "trend_check": has_consistent_trend and trend_consistency >= 0.85,  # 提高趋势一致性要求
            "small_offset_no_misalignment": abs(mean_offset) < (self.threshold_sec * 1.2) and not progressive_misalignment,  # 增加容忍度
            "vocal_excluded": False
        }

    def _calculate_cross_correlation_data(self, data1, data2):
        """
        计算两个音频数据的互相关
        
        Args:
            data1: 第一个音频数据
            data2: 第二个音频数据
            
        Returns:
            np.ndarray: 互相关结果
        """
        # 检查采样率是否相同
        if data1['sample_rate'] != data2['sample_rate']:
            raise Exception("两个音频的采样率必须相同")
            
        audio1_data = data1['data']
        audio2_data = data2['data']
        
        # 归一化音频数据
        audio1_norm = audio1_data / np.sqrt(np.sum(audio1_data**2)) if np.sum(audio1_data**2) > 0 else audio1_data
        audio2_norm = audio2_data / np.sqrt(np.sum(audio2_data**2)) if np.sum(audio2_data**2) > 0 else audio2_data
        
        # 计算互相关
        correlation = signal.correlate(audio2_norm, audio1_norm, mode='full')
        
        return correlation 
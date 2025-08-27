import argparse
import os
import joblib

from .orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="原唱伴奏一键对齐与检测编排")
    parser.add_argument('--vocal_dir', required=True, help='原唱目录')
    parser.add_argument('--inst_dir', required=True, help='伴奏目录')
    parser.add_argument('--output_dir', required=True, help='成品输出目录')
    parser.add_argument('--name_tpl', default='{id}-原唱_已改', help='输出命名模板，可用{id},{name}')
    parser.add_argument('--model', default='', help='首轮检测可选模型.joblib（火眼一号）')
    parser.add_argument('--no-ml', action='store_true', help='首轮检测不使用模型')
    parser.add_argument('--threshold_ms', type=float, default=20.0, help='检测阈值（毫秒）')

    args = parser.parse_args()

    ml_model = None
    feature_names = None
    use_ml = not args.no_ml
    if use_ml and args.model and os.path.exists(args.model):
        model_data = joblib.load(args.model)
        ml_model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
    else:
        use_ml = False

    all_records, success_records, excel_all, excel_success = run_pipeline(
        vocal_dir=args.vocal_dir,
        inst_dir=args.inst_dir,
        output_dir=args.output_dir,
        naming_template=args.name_tpl,
        ml_model=ml_model,
        ml_feature_names=feature_names,
        first_pass_use_ml=use_ml,
        detector_threshold_ms=args.threshold_ms,
    )

    print(f"完成。成功对齐: {len(success_records)} / 全量: {len(all_records)}")
    print(f"全量结果: {excel_all}")
    print(f"成功结果: {excel_success}")


if __name__ == '__main__':
    main()



import argparse
from pathlib import Path
from lightworld.application.full_run_service import FullRunService
from lightworld.simulation.cluster_cli import (
    CLUSTER_METHOD_LLM_KEYWORD,
    CLUSTER_METHOD_THRESHOLD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightWorld 一键全流程运行器")
    parser.add_argument("--config", required=True, help="全流程配置文件路径（JSON）")
    parser.add_argument(
        "--cluster-method",
        choices=[CLUSTER_METHOD_THRESHOLD, CLUSTER_METHOD_LLM_KEYWORD],
        default=None,
        help="覆盖本次运行的 cluster 方法；未指定时，交互终端会提示选择。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = FullRunService()
    service.run(Path(args.config), cluster_method_override=args.cluster_method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

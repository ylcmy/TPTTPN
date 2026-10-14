import os
import logging


def setup_logger(args, exp_dir):
    """设置日志"""
    # 创建日志文件名
    log_file = os.path.join(exp_dir, "run.log")

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(args.model)

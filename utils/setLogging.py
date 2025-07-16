import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Union


def set_logger(
    log_path: str,
    name: Optional[str] = None,
    level: Union[str, int] = logging.INFO,
    console_level: Union[str, int] = logging.DEBUG,
    file_level: Union[str, int] = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置一个日志记录器，支持文件和控制台输出

    Args:
        log_path (str): 日志文件路径
        name (str, optional): 日志记录器名称，默认为调用模块名
        level (Union[str, int]): 日志记录器的整体日志级别
        console_level (Union[str, int]): 控制台输出的日志级别
        file_level (Union[str, int]): 文件输出的日志级别
        max_bytes (int): 日志文件的最大字节数，超过后会轮转
        backup_count (int): 保留的备份文件数量
        format_string (str, optional): 自定义日志格式字符串

    Returns:
        logging.Logger: 配置好的日志记录器

    Raises:
        OSError: 当无法创建日志文件或目录时
    """
    # 使用调用者的模块名称作为默认logger名称
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    logger = logging.getLogger(name)

    # 避免重复添加handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 设置日志格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建轮转文件handler，防止日志文件过大
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

    except (OSError, IOError) as e:
        print(f"警告: 无法创建日志文件 {log_path}: {e}")
        print("日志将只输出到控制台")

    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

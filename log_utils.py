# log_utils.py
"""
日志工具模块 - 将终端输出同时保存到文本文件

使用方法:
    from log_utils import setup_logger, close_logger
    
    # 在训练开始时调用
    setup_logger("train_timm")  # 会生成 logs/train_timm_20231225_143052.log
    
    # 正常使用 print() 即可，输出会同时写入文件
    print("开始训练...")
    
    # 训练结束时调用（可选，程序退出时会自动关闭）
    close_logger()
"""

import os
import sys
from datetime import datetime
from cxr_config import LOGS_DIR


class TeeLogger:
    """
    Tee Logger: 同时输出到控制台和文件
    
    将 stdout 重定向，使得所有 print() 的内容既显示在终端，
    又保存到指定的日志文件中。
    """
    
    def __init__(self, log_file: str, console_stream=None):
        """
        Args:
            log_file: 日志文件路径
            console_stream: 原始控制台流（默认为 sys.stdout）
        """
        self.console = console_stream if console_stream else sys.__stdout__
        self.log_file = open(log_file, "w", encoding="utf-8", buffering=1)  # 行缓冲
        self._closed = False
    
    def write(self, message):
        """写入消息到控制台和文件"""
        if self._closed:
            return
        # 写入控制台
        self.console.write(message)
        self.console.flush()
        # 写入文件
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """刷新缓冲区"""
        if self._closed:
            return
        self.console.flush()
        self.log_file.flush()
    
    def close(self):
        """关闭日志文件"""
        if not self._closed:
            self._closed = True
            self.log_file.close()
    
    def __del__(self):
        self.close()


# 全局变量，用于存储当前的 logger 实例
_current_logger = None
_original_stdout = None
_original_stderr = None


def setup_logger(script_name: str, log_dir: str = None) -> str:
    """
    设置日志记录器，将所有 print 输出同时保存到文件
    
    Args:
        script_name: 脚本名称（用于日志文件命名）
        log_dir: 日志目录（默认使用 LOGS_DIR）
    
    Returns:
        日志文件路径
    
    Example:
        log_path = setup_logger("train_timm")
        print("这条消息会同时显示在终端和保存到日志文件")
    """
    global _current_logger, _original_stdout, _original_stderr
    
    # 如果已有 logger，先关闭
    if _current_logger is not None:
        close_logger()
    
    # 确定日志目录
    if log_dir is None:
        log_dir = LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 保存原始 stdout 和 stderr
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    # 创建 TeeLogger 并重定向 stdout
    _current_logger = TeeLogger(log_path, _original_stdout)
    sys.stdout = _current_logger
    
    # 打印日志头部信息
    print("=" * 70)
    print(f"日志文件: {log_path}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    return log_path


def close_logger():
    """
    关闭日志记录器，恢复标准输出
    """
    global _current_logger, _original_stdout, _original_stderr
    
    if _current_logger is not None:
        # 打印日志尾部信息
        print()
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 恢复原始 stdout
        sys.stdout = _original_stdout
        
        # 关闭 logger
        _current_logger.close()
        _current_logger = None


def get_log_path() -> str:
    """
    获取当前日志文件路径
    
    Returns:
        当前日志文件路径，如果没有设置 logger 则返回 None
    """
    global _current_logger
    if _current_logger is not None:
        return _current_logger.log_file.name
    return None


class LoggerContext:
    """
    日志上下文管理器，支持 with 语句
    
    Example:
        with LoggerContext("train_timm"):
            print("训练过程...")
    """
    
    def __init__(self, script_name: str, log_dir: str = None):
        self.script_name = script_name
        self.log_dir = log_dir
        self.log_path = None
    
    def __enter__(self):
        self.log_path = setup_logger(self.script_name, self.log_dir)
        return self.log_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        close_logger()
        return False  # 不吞没异常


# 注册退出时自动关闭 logger
import atexit
atexit.register(close_logger)


import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_dir(path: str) -> None:
    """
    确保目录存在
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    return Path(__file__).parent.parent

class Config:
    """配置类"""
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = get_project_root() / "config" / "config.yaml"
        
        self.config = load_config(str(config_path))
        self._setup_paths()
    
    def _setup_paths(self):
        """设置路径"""
        root = get_project_root()
        self.data_dir = root / self.config['data']['raw_dir']
        self.processed_dir = root / self.config['data']['processed_dir']
        self.mask_dir = root / self.config['data']['mask_dir']
        self.results_dir = root / self.config['output']['save_dir']
        
        ensure_dir(self.processed_dir)
        ensure_dir(self.mask_dir)
        ensure_dir(self.results_dir)
    
    def __getitem__(self, key):
        return self.config[key]

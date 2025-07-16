"""
数据处理相关的工具函数
"""
import torch
import torch.nn.utils.rnn as rnn_utils
import pickle
from utils.dataset import CPMDataset


def collate_fn(batch):
    """
    数据整理函数，用于DataLoader
    
    Args:
        batch: 批次数据
        
    Returns:
        input_ids: 输入序列
        labels: 标签序列
    """
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(logger, args):
    """
    加载训练集
    
    Args:
        logger: 日志记录器
        args: 训练参数
        
    Returns:
        train_dataset: 训练数据集
    """
    logger.info("loading training dataset")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        train_list = pickle.load(f)

    # test
    # train_list = train_list[:24]

    train_dataset = CPMDataset(train_list, args.max_len)

    return train_dataset

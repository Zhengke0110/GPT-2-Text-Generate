import torch
from torch.utils.data import Dataset


class CPMDataset(Dataset):
    """
    CPM模型的数据集类
    """
    def __init__(self, data_list, max_len):
        """
        初始化数据集
        
        Args:
            data_list: 预处理后的token id列表
            max_len: 最大序列长度
        """
        self.data_list = data_list
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        获取单个数据项
        
        Args:
            index: 数据索引
            
        Returns:
            torch.Tensor: token ids的张量
        """
        data = self.data_list[index]
        
        # 截断或填充到max_len长度
        if len(data) > self.max_len:
            data = data[:self.max_len]
        
        # 转换为tensor
        data = torch.tensor(data, dtype=torch.long)
        
        return data

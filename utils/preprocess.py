from .setLogging import set_logger
from transformers import CpmTokenizer
import os
import pickle
from tqdm import tqdm


def preprocess(vocab_file, log_path, data_path, save_path, win_size, step, data_type="zuowen"):
    """
    对故事数据集进行预处理
    
    Args:
        vocab_file: 词表路径
        log_path: 日志存放位置
        data_path: 数据集存放位置
        save_path: 对训练数据集进行tokenize之后的数据存放位置
        win_size: 滑动窗口的大小，相当于每条数据的最大长度
        step: 滑动窗口的滑动步幅
        data_type: 数据类型，支持 "zuowen" 和 "novel"
    """

    # 初始化日志对象
    logger = set_logger(log_path)

    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file=vocab_file)  # pip install jieba
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id

    # 读取数据集目录下的所有文件
    train_list = []
    logger.info(f"start tokenizing {data_type} data")
    for file in tqdm(os.listdir(data_path)):
        file = os.path.join(data_path, file)
        with open(file, "r", encoding="utf8") as reader:
            lines = reader.readlines()
            
            if data_type == "zuowen":
                # 作文数据处理逻辑
                title = lines[1][3:].strip()  # 取出标题
                lines = lines[7:]  # 取出正文内容
                article = ""
                for line in lines:
                    if line.strip() != "":  # 去除换行
                        article += line
                title_ids = tokenizer.encode(title, add_special_tokens=False)
                article_ids = tokenizer.encode(article, add_special_tokens=False)
                token_ids = title_ids + [sep_id] + article_ids + [eod_id]
                
                # 对于每条数据，使用滑动窗口对其进行截断
                start_index = 0
                end_index = win_size
                data = token_ids[start_index:end_index]
                train_list.append(data)
                start_index += step
                end_index += step
                while end_index + 50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
                    data = token_ids[start_index:end_index]
                    train_list.append(data)
                    start_index += step
                    end_index += step
                    
            elif data_type == "novel":
                # 小说数据处理逻辑
                for i in range(len(lines)):
                    if lines[i].isspace() != True and lines[i] != '\n':
                        token_ids = tokenizer.encode(lines[i].strip(), add_special_tokens=False) + [eod_id]
                        if i % 1000 == 0:
                            logger.info(f'cur_step {i}: {lines[i].strip()[:50]}...')  # 使用logger而不是print
                        
                        # 对于每条数据，使用滑动窗口对其进行截断
                        start_index = 0
                        end_index = win_size
                        data = token_ids[start_index:end_index]
                        train_list.append(data)
                        start_index += step
                        end_index += step
                        while end_index + 50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
                            data = token_ids[start_index:end_index]
                            train_list.append(data)
                            start_index += step
                            end_index += step
            else:
                raise ValueError(f"Unsupported data_type: {data_type}. Supported types: 'zuowen', 'novel'")

    # 序列化训练数据
    with open(save_path, "wb") as f:
        pickle.dump(train_list, f)

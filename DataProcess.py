import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

import argparse
from utils.preprocess import preprocess

# 作文数据示例: --data_path data/zuowen --save_path data/train.pkl --win_size 200 --step 200 --data_type zuowen
# 小说数据示例: --data_path data/novel --save_path data/train_novel.pkl --win_size 200 --step 200 --data_type novel
# https://huggingface.co/docs/transformers/main/en/model_doc/cpm#transformers.CpmTokenizer


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="对故事数据集进行预处理")
    parser.add_argument(
        "--vocab_file",
        default="vocab/chinese_vocab.model",
        type=str,
        required=False,
        help="词表路径",
    )
    parser.add_argument(
        "--log_path",
        default="log/preprocess.log",
        type=str,
        required=False,
        help="日志存放位置",
    )
    parser.add_argument(
        "--data_path",
        default="data/zuowen",
        type=str,
        required=False,
        help="数据集存放位置",
    )
    parser.add_argument(
        "--save_path",
        default="data/train.pkl",
        type=str,
        required=False,
        help="对训练数据集进行tokenize之后的数据存放位置",
    )
    parser.add_argument(
        "--win_size",
        default=200,
        type=int,
        required=False,
        help="滑动窗口的大小，相当于每条数据的最大长度",
    )
    parser.add_argument(
        "--step", default=200, type=int, required=False, help="滑动窗口的滑动步幅"
    )
    parser.add_argument(
        "--data_type",
        default="zuowen",
        type=str,
        required=False,
        choices=["zuowen", "novel"],
        help="数据类型：zuowen（作文数据）或 novel（小说数据）"
    )
    args = parser.parse_args()

    # 调用预处理函数
    preprocess(
        vocab_file=args.vocab_file,
        log_path=args.log_path,
        data_path=args.data_path,
        save_path=args.save_path,
        win_size=args.win_size,
        step=args.step,
        data_type=args.data_type
    )

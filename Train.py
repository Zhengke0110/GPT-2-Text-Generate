import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

import argparse
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from os.path import join
import transformers
from utils.setLogging import set_logger
from utils.dataParallel import BalancedDataParallel
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    CpmTokenizer,
)  # pip install sentencepiece
from utils.trainUtils import set_random_seed, calculate_acc, calculate_loss, train_epoch
from utils.dataUtils import collate_fn, load_dataset


# --epochs 5 --batch_size 4 --device 0,1 --train_path data/train.pkl
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="0,1", type=str, required=False, help="设置使用哪些显卡"
    )
    parser.add_argument("--no_cuda", action="store_true", help="不使用GPU进行训练")
    parser.add_argument(
        "--vocab_path",
        default="vocab/chinese_vocab.model",
        type=str,
        required=False,
        help="sp模型路径",
    )
    parser.add_argument(
        "--model_config",
        default="config/cpm-small.json",
        type=str,
        required=False,
        help="需要从头训练一个模型时，模型参数的配置文件",
    )
    parser.add_argument(
        "--train_path",
        default="data/train.pkl",
        type=str,
        required=False,
        help="经过预处理之后的数据存放路径",
    )
    parser.add_argument(
        "--max_len",
        default=200,
        type=int,
        required=False,
        help="训练时，输入数据的最大长度",
    )

    parser.add_argument(
        "--log_path",
        default="log/train.log",
        type=str,
        required=False,
        help="训练日志存放位置",
    )
    parser.add_argument(
        "--ignore_index",
        default=-100,
        type=int,
        required=False,
        help="对于ignore_index的label token不计算梯度",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, required=False, help="训练的最大轮次"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, required=False, help="训练的batch size"
    )
    parser.add_argument(
        "--gpu0_bsz", default=6, type=int, required=False, help="0号卡的batch size"
    )
    parser.add_argument(
        "--lr", default=1.5e-4, type=float, required=False, help="学习率"
    )
    parser.add_argument(
        "--eps", default=1.0e-09, type=float, required=False, help="AdamW优化器的衰减率"
    )
    parser.add_argument(
        "--log_step", default=10, type=int, required=False, help="多少步汇报一次loss"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=6,
        type=int,
        required=False,
        help="梯度积累的步数",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False)
    parser.add_argument(
        "--save_model_path",
        default="model",
        type=str,
        required=False,
        help="模型输出路径",
    )
    parser.add_argument(
        "--pretrained_model",
        default="model/novel/epoch50",
        type=str,
        required=False,
        help="预训练的模型的路径，留空则从头开始训练",
    )
    parser.add_argument("--seed", type=int, default=1234, help="设置随机种子")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader加载数据时使用的线程数量"
    )
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="warm up步数")
    # parser.add_argument('--label_smoothing', default=True, action='store_true', help='是否进行标签平滑')
    args = parser.parse_args()
    return args


def train(model, logger, train_dataset, args):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info("total_steps:{}".format(len(train_dataloader) * args.epochs))
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )  # 设置warmup

    logger.info("start training")

    train_losses = []  # 记录每个epoch的平均loss
    # ========== start training ========== #
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            epoch=epoch,
            args=args,
        )
        train_losses.append(round(train_loss, 4))
        logger.info("train loss list:{}".format(train_losses))

    logger.info("training finished")
    logger.info("train_losses:{}".format(train_losses))


def main():
    # 初始化参数
    args = set_args()

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = not args.no_cuda

    # if args.batch_size < 2048 and args.warmup_steps <= 4000:
    #     print('[Warning] The warmup steps may be not enough.\n' \
    #           '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
    #           'Using smaller batch w/o longer warmup may cause ' \
    #           'the warmup stage ends with only little data trained.')

    # 创建日志对象
    logger = set_logger(args.log_path)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = "cuda:0" if args.cuda else "cpu"
    args.device = device
    logger.info("using device:{}".format(device))

    # 设置随机种子
    set_random_seed(args.seed, args.cuda)

    # 初始化tokenizer https://www.sciencedirect.com/science/article/pii/S266665102100019X
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    args.pad_id = tokenizer.pad_token_id

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info("model config:\n{}".format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 多卡并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        # model = DataParallel(model).cuda()
        model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("number of model parameters: {}".format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, args)


if __name__ == "__main__":
    main()

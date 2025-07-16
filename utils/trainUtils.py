"""
训练相关的工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os
from os.path import join


def set_random_seed(seed, cuda):
    """
    设置随机种子，确保结果可复现

    Args:
        seed: 随机种子值
        cuda: 是否使用CUDA
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_acc(logit, labels, ignore_index=-100):
    """
    计算预测准确率

    Args:
        logit: 模型输出的logits
        labels: 真实标签
        ignore_index: 忽略的索引值

    Returns:
        n_correct: 预测正确的数量
        n_word: 总词数
    """
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def calculate_loss(logit, target, pad_idx, smoothing=True):
    """
    计算损失函数

    Args:
        logit: 模型输出的logits
        target: 目标标签
        pad_idx: padding索引
        smoothing: 是否使用标签平滑

    Returns:
        loss: 计算得到的损失值
    """
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args):
    """
    训练一个epoch

    Args:
        model: 训练模型
        train_dataloader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        logger: 日志记录器
        epoch: 当前epoch数
        args: 训练参数

    Returns:
        epoch_mean_loss: 当前epoch的平均损失
    """
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()

    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0  # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(
                logits, labels, ignore_index=ignore_index
            )
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1,
                        epoch + 1,
                        loss.item() * args.gradient_accumulation_steps,
                        batch_acc,
                        scheduler.get_lr(),
                    )
                )

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(
            epoch + 1, epoch_mean_loss, epoch_mean_acc
        )
    )

    # save model
    logger.info("saving model for epoch {}".format(epoch + 1))
    model_path = join(args.save_model_path, "epoch{}".format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(model_path)
    logger.info("epoch {} finished".format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info("time for one epoch: {}".format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, CpmTokenizer


def set_args():
    """
    设置预测参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="0", type=str, required=False, help="设置使用哪些显卡"
    )
    parser.add_argument("--no_cuda", action="store_true", help="不使用GPU进行预测")
    parser.add_argument(
        "--vocab_path",
        default="vocab/chinese_vocab.model",
        type=str,
        required=False,
        help="sp模型路径",
    )
    parser.add_argument(
        "--model_path",
        default="model/epoch2",
        type=str,
        required=False,
        help="训练好的模型路径",
    )
    parser.add_argument(
        "--max_len",
        default=200,
        type=int,
        required=False,
        help="生成文本的最大长度",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="生成文本的温度参数，越高越随机",
    )
    parser.add_argument(
        "--top_k",
        default=50,
        type=int,
        required=False,
        help="top-k采样参数",
    )
    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        required=False,
        help="top-p采样参数",
    )
    parser.add_argument(
        "--repetition_penalty",
        default=1.2,
        type=float,
        required=False,
        help="重复惩罚参数",
    )
    parser.add_argument(
        "--prompt",
        default="",
        type=str,
        required=False,
        help="输入的提示文本",
    )
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        required=False,
        help="生成样本的数量",
    )
    parser.add_argument("--seed", type=int, default=1234, help="设置随机种子")

    args = parser.parse_args()
    return args


def filter_tokens(input_ids, vocab_size):
    """
    过滤超出词汇表范围的token
    """
    # 将超出范围的token替换为unk_token_id (通常是3)
    unk_token_id = 3
    filtered_ids = []
    for token_id in input_ids:
        if token_id >= vocab_size:
            print(f"警告: token {token_id} 超出词汇表范围，替换为 {unk_token_id}")
            filtered_ids.append(unk_token_id)
        else:
            filtered_ids.append(token_id)
    return filtered_ids


def safe_encode(tokenizer, text, vocab_size):
    """
    安全的编码函数，确保token在有效范围内
    """
    try:
        # 直接编码
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        # 过滤无效token
        input_ids = filter_tokens(input_ids, vocab_size)
        return input_ids
    except Exception as e:
        print(f"编码失败: {e}")
        # 如果编码失败，尝试逐字符编码
        input_ids = []
        for char in text:
            try:
                char_ids = tokenizer.encode(char, add_special_tokens=False)
                char_ids = filter_tokens(char_ids, vocab_size)
                input_ids.extend(char_ids)
            except:
                # 如果单个字符也无法编码，跳过
                print(f"跳过无法编码的字符: {char}")
                continue
        return input_ids


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    assert logits.dim() == 1  # batch size 1 for simplicity
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate_text(model, tokenizer, prompt, args, device):
    """
    生成文本
    """
    model.eval()
    vocab_size = model.config.vocab_size

    if prompt:
        # 安全编码输入文本
        input_ids = safe_encode(tokenizer, prompt, vocab_size)
        if not input_ids:
            print("无法编码输入文本，使用默认开始token")
            input_ids = (
                [tokenizer.bos_token_id] if tokenizer.bos_token_id < vocab_size else [1]
            )
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    else:
        # 如果没有提示文本，从BOS token开始
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id < vocab_size else 1
        input_ids = torch.tensor([[bos_id]], dtype=torch.long).to(device)

    generated_texts = []

    for sample_idx in range(args.num_samples):
        generated = input_ids.clone()

        with torch.no_grad():
            for step in range(args.max_len):
                try:
                    outputs = model(generated)
                    next_token_logits = outputs.logits[0, -1, :] / args.temperature

                    # 应用重复惩罚
                    for token_id in set(generated[0].tolist()):
                        if token_id < vocab_size:  # 确保token在有效范围内
                            next_token_logits[token_id] /= args.repetition_penalty

                    # 应用top-k和top-p过滤
                    filtered_logits = top_k_top_p_filtering(
                        next_token_logits, top_k=args.top_k, top_p=args.top_p
                    )

                    # 采样下一个token
                    probabilities = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                    # 检查生成的token是否有效
                    if next_token.item() >= vocab_size:
                        print(f"生成了无效token {next_token.item()}，使用随机有效token")
                        next_token = torch.randint(0, vocab_size - 1, (1,)).to(device)

                    # 检查是否生成结束符
                    if (
                        next_token.item() == tokenizer.eos_token_id
                        and tokenizer.eos_token_id < vocab_size
                    ) or next_token.item() == 2:  # 常见的EOS token ID
                        break

                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                except Exception as e:
                    print(f"生成过程中出错: {e}")
                    break

        # 解码生成的文本
        try:
            generated_text = tokenizer.decode(
                generated[0].tolist(), skip_special_tokens=True
            )
        except:
            # 如果解码失败，尝试逐个token解码
            tokens = generated[0].tolist()
            generated_text = ""
            for token in tokens:
                try:
                    if token < vocab_size:
                        decoded_token = tokenizer.decode([token])
                        generated_text += decoded_token
                except:
                    continue

        generated_texts.append(generated_text)

        if args.num_samples > 1:
            print(f"样本 {sample_idx + 1}: {generated_text[:100]}...")

    return generated_texts


def main():
    # 设置参数
    args = set_args()

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = CpmTokenizer(vocab_file=args.vocab_path)

    # 加载模型
    print(f"正在从 {args.model_path} 加载模型...")
    try:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.to(device)
        print("模型加载成功!")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"模型词汇表大小: {model.config.vocab_size}")
        print(f"Tokenizer词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 交互式生成文本
    print("\n" + "=" * 50)
    print("GPT-2 中文文本生成器 (安全模式)")
    print("输入 'quit' 退出程序")
    print("=" * 50 + "\n")

    while True:
        # 获取用户输入
        if not args.prompt:
            prompt = input("请输入提示文本 (按回车使用默认开始): ").strip()
            if prompt.lower() == "quit":
                break
        else:
            prompt = args.prompt

        print(f"\n正在生成文本...")
        print(f"提示文本: '{prompt}'")
        print(
            f"参数设置: max_len={args.max_len}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}"
        )
        print("-" * 50)

        # 生成文本
        try:
            generated_texts = generate_text(model, tokenizer, prompt, args, device)

            for i, text in enumerate(generated_texts, 1):
                print(f"\n生成文本 {i}:")
                print(text)
                print("-" * 30)

        except Exception as e:
            print(f"生成文本时出错: {e}")

        # 如果是通过命令行参数指定的prompt，只生成一次就退出
        if args.prompt:
            break

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, CpmTokenizer
import time

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log/gpt2_interactive.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class GPT2Generator:
    """GPT-2文本生成器类"""

    def __init__(
        self, model_path, vocab_path="vocab/chinese_vocab.model", device="auto"
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = self._setup_device(device)

        # 初始化模型和tokenizer
        self.tokenizer = None
        self.model = None
        self.vocab_size = None

        # 默认生成参数
        self.default_params = {
            "max_len": 150,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "num_samples": 1,
        }

        self.load_model()

    def _setup_device(self, device):
        """设置计算设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            return "cpu"
        else:
            return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型和tokenizer"""
        logger.info("正在初始化模型...")

        try:
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = CpmTokenizer(vocab_file=self.vocab_path)

            # 加载模型
            logger.info(f"从 {self.model_path} 加载模型...")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            self.vocab_size = self.model.config.vocab_size

            logger.info("模型加载成功!")
            logger.info(f"设备: {self.device}")
            logger.info(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"词汇表大小: {self.vocab_size}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def filter_tokens(self, input_ids):
        """过滤超出词汇表范围的token"""
        unk_token_id = 3
        filtered_ids = []
        for token_id in input_ids:
            if token_id >= self.vocab_size:
                filtered_ids.append(unk_token_id)
            else:
                filtered_ids.append(token_id)
        return filtered_ids

    def safe_encode(self, text):
        """安全编码文本"""
        try:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids = self.filter_tokens(input_ids)
            return input_ids
        except Exception:
            # 逐字符编码
            input_ids = []
            for char in text:
                try:
                    char_ids = self.tokenizer.encode(char, add_special_tokens=False)
                    char_ids = self.filter_tokens(char_ids)
                    input_ids.extend(char_ids)
                except:
                    continue
            return input_ids

    def top_k_top_p_filtering(
        self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")
    ):
        """Top-k和top-p过滤"""
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def generate(self, prompt, **kwargs):
        """生成文本"""
        # 合并参数
        params = self.default_params.copy()
        params.update(kwargs)

        start_time = time.time()

        if prompt:
            input_ids = self.safe_encode(prompt)
            if not input_ids:
                input_ids = [1]  # 使用BOS token
            input_ids = (
                torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            )
        else:
            input_ids = torch.tensor([[1]], dtype=torch.long).to(self.device)

        generated_texts = []

        for sample_idx in range(params["num_samples"]):
            generated = input_ids.clone()

            with torch.no_grad():
                for step in range(params["max_len"]):
                    try:
                        outputs = self.model(generated)
                        next_token_logits = (
                            outputs.logits[0, -1, :] / params["temperature"]
                        )

                        # 重复惩罚
                        for token_id in set(generated[0].tolist()):
                            if token_id < self.vocab_size:
                                next_token_logits[token_id] /= params[
                                    "repetition_penalty"
                                ]

                        # 过滤
                        filtered_logits = self.top_k_top_p_filtering(
                            next_token_logits,
                            top_k=params["top_k"],
                            top_p=params["top_p"],
                        )

                        # 采样
                        probabilities = F.softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(probabilities, num_samples=1)

                        # 检查有效性
                        if next_token.item() >= self.vocab_size:
                            next_token = torch.randint(0, self.vocab_size - 1, (1,)).to(
                                self.device
                            )

                        # 检查结束符
                        if next_token.item() == 2:  # EOS
                            break

                        generated = torch.cat(
                            [generated, next_token.unsqueeze(0)], dim=1
                        )

                    except Exception as e:
                        logger.warning(f"生成过程出错: {e}")
                        break

            # 解码
            try:
                generated_text = self.tokenizer.decode(
                    generated[0].tolist(), skip_special_tokens=True
                )
            except:
                tokens = generated[0].tolist()
                generated_text = ""
                for token in tokens:
                    try:
                        if token < self.vocab_size:
                            decoded_token = self.tokenizer.decode([token])
                            generated_text += decoded_token
                    except:
                        continue

            generated_texts.append(generated_text)

        end_time = time.time()
        generation_time = end_time - start_time

        return generated_texts, generation_time

    def update_params(self, **kwargs):
        """更新生成参数"""
        self.default_params.update(kwargs)
        logger.info(f"参数已更新: {kwargs}")


def print_help():
    """打印帮助信息"""
    help_text = """
交互式GPT-2文本生成器

基本命令:
  直接输入文本 - 生成文本
  /help        - 显示帮助
  /params      - 显示当前参数
  /set <参数>  - 设置参数
  /model <路径> - 切换模型
  /clear       - 清屏
  /quit        - 退出程序

可设置参数:
  max_len          - 最大生成长度 (默认: 150)
  temperature      - 温度参数 (默认: 1.0)
  top_k           - Top-K采样 (默认: 50) 
  top_p           - Top-P采样 (默认: 0.95)
  repetition_penalty - 重复惩罚 (默认: 1.2)
  num_samples     - 生成样本数 (默认: 1)

示例:
  /set temperature 0.8
  /set max_len 200 temperature 0.9
  /model model/novel/epoch50
  武林传说
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(description="交互式GPT-2文本生成器")
    parser.add_argument("--model_path", default="model/epoch2", help="模型路径")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="计算设备"
    )
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU")

    args = parser.parse_args()

    if args.no_cuda:
        device = "cpu"
    else:
        device = args.device

    print("启动交互式GPT-2文本生成器")
    print("=" * 60)

    try:
        # 初始化生成器
        generator = GPT2Generator(args.model_path, device=device)

        print("\n" + "=" * 60)
        print("初始化完成！开始交互式生成")
        print("输入 '/help' 查看帮助，输入 '/quit' 退出")
        print("=" * 60)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入提示文本或命令: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    command = user_input[1:].split()
                    cmd = command[0].lower()

                    if cmd == "quit" or cmd == "exit":
                        print("再见！")
                        break

                    elif cmd == "help":
                        print_help()

                    elif cmd == "params":
                        print("当前生成参数:")
                        for key, value in generator.default_params.items():
                            print(f"   {key}: {value}")

                    elif cmd == "set":
                        if len(command) < 2:
                            print(
                                "错误：用法: /set <参数名> <值> [<参数名2> <值2> ...]"
                            )
                            continue

                        try:
                            # 解析参数
                            params = {}
                            for i in range(1, len(command), 2):
                                if i + 1 < len(command):
                                    param_name = command[i]
                                    param_value = command[i + 1]

                                    # 类型转换
                                    if param_name in [
                                        "max_len",
                                        "top_k",
                                        "num_samples",
                                    ]:
                                        params[param_name] = int(param_value)
                                    elif param_name in [
                                        "temperature",
                                        "top_p",
                                        "repetition_penalty",
                                    ]:
                                        params[param_name] = float(param_value)
                                    else:
                                        print(f"警告：未知参数: {param_name}")
                                        continue

                            generator.update_params(**params)

                        except ValueError as e:
                            print(f"错误：参数值错误: {e}")

                    elif cmd == "model":
                        if len(command) < 2:
                            print("错误：用法: /model <模型路径>")
                            continue

                        try:
                            new_model_path = " ".join(command[1:])
                            generator.model_path = new_model_path
                            generator.load_model()
                        except Exception as e:
                            print(f"错误：模型切换失败: {e}")

                    elif cmd == "clear":
                        os.system("cls" if os.name == "nt" else "clear")
                        print("交互式GPT-2文本生成器")
                        print("输入 '/help' 查看帮助")

                    else:
                        print(f"错误：未知命令: {cmd}，输入 '/help' 查看帮助")

                else:
                    # 生成文本
                    print(f"\n正在生成...")
                    print(f"提示: '{user_input}'")
                    print(f"参数: {generator.default_params}")
                    print("-" * 50)

                    try:
                        generated_texts, gen_time = generator.generate(user_input)

                        for i, text in enumerate(generated_texts, 1):
                            print(f"\n生成文本 {i}:")
                            print(text)
                            if i < len(generated_texts):
                                print("-" * 30)

                        print(f"\n生成用时: {gen_time:.2f}秒")

                    except Exception as e:
                        print(f"错误：生成失败: {e}")

            except KeyboardInterrupt:
                print("\n\n程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"错误：发生错误: {e}")
                continue

    except Exception as e:
        print(f"错误：初始化失败: {e}")
        return


if __name__ == "__main__":
    main()

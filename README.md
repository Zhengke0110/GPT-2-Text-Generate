# GPT-2 中文文本生成项目

基于 GPT-2 的中文文本生成系统，支持作文和小说数据的训练与生成。

## 项目结构

```
├── DataProcess.py          # 数据预处理脚本
├── Train.py               # 模型训练脚本
├── Predict.py             # 单次预测脚本
├── InteractivePredict.py  # 交互式预测脚本
├── config/                # 配置文件
│   └── cpm-small.json    # GPT-2模型配置
├── data/                  # 数据目录
│   ├── novel/            # 小说原始数据
│   ├── zuowen/           # 作文原始数据
│   ├── train.pkl         # 预处理后的训练数据
│   ├── train_novel.pkl   # 预处理后的小说数据
│   └── train_novel_small.pkl
├── model/                 # 模型保存目录
│   ├── novel/            # 小说模型
│   └── zuowen_epoch40/   # 作文模型
├── utils/                 # 工具函数
├── vocab/                 # 词表文件
└── log/                   # 日志文件
```

## 环境要求

- Python 3.7+
- PyTorch
- transformers
- sentencepiece
- tqdm

## 安装依赖

```bash
pip install torch transformers sentencepiece tqdm
```

## 使用说明

### 1. 数据预处理

在训练模型之前，需要先对原始数据进行预处理。

#### 作文数据预处理

```bash
python DataProcess.py \
    --data_path data/zuowen \
    --save_path data/train.pkl \
    --win_size 200 \
    --step 200 \
    --data_type zuowen
```

#### 小说数据预处理

```bash
python DataProcess.py \
    --data_path data/novel \
    --save_path data/train_novel.pkl \
    --win_size 200 \
    --step 200 \
    --data_type novel
```

**参数说明：**

- `--data_path`: 原始数据目录路径
- `--save_path`: 预处理后数据保存路径
- `--win_size`: 滑动窗口大小（文本最大长度）
- `--step`: 滑动窗口步长
- `--data_type`: 数据类型（zuowen 或 novel）
- `--vocab_file`: 词表文件路径（默认：vocab/chinese_vocab.model）

### 2. 模型训练

使用预处理好的数据训练 GPT-2 模型。

#### 基础训练命令

```bash
python Train.py \
    --epochs 5 \
    --batch_size 4 \
    --device 0,1 \
    --train_path data/train.pkl
```

#### 小说模型训练

```bash
python Train.py \
    --epochs 50 \
    --batch_size 4 \
    --device 0 \
    --train_path data/train_novel.pkl \
    --output_dir model/novel
```

**参数说明：**

- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--device`: GPU 设备号（支持多 GPU）
- `--train_path`: 训练数据路径
- `--output_dir`: 模型保存目录
- `--learning_rate`: 学习率（默认：1.5e-4）
- `--warmup_steps`: 预热步数（默认：2000）
- `--log_step`: 日志输出间隔（默认：1）
- `--no_cuda`: 强制使用 CPU 训练

### 3. 文本生成预测

项目提供两种预测方式：单次预测和交互式预测。

#### 单次预测

```bash
python Predict.py \
    --model_path model/epoch2 \
    --max_len 200 \
    --temperature 1.0 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 1.2 \
    --prefix "武林传说"
```

#### 交互式预测（推荐）

```bash
python InteractivePredict.py \
    --model_path model/epoch2 \
    --device auto
```

进入交互模式后，可以使用以下命令：

- 直接输入文本：生成续写内容
- `/help`：显示帮助信息
- `/params`：查看当前生成参数
- `/set <参数名> <值>`：设置生成参数
- `/model <路径>`：切换模型
- `/clear`：清屏
- `/quit`：退出程序

**生成参数说明：**

- `--max_len`: 生成文本最大长度（默认：150）
- `--temperature`: 温度参数，控制随机性（默认：1.0）
- `--top_k`: Top-K 采样参数（默认：50）
- `--top_p`: Top-P 采样参数（默认：0.95）
- `--repetition_penalty`: 重复惩罚系数（默认：1.2）
- `--num_samples`: 生成样本数量（默认：1）

### 4. 交互式预测示例

```bash
# 启动交互式预测
python InteractivePredict.py --model_path model/novel/epoch50

# 设置生成参数
/set temperature 0.8 max_len 300

# 输入提示文本
武林传说

# 切换模型
/model model/zuowen_epoch40

# 查看当前参数
/params
```

## 数据格式

### 输入数据

- **作文数据**：将`.txt`文件放入`data/zuowen/`目录
- **小说数据**：将`.txt`文件放入`data/novel/`目录
- 文件编码：UTF-8
- 每个文件包含完整的文本内容

### 预处理输出

- 生成`.pkl`格式的训练数据
- 使用滑动窗口切分长文本
- 进行 tokenization 处理

## 模型配置

项目使用 GPT-2 架构，默认配置：

- 词汇表大小：13317
- 嵌入维度：768
- 注意力头数：12
- 层数：12
- 上下文长度：1024

可以通过修改`config/cpm-small.json`来调整模型参数。

## 注意事项

1. **显存要求**：建议使用 4GB+显存的 GPU 进行训练
2. **数据质量**：输入文本质量直接影响生成效果
3. **训练时间**：小说数据训练 50 轮需要较长时间
4. **模型选择**：不同类型的数据建议训练专门的模型

## 日志文件

- 预处理日志：`log/preprocess.log`
- 训练日志：`log/train.log`
- 交互预测日志：`log/gpt2_interactive.log`

## 已训练模型

项目包含以下预训练模型：

- `model/novel/epoch50/`：小说专用模型
- `model/zuowen_epoch40/`：作文专用模型

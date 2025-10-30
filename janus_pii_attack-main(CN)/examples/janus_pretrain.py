# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint

import transformers
from datasets import load_dataset

# ------------------------- 参数与主类导入 -----------------------------
from pii_leakage.arguments.config_args import ConfigArgs        # 配置参数
from pii_leakage.arguments.dataset_args import DatasetArgs      # 数据集参数
from pii_leakage.arguments.env_args import EnvArgs              # 环境参数
from pii_leakage.arguments.model_args import ModelArgs          # 模型参数
from pii_leakage.arguments.ner_args import NERArgs              # NER参数
from pii_leakage.arguments.outdir_args import OutdirArgs        # 输出目录参数
from pii_leakage.arguments.sampling_args import SamplingArgs    # 采样参数
from pii_leakage.arguments.trainer_args import TrainerArgs      # 训练参数
from pii_leakage.dataset.real_dataset import RealDataset        # 真实数据集
from pii_leakage.models.language_model import LanguageModel     # 语言模型类
from pii_leakage.models.model_factory import ModelFactory       # 语言模型工厂
from pii_leakage.dataset.dataset_factory import DatasetFactory  # 数据集工厂
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted # 输出美化

# ------------------------- 加载WikiText辅助函数 ------------------------------
def load_wikitext():
    """ 加载 Huggingface WikiText 数据集，并规整为本项目可用格式 """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    results = []
    for row in dataset["train"]["text"]:
        if (row.strip() and not row.strip().startswith("=")):
            results.append(row)
    return Dataset.from_dict({'text': results})

# ------------------------- 参数解析 ------------------------------
def parse_args():
    # 解析命令行参数，生成各主流程参数对象（模型、NER、训练、数据集等）
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

# ------------------------- 微调预训练主流程 ------------------------------
def pre_train(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """
    1. 若有配置文件优先加载（覆盖参数）
    2. 加载实体标注并预处理的训练&验证数据
    3. 构造、加载和微调语言模型
    4. 先用隐私数据微调，再用WikiText覆盖数据模拟遗忘，观测模型敏感泄漏变化
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()
    print_dict_highlighted(vars(config_args.get_privacy_args()))    # 展示隐私训练配置
    # 1. 加载实体标注训练集/验证集
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)
    # 2. 加载模型
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()
    output_folder = outdir_args.create_folder_name()
    print_highlighted(f"模型保存路径: {output_folder}. 训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
    print_highlighted(f"训练样本示例: {train_dataset.shuffle().first()}")
    # 3. 模型在隐私数据集微调
    lm._fine_tune(train_dataset, eval_dataset, train_args)
    pprint(lm.generate(SamplingArgs(N=1)))  # 结果生成显示
    # 4. 用WikiText继续微调，模拟遗忘
    dataset_wiki = load_wikitext()
    train_dataset._base_dataset = dataset_wiki
    print_highlighted(f"覆盖后训练集样本: {train_dataset.shuffle().first()}")
    lm._fine_tune(train_dataset, eval_dataset, train_args)
    pprint(lm.generate(SamplingArgs(N=1)))

# 程序主入口
if __name__ == "__main__":
    pre_train(*parse_args())

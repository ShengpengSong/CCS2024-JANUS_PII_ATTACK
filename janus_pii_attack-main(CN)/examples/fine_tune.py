# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint

import transformers

# ------------------------- 参数导入 ------------------------------
from pii_leakage.arguments.config_args import ConfigArgs            # 配置参数
from pii_leakage.arguments.dataset_args import DatasetArgs          # 数据集参数
from pii_leakage.arguments.env_args import EnvArgs                  # 环境参数
from pii_leakage.arguments.model_args import ModelArgs              # 模型参数
from pii_leakage.arguments.ner_args import NERArgs                  # 实体识别参数
from pii_leakage.arguments.outdir_args import OutdirArgs            # 输出目录参数
from pii_leakage.arguments.privacy_args import PrivacyArgs          # 隐私参数
from pii_leakage.arguments.sampling_args import SamplingArgs        # 采样参数
from pii_leakage.arguments.trainer_args import TrainerArgs          # 训练参数
from pii_leakage.dataset.real_dataset import RealDataset            # 真实数据集类
from pii_leakage.models.language_model import LanguageModel         # 语言模型主类
from pii_leakage.models.model_factory import ModelFactory           # 语言模型工厂
from pii_leakage.dataset.dataset_factory import DatasetFactory      # 数据集工厂
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted # 输出美化

# ------------------------- 参数解析 ------------------------------
def parse_args():
    # 解析命令行参数，生成各主流程参数对象
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            PrivacyArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

# ------------------------- 微调主流程 ------------------------------
def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """
    训练主流程：先解析/加载配置，然后根据数据集及参数对模型进行训练，支持差分隐私训练。训练完成可采样生成文本。
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()
    print_dict_highlighted(vars(config_args.get_privacy_args())) # 展示隐私相关参数
    # 加载训练集和验证集
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)
    # 加载模型
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()
    output_folder = outdir_args.create_folder_name()
    print_highlighted(f"模型保存路径: {output_folder}. 训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
    print_highlighted(f"训练样本示例: {train_dataset.shuffle().first()}")
    # 微调训练，支持差分隐私参数
    lm.fine_tune(train_dataset, eval_dataset, train_args, privacy_args)
    # 采样生成文本观测模型效果
    pprint(lm.generate(SamplingArgs(N=1)))

# 程序主入口
if __name__ == "__main__":
    fine_tune(*parse_args())

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import transformers

# ------------------------- 参数导入 ------------------------------
from pii_leakage.arguments.attack_args import AttackArgs                # 攻击参数
from pii_leakage.arguments.config_args import ConfigArgs                # 配置参数
from pii_leakage.arguments.env_args import EnvArgs                      # 环境参数
from pii_leakage.arguments.model_args import ModelArgs                  # 模型参数
from pii_leakage.arguments.ner_args import NERArgs                      # NER参数
from pii_leakage.attacks.attack_factory import AttackFactory            # 工厂-生成攻击实例
from pii_leakage.attacks.extraction.naive_extraction import NaiveExtractionAttack # 简单提取攻击实现
from pii_leakage.models.language_model import LanguageModel             # 语言模型
from pii_leakage.models.model_factory import ModelFactory               # 语言模型工厂
from pii_leakage.utils.output import print_separator, bcolors, print_dict_highlighted # 输出美化

# ------------------------- 参数解析 ------------------------------
def parse_args():
    # 命令行参数对象解析
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            AttackArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

# ------------------------- PII批量抽取 ------------------------------
def extract_pii(model_args: ModelArgs,
                ner_args: NERArgs,
                attack_args: AttackArgs,
                env_args: EnvArgs,
                config_args: ConfigArgs):
    """
    利用攻击工厂、NER与模型联合批量生成文本并抽取PII信息
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()
    print_dict_highlighted(vars(attack_args))  # 输出攻击参数
    # 加载模型
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
    # 构造攻击器
    attack: NaiveExtractionAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    results: dict = attack.attack(lm, verbose=True)   # 执行PII抽取
    print_separator()
    print(f"{bcolors.OKBLUE}Best Guess:{bcolors.ENDC} {results}")
    print_separator()

# 主入口
if __name__ == "__main__":
    extract_pii(*parse_args())

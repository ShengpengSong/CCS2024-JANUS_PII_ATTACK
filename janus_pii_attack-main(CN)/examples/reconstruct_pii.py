# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import transformers

# ------------------------- 参数导入 ------------------------------
from pii_leakage.arguments.attack_args import AttackArgs                # 攻击参数
from pii_leakage.arguments.config_args import ConfigArgs                # 配置参数
from pii_leakage.arguments.env_args import EnvArgs                      # 环境参数
from pii_leakage.arguments.model_args import ModelArgs                  # 模型参数
from pii_leakage.arguments.ner_args import NERArgs                      # 实体识别参数
from pii_leakage.attacks.attack_factory import AttackFactory            # 工厂模式-生成攻击对象
from pii_leakage.attacks.reconstruction.perplexity_reconstruction import PerplexityReconstructionAttack # 重建攻击类
from pii_leakage.models.language_model import LanguageModel             # 语言模型
from pii_leakage.models.model_factory import ModelFactory               # 语言模型工厂
from pii_leakage.utils.output import print_dict_highlighted, bcolors, print_separator # 彩色输出工具

# ------------------------- 参数解析 ------------------------------
def parse_args():
    # 解析命令行参数并生成相关参数对象
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            AttackArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

# ------------------------- PII重建/恢复主流程 ------------------------------
def reconstruct_pii(model_args: ModelArgs,
                ner_args: NERArgs,
                attack_args: AttackArgs,
                env_args: EnvArgs,
                config_args: ConfigArgs):
    """
    已知文本包含<T-MASK>和<MASK>，模型推断哪一个候选PII最有可能是真实目标PII。
    给定一个被遮蔽的句子，其中 <T-MASK> 是目标遮蔽符（需要推断），而 <MASK> 是其他任何个人身份信息（PII）的遮蔽符，
    此函数会推断出目标遮蔽符最有可能的候选替换词。
    
    用于演示目的。
    """
    if config_args.exists():
        attack_args = config_args.get_attack_args()
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()
    print_dict_highlighted(vars(attack_args)) # 打印参数
    # 加载微调模型
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
    # 生成攻击器实例
    attack: PerplexityReconstructionAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    results: dict = attack.attack(lm, verbose=True) # 执行重建攻击
    # 输出推断结果，将<T-MASK>替换为预测PII并彩色显示
    target_pii = results[min(results, key=results.get)]
    full_sequence = attack_args.target_sequence.replace("<T-MASK>", f"{bcolors.OKGREEN}{target_pii}{bcolors.ENDC}")
    print_separator()
    print(f"{bcolors.OKBLUE}最好的猜测:{bcolors.ENDC} {full_sequence}")
    print_separator()

# 主入口
if __name__ == "__main__":
    reconstruct_pii(*parse_args())

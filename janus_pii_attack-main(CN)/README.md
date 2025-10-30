## Janus 接口：大语言模型微调如何加剧隐私风险

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.11-PyTorch-orange">
    </a>
    <a href="https://github.com/pytorch/opacus">
            <img alt="Build" src="https://img.shields.io/badge/1.12-opacus-orange">
    </a>
</p>

本仓库包含了我们发表于 ACM CCS 2024 论文的官方代码，使用了 GPT-2 语言模型与 Flair 实体识别（NER）模型。此代码基于 https://github.com/microsoft/analysing_pii_leakage ，并支持我们提出的针对性隐私攻击——Janus 攻击。

## 论文信息

> **Janus 接口：大语言模型微调如何加剧隐私风险**  
> 作者：Xiaoyi Chen, Siyuan Tang, Rui Zhu（等贡献）, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang.  
> ACM 计算机与通信安全大会 CCS'24，美国盐湖城。
> 
> [![arXiv](https://img.shields.io/badge/arXiv-2310.15469-green)](https://arxiv.org/abs/2310.15469)

## 环境搭建与运行

推荐使用 conda 环境管理项目依赖。

```shell
$ conda create -n pii-leakage python=3.10
$ conda activate pii-leakage
$ pip install -e .
```

## 使用说明

### examples 目录
**这一部分放的是项目的主要实验脚本，每个脚本对应一种流程：**
- **janus_pretrain.py**：模拟预训练过程，对语言模型进行初步训练。
- **janus_attack.py**：实现了Janus攻击，把特定PII注入到微调过程中，攻击模型隐私。
- **janus_evaluation.py**：对攻击效果做评测，包括提取准确率等。
- **fine_tune.py**：对语言模型在（包含或不包含DP技术）隐私数据集上进行微调。
- **extract_pii.py**：直接抽取模型生成文本中的PII。
- **evaluate.py**：评估模型的PII泄漏。
- **reconstruct_pii.py**：通过重构攻击还原被掩码的PII。

### configs 目录
**用来存放各类yaml配置文件，指定实验所需的参数、模型路径、输出地址等。**

**其中 `src/pii_leakage` 目录**：
- `arguments/`：用于脚本参数定义，比如模型参数、NLP处理参数、隐私参数、训练参数等（如 ModelArgs, NERArgs, TrainerArgs 等）。
- `attacks/`：各种攻击方法的具体实现，包括工厂模式（attack_factory.py）和多种攻击方式，如NaiveExtraction、PerplexityInference、PerplexityReconstruction。
- `dataset/`：封装数据集的加载、处理和切分，包括真实数据集和生成式数据集。
- `models/`：对模型的加载、微调/训练、生成等操作封装，包括各种语言模型和相应工厂类。
- `ner/`：实体识别相关类，包括Tagger（基础NER接口）、FlairTagger、结果处理（pii_results.py）、NER工厂模式（tagger_factory.py）。
- `utils/`：通用工具，如输出美化、回调、集合操作、web工具等。
- `extern/`：第三方扩展接口实现和自定义数据集处理。

我们说明如下主要功能。所有脚本均在 `./examples` 文件夹，运行配置在 `./configs` 文件夹。
- **预训练(Pretrain)**：通过持续学习模拟大语言模型的预训练过程。
- **攻击(Attack)**：对语言模型实施 Janus 攻击。
- **评估(Evaluation)**：对攻击效果进行评测。

## 预训练流程（预训练）

我们展示如何在 [ECHR 数据集](https://huggingface.co/datasets/ecthr_cases) 和 [WikiText 数据集](https://huggingface.co/datasets/Salesforce/wikitext) 上，模拟对 GPT-2（[Huggingface](https://huggingface.co/gpt2)）模型的持续预训练。

请在 `../configs/targted-attack/echr-gpt2-janus-pretrain.yml` 文件中，按需修改原始模型和保存的预训练模型的路径。默认输出文件夹为当前目录。

```shell
$ python janus_pretrain.py --config_path ../configs/targted-attack/echr-gpt2-janus-pretrain.yml
```

## 攻击流程（Attack）

请在 `../configs/targted-attack/echr-gpt2-janus-attack.yml` 文件调整 `model_ckpt` 字段，指向保存的预训练模型。调整 `root` 字段，指定攻击输出模型的目录。

```shell
$ python janus_attack.py --config_path ../configs/targted-attack/echr-gpt2-janus-attack.yml
```

## 评估流程（Evaluation）

在 `../configs/targted-attack/echr-gpt2-janus-eval.yml` 配置文件中修改 `model_ckpt` 字段，指向待评测模型。

```shell
$ python evaluate.py --config_path ../configs/targted-attack/echr-gpt2-janus-eval.yml
```

## 数据集说明

我们自带的 ECHR 数据集封装会对其中所有PII自动做标注。
PII实体标注工作由 Flair NER 模块完成，依据机器配置可花费数小时，但仅需运行一次（后续会缓存）。

## 微调模型

目前本仓库不直接提供已微调的模型。如有需要请邮件联系作者。

## 引用信息

如果您觉得本工作有帮助，欢迎引用我们的论文。

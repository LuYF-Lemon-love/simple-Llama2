# Llama2-Chinese

## 🗂️ 内容导引
- [Llama2-Chinese](#llama2-chinese)
  - [🗂️ 内容导引](#️-内容导引)
  - [🐼 国内Llama2最新下载地址！](#-国内llama2最新下载地址)
  - [📢 社区公告](#-社区公告)
  - [📝 数据来源](#-数据来源)
  - [⏬ 模型部署](#-模型部署)
    - [模型下载](#模型下载)
      - [Meta官方Llama2模型](#meta官方llama2模型)
      - [基于Llama2的中文微调模型](#基于llama2的中文微调模型)
      - [基于Llama2的中文预训练模型Atom](#基于llama2的中文预训练模型atom)
    - [模型调用代码示例](#模型调用代码示例)
    - [FastAPI接口搭建](#fastapi接口搭建)
    - [Gradio快速搭建问答平台](#gradio快速搭建问答平台)
    - [Docker部署问答接口](#docker部署问答接口)
  - [💡 模型微调](#-模型微调)
    - [微调过程](#微调过程)
      - [Step1: 环境准备](#step1-环境准备)
      - [Step2: 数据准备](#step2-数据准备)
      - [Step3: 微调脚本](#step3-微调脚本)
    - [加载微调模型](#加载微调模型)
  - [🍄 模型量化](#-模型量化)
  - [🚀 推理加速](#-推理加速)
    - [lmdeploy](#lmdeploy)
    - [FasterTransformer](#fastertransformer)
    - [vLLM](#vllm)
  - [🥇 模型评测](#-模型评测)
  - [💪 外延能力](#-外延能力)
    - [LangChain](#langchain)
  - [🐞 代码模型](#-代码模型)
  - [📖 学习资料](#-学习资料)
    - [Meta官方对于Llama2的介绍](#meta官方对于llama2的介绍)
    - [Llama相关论文](#llama相关论文)
    - [Llama2的评测结果](#llama2的评测结果)
  - [参考](#参考)

## 🐼 国内Llama2最新下载地址！

本仓库中的代码示例主要是基于Hugging Face版本参数进行调用，我们提供了脚本将Meta官网发布的模型参数转换为Hugging Face支持的格式，可以直接通过transformers库进行加载：[参数格式转化](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/scripts/convert2hf/READMD.md)


<details>

- Llama2-7B官网版本：https://pan.xunlei.com/s/VN_kR2fwuJdG1F3CoF33rwpIA1?pwd=z9kf

- Llama2-7B-Chat官网版本：https://pan.xunlei.com/s/VN_kQa1_HBvV-X9QVI6jV2kOA1?pwd=xmra

- Llama2-7B Hugging Face版本：https://pan.xunlei.com/s/VN_t0dUikZqOwt-5DZWHuMvqA1?pwd=66ep

- Llama2-7B-Chat Hugging Face版本：https://pan.xunlei.com/s/VN_oaV4BpKFgKLto4KgOhBcaA1?pwd=ufir

</details>

## 📢 社区公告

<details>

- 2023年8月26日：提供将Meta原始模型参数转换为兼容Hugging Face的[格式转化脚本](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/scripts/convert2hf/READMD.md)！

</details>

## 📝 数据来源

我们通过以下数据来优化Llama2的中文能力:

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| 网络数据                                                   | 互联网上公开的网络数据，挑选出去重后的高质量中文数据，涉及到百科、书籍、博客、新闻、公告、小说等高质量长文本数据。 |
| [Wikipedia](https://github.com/goldsmith/Wikipedia)        | 中文Wikipedia的数据                                          |
| [悟道](https://github.com/BAAI-WuDao/Model)                | 中文悟道开源的200G数据                                       |
| [Clue](https://github.com/CLUEbenchmark/CLUEDatasetSearch) | Clue开放的中文预训练数据，进行清洗后的高质量中文长文本数据   |
| 竞赛数据集                                                 | 近年来中文自然语言处理多任务竞赛数据集，约150个              |
| [MNBVC](https://github.com/esbatmop/MNBVC)                 | MNBVC 中清洗出来的部分数据集                                 |

## ⏬ 模型部署

Meta在🤗Hugging Face上提供了所有模型的下载链接：https://huggingface.co/meta-llama

Llama中文社区的中文模型下载链接：https://huggingface.co/FlagAlpha

### 模型下载

#### Meta官方Llama2模型

Llama2预训练模型包含7B、13B和70B三个版本。Llama2-Chat模型基于预训练模型进行了监督微调，具备更强的对话能力。

|  类别  | 模型名称   | 🤗模型加载名称             | 下载地址                                                     |
|  ----------  | ---------- | ------------------------- | --------------------- |
|  预训练  | Llama2-7B  | meta-llama/Llama-2-7b-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
|  预训练  | Llama2-13B | meta-llama/Llama-2-13b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
|  预训练  | Llama2-70B | meta-llama/Llama-2-70b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-hf) |
|  Chat  | Llama2-7B-Chat  | meta-llama/Llama-2-7b-chat-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
|  Chat  | Llama2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
|  Chat  | Llama2-70B-Chat | meta-llama/Llama-2-70b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |


#### 基于Llama2的中文微调模型

我们基于中文指令数据集对Llama2-Chat模型进行了微调，使得Llama2模型有着更强的中文对话能力。LoRA参数以及与基础模型合并的参数均已上传至[Hugging Face](https://huggingface.co/FlagAlpha)，目前包含7B和13B的模型。

|  类别  | 模型名称   | 🤗模型加载名称             | 基础模型版本 |    下载地址                                                     |
|  ----------  | ---------- | ------------- |  ----------------- | ------------------- |
|  合并参数 | Llama2-Chinese-7b-Chat | FlagAlpha/Llama2-Chinese-7b-Chat  |    meta-llama/Llama-2-7b-chat-hf       |[模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)  |
|  合并参数 | Llama2-Chinese-13b-Chat | FlagAlpha/Llama2-Chinese-13b-Chat|     meta-llama/Llama-2-13b-chat-hf     |[模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat) |
|  LoRA参数 | Llama2-Chinese-7b-Chat-LoRA  | FlagAlpha/Llama2-Chinese-7b-Chat-LoRA  |     meta-llama/Llama-2-7b-chat-hf      |[模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat-LoRA) |
|  LoRA参数 | Llama2-Chinese-13b-Chat-LoRA | FlagAlpha/Llama2-Chinese-13b-Chat-LoRA |     meta-llama/Llama-2-13b-chat-hf     |[模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-LoRA) |


#### 基于Llama2的中文预训练模型Atom

社区提供Atom-7B模型的开放下载，模型参数会持续不断更新，关于模型的进展详见社区官网[llama.family](https://llama.family)。

| 模型名称        | 🤗模型加载名称                  | 下载地址                                                     |
| --------------- | ------------------------------ | ------------------------------------------------------------ |
| Atom-7B  | FlagAlpha/Atom-7B  | [模型下载](https://huggingface.co/FlagAlpha/Atom-7B) |


### 模型调用代码示例

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Atom-7B',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

### FastAPI接口搭建

为了方便通过API方式调用模型，我们提供了脚本用来快速搭建[FastAPI](https://github.com/tiangolo/fastapi)接口，相关测试代码与API参数设置见[API 调用](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/scripts/api/READMD.md)。

### Gradio快速搭建问答平台

基于gradio搭建的问答界面，实现了流式的输出，将下面代码复制到控制台运行，以下代码以Atom-7B模型为例，<font color="#006600">不同模型只需修改一下代码里的模型名称就好了😊</font><br/>
```
python examples/chat_gradio.py --model_name_or_path FlagAlpha/Atom-7B
```

### Docker部署问答接口
详情参见：[Docker部署](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/docs/chat_gradio_guide.md)

第一步：准备docker镜像，通过docker容器启动[chat_gradio.py](../examples/chat_gradio.py)
```bash
git clone https://github.com/FlagAlpha/Llama2-Chinese.git

cd Llama2-Chinese

docker build -f docker/Dockerfile -t flagalpha/llama2-chinese-7b:gradio .
```

第二步：通过docker-compose启动chat_gradio
```bash
cd Llama2-Chinese/docker
doker-compose up -d --build
```


## 💡 模型微调

本仓库中提供了基于LoRA的微调代码，未来我们将会扩展更多的微调算法，敬请期待！关于LoRA的详细介绍可以参考论文“[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)”以及微软Github仓库[LoRA](https://github.com/microsoft/LoRA)。

### 微调过程

#### Step1: 环境准备

根据[requirements.txt](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/requirements.txt)安装对应的环境依赖。

#### Step2: 数据准备
在data目录下提供了一份用于模型sft的数据样例：
- 训练数据：[data/train_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/train_sft.csv)
- 验证数据：[data/dev_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/dev_sft.csv)

每个csv文件中包含一列“text”，每一行为一个训练样例，每个训练样例按照以下格式将问题和答案组织为模型输入，您可以按照以下格式自定义训练和验证数据集：
```
"<s>Human: "+问题+"\n</s><s>Assistant: "+答案
```
例如，
```
<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>
```

#### Step3: 微调脚本

我们提供了用于微调的脚本[train/sft/finetune.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune.sh)，通过修改脚本的部分参数实现模型的微调，关于微调的具体代码见[train/sft/finetune_clm_lora.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_clm_lora.py)，单机多卡的微调可以通过修改脚本中的`--include localhost:0`来实现。


### 加载微调模型
微调模型参数见：[基于Llama2的中文微调模型](#基于llama2的中文微调模型)，LoRA参数需要和基础模型参数结合使用。

通过[PEFT](https://github.com/huggingface/peft)加载预训练模型参数和微调模型参数，以下示例代码中，base_model_name_or_path为预训练模型参数保存路径，finetune_model_path为微调模型参数保存路径。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
finetune_model_path=''  
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```


<!-- ## 🚀 未来计划 -->


## 🍄 模型量化
我们对中文微调的模型参数进行了量化，方便以更少的计算资源运行。目前已经在[Hugging Face](https://huggingface.co/FlagAlpha)上传了13B中文微调模型[FlagAlpha/Llama2-Chinese-13b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat)的4bit压缩版本[FlagAlpha/Llama2-Chinese-13b-Chat-4bit](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-4bit)，具体调用方式如下：
```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized('FlagAlpha/Llama2-Chinese-13b-Chat-4bit', device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Llama2-Chinese-13b-Chat-4bit',use_fast=False)
input_ids = tokenizer(['<s>Human: 怎么登上火星\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

## 🚀 推理加速
随着大模型参数规模的不断增长，在有限的算力资源下，提升模型的推理速度逐渐变为一个重要的研究方向。常用的推理加速框架包含 lmdeploy、FasterTransformer 和 vLLM 等。

### lmdeploy
[lmdeploy](https://github.com/InternLM/lmdeploy/) 由上海人工智能实验室开发，推理使用 C++/CUDA，对外提供 python/gRPC/http 接口和 WebUI 界面，支持 tensor parallel 分布式推理、支持 fp16/weight int4/kv cache int8 量化。

详细的推理文档见：[inference-speed/GPU/lmdeploy_example](https://github.com/FlagAlpha/Llama2-Chinese/tree/main/inference-speed/GPU/lmdeploy_example)

### FasterTransformer
[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)由NVIDIA开发，采用C++/CUDA编写，支持分布式推理，transformer编码器和解码器均可进行加速。
通过FasterTransformer和[Triton](https://github.com/openai/triton)加速LLama2模型推理，目前支持FP16或者Int8推理，Int4目前还不支持。

详细的推理文档见：[inference-speed/GPU/FasterTransformer_example](https://github.com/FlagAlpha/Llama2-Chinese/tree/main/inference-speed/GPU/FasterTransformer_example)

### vLLM
[vLLM](https://github.com/vllm-project/vllm)由加州大学伯克利分校开发，核心技术是PageAttention，吞吐量比HuggingFace Transformers高出24倍。相较与FasterTrainsformer，vLLM更加的简单易用，不需要额外进行模型的转换，支持fp16推理。

详细的推理文档见：[inference-speed/GPU/vllm_example](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/inference-speed/GPU/vllm_example/README.md)


## 🥇 模型评测
为了能够更加清晰地了解Llama2模型的中文问答能力，我们筛选了一些具有代表性的中文问题，对Llama2模型进行提问。我们测试的模型包含Meta公开的Llama2-7B-Chat和Llama2-13B-Chat两个版本，没有做任何微调和训练。测试问题筛选自[AtomBulb](https://github.com/AtomEcho/AtomBulb)，共95个测试问题，包含：通用知识、语言理解、创作能力、逻辑推理、代码编程、工作技能、使用工具、人格特征八个大的类别。

测试中使用的Prompt如下，例如对于问题“列出5种可以改善睡眠质量的方法”：
```
[INST] 
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. The answer always been translate into Chinese language.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

The answer always been translate into Chinese language.
<</SYS>>

列出5种可以改善睡眠质量的方法
[/INST]
```
Llama2-7B-Chat的测试结果见[meta_eval_7B.md](assets/meta_eval_7B.md)，Llama2-13B-Chat的测试结果见[meta_eval_13B.md](assets/meta_eval_13B.md)。

通过测试我们发现，Meta原始的Llama2 Chat模型对于中文问答的对齐效果一般，大部分情况下都不能给出中文回答，或者是中英文混杂的形式。因此，基于中文数据对Llama2模型进行训练和微调十分必要，我们的中文版Llama2模型也已经在训练中，近期将对社区开放。


## 💪 外延能力

除了持续增强大模型内在的知识储备、通用理解、逻辑推理和想象能力等，未来，我们也会不断丰富大模型的外延能力，例如知识库检索、计算工具、WolframAlpha、操作软件等。
我们首先集成了LangChain框架，可以更方便地基于Llama2开发文档检索、问答机器人和智能体应用等，关于LangChain的更多介绍参见[LangChain](https://github.com/langchain-ai/langchain)。
### LangChain
针对LangChain框架封装的Llama2 LLM类见[examples/llama2_for_langchain.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/examples/llama2_for_langchain.py)，简单的调用代码示例如下：
```python
from llama2_for_langchain import Llama2

# 这里以调用4bit量化压缩的Llama2-Chinese参数FlagAlpha/Llama2-Chinese-13b-Chat-4bit为例
llm = Llama2(model_name_or_path='FlagAlpha/Llama2-Chinese-13b-Chat-4bit', bit4=True)

while True:
    human_input = input("Human: ")
    response = llm(human_input)
    print(f"Llama2: {response}")
```

## 🐞 代码模型
Meta官方在2023年8月24日发布了发布了Code Llama，基于代码数据对Llama2进行了微调，提供三个不同功能的版本：基础模型（Code Llama）、Python专用模型（Code Llama - Python）和指令跟随模型（Code Llama - Instruct），包含7B、13B、34B三种不同参数规模。不同模型能力区别如下表所示：

|  模型类别          |        模型名称         | 代码续写 | 代码填充 | 指令编程 |
|-----------------------|------------------------|------|------|------|
| Code Llama            | CodeLlama-7b           | ✅    | ✅    | ❌    |
|                       | CodeLlama-13b          | ✅    | ✅    | ❌    |
|                       | CodeLlama-34b          | ✅    | ❌    | ❌    |
| Code Llama - Python   | CodeLlama-7b-Python    | ✅    | ❌    | ❌    |
|                       | CodeLlama-13b-Python   | ✅    | ❌    | ❌    |
|                       | CodeLlama-34b-Python   | ✅    | ❌    | ❌    |
| Code Llama - Instruct | CodeLlama-7b-Instruct  | ❌    | ✅    | ✅    |
|                       | CodeLlama-13b-Instruct | ❌    | ✅    | ✅    |
|                       | CodeLlama-34b-Instruct | ❌    | ❌    | ✅    |

我们提供了Code Llama的[国内下载链接](#-国内llama2最新下载地址上线)以及在线体验地址[llama.family](https://llama.family/)，关于Code Llama的详细信息可以参考官方Github仓库[codellama](https://github.com/facebookresearch/codellama)。


## 📖 学习资料
### Meta官方对于[Llama2](https://ai.meta.com/llama)的介绍
自从Meta公司发布第一代LLaMA模型以来，羊驼模型家族繁荣发展。近期Meta发布了Llama2版本，开源可商用，在模型和效果上有了重大更新。Llama2总共公布了7B、13B和70B三种参数大小的模型。相比于LLaMA，Llama2的训练数据达到了2万亿token，上下文长度也由之前的2048升级到4096，可以理解和生成更长的文本。Llama2 Chat模型基于100万人类标记数据微调得到，在英文对话上达到了接近ChatGPT的效果。      

### Llama相关论文
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
### Llama2的评测结果
<p align="center" width="100%">
<img src="./assets/llama_eval.jpeg" style="width: 100%; display: block; margin: auto;">
</p>

## 参考

[1] Llama2-Chinese: [FlagAlpha/Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese), [Llama 中文社区](https://llama.family/), [飞书知识库文档](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink)
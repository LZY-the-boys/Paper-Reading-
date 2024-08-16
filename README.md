# Paper-Reading

## Data-Mining

- D-CPT Law: Domain-specific Continual Pre-Training Scaling Law
- Instruction-Mining
- [domain upsample](https://arxiv.org/pdf/2406.03476) 
- LLaMA3.1 tech report 数据清理策略：
	- Post-training
		- preference data：多个模型生成，让annotators去标注或者edit，用edited > chosen > rejected排序, 只采样 chosen明显好于rejected的数据，防止混淆, 并且在每一轮都会加大难度
		- SFT data: 用reward model选择 最新模型的对话回复 ，并在后期加system prompt引导风格语气
		- data clean： identify overused phrases (such as “I’m sorry” or “I apologize”)， excessive use of emojis or exclamation points
		- data pruning:  （topic）llama8b 作为topic classifier；（quality） llama3 2/3个级别的质量打分，以及reward model 前1/4的打分 两者 **或关系**； （difficulty）Instag 意图数量和 Llama 3个级别的打分；(semantic deduplication) RoBERTa cluster
		- average models: branch-train-mix
		- 按照Llama 2的做法，我们应用上述方法进行六轮迭代。在每一轮中，我们收集新的偏好注释和SFT数据，从最新模型中采样合成数据。
	- Pre-training
		- data mix： contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.
		- data annealing: 在大模型训练的最后阶段，用高质量的数据学习能提高性能。于是在最后40B数据上，作者逐渐将学习率衰减到0。并且作者发现，数据退火方法，可以用来筛数据
		- long-context pretraining: 用6个stage 逐步将长度从8k扩展到128k，并且加attention mask避免不同数据串味（对长文影响很大）
		- 数据的安全性和质量，web data curation
			1. 过滤器移除可能含有不安全内容或大量个人身份信息（PII）的网站数据，以及根据多种Meta安全标准被评为有害的域名和已知含有成人内容的域名。
			2. 使用自定义解析器处理非截断的网页文档 在URL、文档和行级别进行多轮去重：保留每个URL对应的最新版本页面。使用全局MinHash去重，移除近似重复的文档。进行类似ccNet的line级别去重，移除在每3000万文档桶中出现超过6次的line
			3. 开发启发式规则移除额外的低质量文档、异常值和重复过多的文档。使用重复n-gram覆盖率去除由日志或错误消息组成的重复内容行，使用“dirty word”计数过滤未被域名阻止列表覆盖的成人网站，以及使用令牌分布的KL divergence过滤含有异常数量的异常令牌的文档。
			4. **基于模型的质量过滤**：实验性地应用各种基于模型的质量分类器来筛选高质量的标记。包括使用fasttext快速分类器识别可能被维基百科引用的文本，基于Roberta的分类器，它们在Llama 2预测上进行训练。
			5. **代码和推理数据**：类似于DeepSeek-AI等，构建特定领域的管道提取代码和与数学相关的网页。代码和推理分类器都是基于Llama 2标注的网页数据训练的DistilledRoberta模型。
			6. **多语言数据**：使用基于fasttext的语言识别模型将文档分类为176种语言。

## Init

- Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer https://thegregyang.com/
	- 保证参数初始化+优化器 使得其和width 无关, hidden是两个都无穷，input和output是只有一个无穷, 两个都无穷用的中心极限，一个无穷用的大数定律
   	- µP 能够确保在模型大小变化时，许多最优超参数保持稳定。这使得可以从较小模型间接调优超参数，然后零样本（zero-shot）迁移到全尺寸模型上
  

## LoRA

- 架构
	- VeRA: freezes random weight tied adapters and learns vector scalings of the internal adapter activations.
	- LoRA-XS: initializes the A and B matrices using the SVD of the pretrained weights and trains a low-rank update of the form BRA where R is a trainable r × r matrix and B, A are fixed.
	- NOLA: parametrizes the adapter matrices to be linear combinations of frozen random matrices and optimizes the linear coefficients of the mixtures.
	- VB-LORA: shares adapter parameters using a global vector bank.
	- MoRA: learns high-rank updates while still preserving parameter efficiency by applying hand-designed compress and decompress operations before and after a trainable adapter matrix.
	- DoRA: decomposes the pretrained weight into magnitude and direction components to allow for better training dynamics
	- GaLoRe: 使用SVD将全参数训练的梯度投影到低秩空间
	- IA3（Implicit Activation Scaling）: 通过修改激活向量的缩放来适应模型，而不是调整权重。
- 训练改进
	- LoRA-FA: freezes the A matrix which leads to small performance loss while reducing memory consumption
	  by up to 1.4×.  
	- https://arxiv.org/pdf/2406.08447v1 [initA] > [initB] 通过对神经网络宽度极限的理论分析（uP）
	- LoRA+: 同样研究无限宽度下的初始化，结论是给AB不同的学习率
	- Pissa：对W0做SVD来初始化A,B
	- LoRA-GA：尽量对齐第一步更新后的W1，对初始梯度G0=∇W0L做SVD，取U的前r列初始化A，取V的第r+1∼2r行初始化B
	- **LoRA-Pro**: 对齐全量微调和LoRA的每一个Wt,
- 效果
	- https://arxiv.org/pdf/2405.09673 LoRA在目标领域的性能通常低于全参数微调，但在保持源领域性能方面表现更好；LoRA提供了比传统正则化技术(finetuned, weight-decay)更强的正则化效果，并有助于保持生成多样性
  - QLoRA: matched full finetuning MMLU (Hendrycks et al., 2020) performance, optimized LoRA configurations perform as well as full finetuning, and that performance is governed by choice of target modules but not rank.
  - DoRA: shows that LoRA is sensitive to ranks. It is likely that some of these discrepancies
    are due to differences in finetuning datasets and evaluations.  

## Decoding-Speedup

- Survey: Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding [[paper]](https://arxiv.org/abs/2401.07851)
- Fast Inference from Transformers via Speculative Decoding [[paper]](https://arxiv.org/pdf/2211.17192)[[repo]](https://github.com/feifeibear/LLMSpeculativeSampling)
- Accelerating Large Language Model Decoding with Speculative Sampling [[paper]](https://arxiv.org/pdf/2302.01318)
- [ASPLOS'24] SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification [[paper]](https://arxiv.org/abs/2305.09781)
- [ICML24] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty [[paper]](https://arxiv.org/pdf/2401.15077) [[blog]](https://sites.google.com/view/eagle-llm)
- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees [[paper]](https://arxiv.org/pdf/2406.16858)
- [ICLR24] DistillSpec: Improving Speculative Decoding via Knowledge Distillation [[paper]](https://arxiv.org/abs/2310.08461) 用target model作为teacher对draft model蒸馏
- [NAACL24] REST: Retrieval-Based Speculative Decoding [[paper]](https://arxiv.org/pdf/2311.08252)
- Graph-Structured Speculative Decoding [[paper]](https://arxiv.org/pdf/2407.16207)

## KV-Cache

## Long-Context

## Image-Tokenizer


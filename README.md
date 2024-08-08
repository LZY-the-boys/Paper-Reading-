# Paper-Reading

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

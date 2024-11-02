# Paper-Reading

## RLHF

### PRM 
- Deepmind PAV： https://arxiv.org/pdf/2410.08146
  - 只做了math
  - 有效的步骤奖励应该衡量“进展”，即在执行某个步骤之前和之后，产生正确响应的可能性的变化，恰好对应强化学习中advantage的概念
  - 所以，方法就是对每个step设法标注（state, action, advantage）用这个数据训练一个PRM（regression-style）
  - advantage的计算方式是，我们对当前state进行mcts搜索，采样一堆final output，算准确率,可以计算return，然后用return计算V和Q， 用bellman equation计算advantage A=Q- V
  - 但是这个paper的insight是说，这个advantage我们不能直接用原始模型/策略计算(base policy, \pi), 而是要引入一个新的prover policy(\mu), 用他采样，选择的条件和理由：
    1. 多样化数据来源：不同于基础策略的数据可以提供更多的多样性和不同的视角，从而丰富训练数据集。这有助于PRM学会更广泛的情况，而不仅仅局限于基础策略所能覆盖的情形。
    2. 补充优势：证明者策略（prover policy）通常是设计来补充基础策略的不足之处。如果基础策略在某些地方不够强，证明者策略可以帮助识别这些弱点，并提供额外的信息，使得过程奖励模型能够更好地评估每一步的影响。
    3. 减少过拟合：使用来自不同策略的数据可以减少模型在训练过程中过度拟合到特定策略的行为。通过引入不同策略产生的数据，PRM可以学习到更为泛化的模式，而不是仅仅优化特定策略的输出。
    4. 提高鲁棒性：通过让PRM接触到不同策略产生的轨迹，可以提高其鲁棒性，使其在面对多种不同的输入时都能给出合理的评估，从而在实际应用中表现更好。
    5. 理论支持：论文中的理论分析表明，选择好的证明者策略可以确保对基础策略进行非平凡的改进。即使是弱的证明者策略也可以显著改善更强的基础策略，这是因为它们能够提供与基础策略不同的优势，从而帮助基础策略更好地学习。
    6.  证明者 \(\mu\) 既不能过于强大，也不能过于薄弱，否则它所提供的过程奖励将无法有效地指导基础策略 \(\pi\) 的改进。
	      1. 如果证明者 \(\mu\) 与基础策略 \(\pi\) 相同，那么产生的过程奖励将等同于只优化最终结果的情况，这对于改进策略是没有帮助的。
	      2. 如果证明者 \(\mu\) 太弱，则它可能会面临与基础策略 \(\pi\) 相似的问题，即无法提供有效的反馈。
	      3. 相反，如果证明者 \(\mu\) 非常强大，那么即使在基础策略 \(\pi\) 执行无关紧要的步骤时，强大的证明者 \(\mu\) 也能成功地从这些状态中找到解决方案，这导致过程奖励 \(A_{\mu}\) 接近于零，因为它没有区分哪些步骤有助于解决问题。
	      4. 因此，论文指出，有效的证明者策略应该是那些能够补充基础策略 \(\pi\) 的策略，即能够有效地区分由基础策略产生的不同步骤，并提供与基础策略对齐的步骤级优势。
  - 因此，具体实现上，用BoN（N=4）作为prover policy，用prover policy sample出来的结果计算advantage，标注数据，训练prm；在RL训练过程中，使用的reward是 Q \pi + Advantage \mu (用prm预测)
  - Special cases: 
    - Self-explore to avoid the pit: Improving the reasoning capabilities of language models with fine-grained rewards.
    - Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold.
    - Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment
- Q-ranking https://arxiv.org/pdf/2410.11287
  - 只做了math
  - 推理中正确的和错误的步骤不同，应该对应不同的score, 因此我们可以先定义一种顺序，第一步错< ... < 第N步错< 第1步对 < ... < 第N步对，然后会有一堆排序分数，用ranking loss训练prm
  - 证明这个顺序理论，用的dpo q-function理论，
- MATH-Shepherd: https://arxiv.org/pdf/2312.08935
  - 用cross entropy训练PRM
    - HE假设只要一个步骤能够到达正确答案，它就是一个好步骤。 
    - SE则将步骤的质量视为它达到正确答案的频率。
  - used automated supervision to annotate steps with 𝑄 values under the base policy, i.e., the PRMs score a step with the likelihood of future success, when continuing to sample from the step.
- https://arxiv.org/pdf/2410.13121， https://arxiv.org/pdf/2410.13246
  - 目前PRM很难推广到其他领域
  - 在其他领域上的PRM，利用programs that can be executed over the scene graph object to verify each QA pair, 和 atomic fact，来做每步的验证
- DPO - Q-function: https://arxiv.org/pdf/2404.12358v2
  	- under the token level formulation, classical search-based algorithms, such as MCTS, which have recently been applied to the language generation space, are equivalent to likelihood-based search on a DPO policy
- Generative Verifier https://arxiv.org/pdf/2408.15240
- PRM-version-of-BoN https://arxiv.org/pdf/2408.03314
  	- test-time beam search
- STAR
  - 目前，让语言模型生成推理过程（即“rationales”）的方法主要有两种：一种是构建包含推理过程的大规模数据集进行微调，这种方法成本高昂且不现实；另一种是使用少量示例（few-shot）进行上下文学习，但这种方法的性能通常远低于直接预测答案的模型。STaR技术通过迭代利用少量推理示例和大量无推理数据集，引导模型逐步提升进行更复杂推理的能力。具体来说，STaR方法包括以下几个步骤：
	    1. 使用少量推理示例引导语言模型生成多个问题的推理过程。
	    2. 对于模型生成的错误答案，通过提供正确答案来生成新的推理过程（称为“rationalization”）。
	    3. 在所有最终生成正确答案的推理上微调模型。
	    4. 重复上述过程，每次都使用改进后的模型来生成下一轮的训练数据。
- Quiter-StAR
  - 修改attn mask 并行采样 生成多个<st>（rationale）<et>，
  - 引入“混合头”（mixing head），一个浅层的多层感知机（MLP），用于生成权重，决定在给定rationale后，模型应该在多大程度上结合rationale生成的下一个标记预测概率和基础语言模型生成的概率。这个是在hidden上做的，不是文本分类器
  - RL We thus define the reward rj for each rationale Tj as the difference between p talk j:j+ntrue and the average across rationales for that token
- V-StaR
  - 现有的自我改进方法（如STaR）在训练过程中只使用正确的解决方案，而忽略了大量生成的错误解决方案。这些错误解决方案可能包含有价值的信息，有助于模型学习并改进其推理过程。
  - 那么怎么利用错误方案呢，可以使用DPO（Direct Preference Optimization）方法直接用于训练验证器
  - combines the approach of STaR  with a DPO trained verifer. At inference time, the STaR model produces several candidate reasoning chains (plans) which are ranked by the DPO verifer likelihood
- PRM-800k https://arxiv.org/abs/2305.20050
  - 第一次提出 结果监督（outcome supervision）ORM 和过程监督（process supervision）PRM概念
  - 直接人工标注了一个包含800,000个step-wise的人类反馈标签的完整数据集（PRM800K），没有开源
- MCTS 流程理解： https://mp.weixin.qq.com/s/BrLxo_p07zX6AqmGtUoVmw

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
- 预训练
	- SwitchLoRA：构造min(m,n)个候选的行向量和列向量，然后每步随机取一个插到A和B上训练，保证训练满秩
## Decoding


- Learning to Decode Collaboratively with Multiple Language Models [[paper]](https://arxiv.org/abs/2403.03870)


- Survey: Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding [[paper]](https://arxiv.org/abs/2401.07851)
- Fast Inference from Transformers via Speculative Decoding [[paper]](https://arxiv.org/pdf/2211.17192)[[repo]](https://github.com/feifeibear/LLMSpeculativeSampling) 
- Accelerating Large Language Model Decoding with Speculative Sampling [[paper]](https://arxiv.org/pdf/2302.01318)
- [ASPLOS'24] SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification [[paper]](https://arxiv.org/abs/2305.09781)
- [ICML24] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty [[paper]](https://arxiv.org/pdf/2401.15077) [[blog]](https://sites.google.com/view/eagle-llm)
- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees [[paper]](https://arxiv.org/pdf/2406.16858)
- [ICLR24] DistillSpec: Improving Speculative Decoding via Knowledge Distillation [[paper]](https://arxiv.org/abs/2310.08461) 用target model作为teacher对draft model蒸馏
- [NAACL24] REST: Retrieval-Based Speculative Decoding [[paper]](https://arxiv.org/pdf/2311.08252)
- Graph-Structured Speculative Decoding [[paper]](https://arxiv.org/pdf/2407.16207)

## Long-Context


- 相对偏置
	- ALIBI [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409), ICLR22
	  > 在Softmax之前减去一个非负矩阵，将Attention的计算从$q_{m} k_{n}$改为 $q_{m} k_{n}-\lambda|m-n|$，其中λ>0是超参数，每个head设置不同的值
- base 缩放
	- Position Interpolation (PI) [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
	  > 位置编码base（默认为10000）乘上因子$L_{train}/L_{test}$
	- Dynamic-NTK  [Reddit](www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
	  > PI的缩放是平等对待地对待所有维度，即高频旋转角度缩小的倍数和低频旋转角度缩小的倍数是一样的。 NTK-Aware Scaled RoPE 可以理解为对低频内插，高频外插
	- [YaRN: Efficient Context Window Extension of Large Language Models](https://openreview.net/forum?id=wHBfxhZu1u), ICLR 2024
	  > 1. 如果维度i对应的波长$$\lambda_i$$远小于文本长度，不进行内插只外推 2. 如果维度i对应的波长$$\lambda_i$$大于文本长度，进行内插 3. 对于中间部分，采用NTK-Aware Scaled RoPE的思路
	- [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://openreview.net/forum?id=ONOtpXLqqw), ICML 2024
	  > RoPE 的不同维度存在不均衡性, 用进化算法搜索非均匀位置插值
- 分块 / chunk
	- MRC / Long Text Match 里边有很多
	- [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
	  >  以一个固定长度划分多个窗口，每个窗口内部做 Attention, 把分窗口注意力的 Attention mask 向下移动半个窗口的长度，让上面提到的不同的窗口之间进行交互; 并且LORA增加对LayerNorm 和 Embedding 的微调
	- [LongHeads: Multi-Head Attention is Secretly a Long Context Processor](https://arxiv.org/abs/2402.10685)
	  > 认为不同的注意力头所关注的是 context 中的不同部分，将长 context 分解成多个块，让每个注意力头分别关注重要的块，并保证分配给每个注意力头的那些块包含的 tokens 数少于预训练的窗口大小
	- [Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs](https://openreview.net/forum?id=ulaUJFd96G), ICLR24
	  >  将长文本输入分割成多个小块（chunks）, 层次化合并策略，通过在不同的transformer层级逐步合并相邻的块，使得信息可以在块之间传递。
	- [Training-Free Long-Context Scaling of Large Language Models](https://arxiv.org/abs/2402.17463), ICML 2024
	  >  按块进行旋转编码（RoPE），提出一种块内，块间以及相邻块Attention的DCA策略
	- [LLM Maybe LongLM: SelfExtend LLM Context Window Without Tuning](https://openreview.net/forum?id=nkOMLBIiI7), ICML2024
	  > 利用 “grouped attention”  距离近的用原来的attention，远的话用grouped attention


## KV-Cache

- Survey: https://github.com/October2001/Awesome-KV-Cache-Compression
- KV Cache Quantization
	- Coupled Quantization (Zhang et al., 2024b). and KIVI (Zirui Liu et al., 2023), have demonstrated that KV cache can be quantized to 1-bit or 2-bit precision while preserving performance.
	- [IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact](https://arxiv.org/abs/2403.01241)
	  > Pivot tokens 作为重要的前缀，不应该进行量化  
- KV Cache Low-Rank
	- insights: 认为KV cache是低秩的
	- [Effectively Compress KV Heads for LLM](https://arxiv.org/abs/2406.07056)
	  > (1) only 25% of the highest singular values need to be retained to get most of the energy.   
	  (2) RoPE generally reduces the rank of key cache  
	  需要数据并且在激活值上进行SVD的方法称为SVD-a，直接在权重矩阵上做SVD称为SVD-w  
	- LESS: [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/abs/2402.09398)
	  > 用低秩矩阵来拟合 kv cache token dropping 带来的误差，attention map 的误差矩阵往往是低秩的，因此可以类似 Linear Attention 的做法，把 softmax 中表示相似度的指数部分，更换为分别对 qk 进行变换然后做内积  
	- [GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527)
	  > 用低秩矩阵拟合kv cache量化带来的误差，因为量化导致的误差矩阵秩比较低，可以用两个矩阵拟合  
- KV Cache Eviction
	- insights: attention本身具有的稀疏性, 50%的 KV cache贡献了0.9以上的 Attention Scores
	- Scissorhands (Liu et al., 2023b)
	  keeps a fixed KV size budget and replies on the Persistence of Importance hypothesis to evict key  
	  and value states for non-important tokens  
	- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf)，NeurIPS 2023
	  > 20%缓存接近于全量KV cache的效果;   
	  utilizes aggregated attention scores to determine so called “heavy hitters”/ important tokens  
	- StreamingLLM [Efficient Streaming Language Models with Attention Sinks](http://arxiv.org/abs/2309.17453), ICLR 2024
	  > attention sink + the recent window tokens is pivotal to maintain LLM’s performance  
	- FastGen [Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs](https://arxiv.org/abs/2310.01801)
	> attention sinks also occurs in the middle of the sentences
	认为注意力头在不同位置下的 attention map 结构是相对稳定的，所以可以通过输入的 prompt 来确定注意力头的全局模式
	- VATP [Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters](https://arxiv.org/abs/2406.12335)
	  > attention sink tokens 对应的值向量的 $l_1$ norm比其他小很多, 基于attention score 和值向量的L1范数的乘积来挑选KV cache  
	- [Keyformer: KV Cache reduction through key tokens selection for Efficient Generative Inference](https://arxiv.org/pdf/2403.09054) 
	  > H2O 改进 使用Gumbel分布引入噪声以调整未归一化logits，从而解决由于丢弃token而导致的uneven score distribution 问题。  
- KV Cache Merging （Token Merging, Token Pooling, Token Pruning）
	- insights：
		- directly eviction may accidentally and permanently remove important tokens;
		- key states exhibit high similarity at the token level within a single sequence
	- token merging is well-established in computer vision (CV)
	  (Zeng et al., 2022) (Bolya et al., 2023) (Kim et al., 2023) (Zhang et al., 2024a),  
	- [CaM: Cache Merging for Memory-efficient LLMs Inference](https://openreview.net/pdf?id=LCTmppB165), ICML 2024. 
	  > 不是直接将其需要逐出的token丢弃，而是通过merge来利用逐出的元素  
	  paper里边理论证明好处在于对attention的输出扰动更小。  
	- [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/abs/2403.09636), ICML 2024
	  > 将每次新进入的KV merge， 对于每个新来的kv，决定是merge还是append  
	- [LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference](https://openreview.net/pdf/d775ca7f5d0bfad0e56d5e710a3953555ccaabda.pdf)
	  > 应用在MLLM上 (输入是interleaved的图文对)，文本不操作，对图像token merge 因为在多模态中文本的相对图片具有更高的attention score，提出4种merge策略：Max，Mean，Pivotal ，weighted  
	- [KVMerger: Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks](https://arxiv.org/abs/2407.08454)
	  > 认为之前的工作在identity merge set上存在缺陷，发现KV cache的压缩比率在不同样本上高度一致(模型固有特性)，因此可以直接用layer-wise的cos-sim静态计算压缩比率；在LLMs的前两层和最后一层注意力得分分布更加均匀，意味着大多数键状态都很重要，应该不merging以避免引入显著的噪声； Gaussian kernel weighted merging algorithm  
- KV Cache CrossAttn
	- CLA [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)
	  > 相邻layer 共享KV cache  

## Model-Merging

- survey: https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications

## Image-Tokenizer

## Interpretable

- [ACL25] https://arxiv.org/pdf/2410.23743  正文8页但是附录有123页
	- **Nuclear Norm** (the ℓ1 norm of singular values) to represent the characteristics for the gradient of each layer $$s_{X,i} = \sum |\sigma_j|, X \in \{QKVO\}, i \in [0, N-1]$$ , N is layernum, QKVO is attention matrix
	- **mean absolute differences (MAD)** is defined as $$\mathrm{MAD}_{s_X}=\frac{1}{N-1} \sum_{i=1}^{N-1}\left|s_{X, i+1}-s_{X, i}\right|$$
	- None CoT / Simplified CoT / Detailed CoT (Slow vs. Fast Thinking)
		 - The large scale of MAD indicates that the response distributions that LLMs are going to learn have large discrepancies with what it has learned from the pretraining phase, which might harm the performances of the original pre-trained models
		- A consistent decrease in MAD is observed in all layers when LLMs are trained to produce more detailed reasoning paths (slow thinking)
		- LLMs can to some extent identify that the responses to be learned have potential conflicts
		  with their internal knowledge, thus requiring more energy to adapt to the new nonsense responses.


## Squence Parallel
- data/tensor/zero/expert/pipeline parallelism.
- Sequence Parallelism of Megatron-LM:  this form of Sequence Parallelism cannot be used independently without tensor parallelism
- [Deepspeed-Ulysses](https://arxiv.org/abs/2309.14509) P2P communication
- [Ring-Attention](https://arxiv.org/pdf/2310.01889) a distributed version of FlashAttention, All2All communication;
	- Context Parallel https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
- [USP-Attention](https://arxiv.org/pdf/2405.07719) 

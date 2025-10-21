

# **混合专家模型的架构原生无失真水印：一种可证明的AI溯源新范式**

## **第一部分：引言：人工智能内容归属的重大挑战**

随着大型语言模型（LLM）能力的飞速发展，其生成的内容在质量和复杂性上已与人类创作无异，深刻地影响着信息生态系统 1。然而，这种进步也带来了严峻的挑战，包括虚假信息的传播、学术不端行为以及知识产权的滥用 2。为了应对这些风险，为人工智能生成内容（AIGC）嵌入可验证来源信号的水印技术，已成为实现模型问责和内容溯源的关键解决方案 2。

### **1.1 水印技术的“不可能三角”**

尽管水印技术前景广阔，但现有方法普遍受困于一个固有的“不可能三角”困境，即在**鲁棒性（Robustness）**、\*\*不可感知性（Imperceptibility）**和**效率（Efficiency）\*\*这三个核心目标之间难以兼得 2。当前主流的水印技术范式及其面临的权衡困境如下：

* **词元级水印（Token-level Watermarking）**：这类方法通过在词元采样阶段引入微小偏置，例如构建一个“绿名单（green list）”，并温和地提升名单内词元的采样概率，从而嵌入水印信号 5。此类方法通常具有较高的计算效率和较低的文本质量失真。然而，其鲁棒性极差，因为水印信号与文本的表层形式（具体的词元选择和顺序）紧密耦合。攻击者只需通过简单的释义、同义词替换或使用另一个模型进行转述，就能轻易地破坏或完全移除水印信号 2。  
* **语义级水印（Semantic-level Watermarking）**：为了对抗释义攻击，语义级水印将单个句子视为基本的水印单元，将信号嵌入到句子的语义表征（embedding）中 1。例如，通过拒绝采样（rejection sampling）确保生成的句子嵌入向量落入一个预定义的“有效”语义区域 7。这种方法显著提升了对内容修改的鲁棒性。然而，它通常以牺牲不可感知性为代价。拒绝采样的过程会改变模型原始的输出概率分布，引入统计上可察觉的失真，甚至在所有候选句子都落入“无效”区域时导致采样失败，从而影响生成质量和稳定性 7。  
* **基于训练的水印（Training-based Watermarking）**：这类方法在模型训练或微调阶段将水印信息嵌入模型参数中，可以实现极高的鲁棒性 2。但是，其效率极低，需要大量的计算资源，并且不适用于已经预训练完成的、作为服务提供（API-based）的闭源模型。

### **1.2 一种综合性解决方案：AND-MoE框架**

本文旨在提出一个全新的理论框架，即**架构原生无失真混合专家水印（Architecture-Native Distortion-Free MoE Watermarking, AND-MoE）**，旨在从根本上解决上述“不可能三角”困境。该框架的核心思想是将两个前沿领域的最优特性进行深度融合：

1. **架构原生信号载体**：利用混合专家（Mixture-of-Experts, MoE）模型内部的专家路由决策作为水印信号的载体。这种方法将水印与模型的**核心计算过程**深度绑定，而非其最终生成的文本内容，从而获得对内容修改攻击的天然免疫力 10。  
2. **可证明的无失真采样**：借鉴并推广PMARK框架的数学理论，通过对模型内部状态进行中位数分割采样，确保在集成了所有可能的密钥后，加水印模型的整体输出概率分布与原始模型**完全相同**，从而在理论上实现完美的不可感知性 1。

### **1.3 核心属性与贡献**

AND-MoE框架的设计旨在满足以下三个严苛的核心属性：

* **架构原生性（Architecture-Native）**：水印的嵌入与检测完全基于模型内部的、可观测的离散状态——即在生成每个词元时所选择的专家组合。信号的载体不再是文本，而是模型的计算路径。  
* **可证明无失真（Provably Distortion-Free）**：在数学上保证，当对所有可能的密钥进行平均时，加水印模型的输出分布在统计上与未加水印的原始模型无法区分。这意味着水印的引入不会以任何方式降低文本质量或引入偏见。  
* **高密度与高鲁棒性（High Density & Robustness）**：通过在模型的多个MoE层级、每个层级的多个独立通道中嵌入信号，构建一个高维度的“水印轨迹”，确保水印证据足够密集，能够抵御包括模型微调和参数篡改在内的强力攻击。

本报告的主要贡献在于为这一新一代水印技术提供了一个全面的理论与实践蓝图。为了清晰地定位AND-MoE框架的创新性，下表系统地比较了现有主流LLM水印范式与本文提出的新框架。

**表1：LLM水印范式对比分析**

| 特性 | 词元级 (例如, Kirchenbauer et al.) | 语义级 (例如, SemStamp) | PMARK框架 | 提议的AND-MoE框架 |
| :---- | :---- | :---- | :---- | :---- |
| **信号载体** | 单个词元的选择 | 句子级别的语义嵌入 | 句子级别的语义嵌入 | **词元级别的专家路由决策（内部状态）** |
| **失真** | 低，但存在统计可察觉的偏置 | 高（基于拒绝采样） | **可证明无失真** | **可证明无失真** |
| **鲁棒性维度** | 词序、词元选择 | 语义含义 | 语义含义 | **模型的内部计算路径** |
| **主要脆弱性** | 释义攻击、同义词替换 | 语义漂移、采样失败 | 证据稀疏、对抗性代理函数 | **路由漂移（微调）、延迟开销** |

---

## **第二部分：AND-MoE框架的理论支柱**

AND-MoE框架的构建并非凭空而来，而是建立在两个坚实的理论支柱之上：混合专家（MoE）模型的独特架构特性，以及PMARK框架提供的可证明无失真采样理论。本节将对这两大支柱进行深入剖析。

### **2.1 稀疏性的计算机制：混合专家架构入门**

MoE架构是近年来实现大型语言模型参数规模有效扩展的关键技术 13。其核心思想是用一组并行的、被称为“专家”（Experts）的子网络（通常是前馈网络FFN）来替换传统Transformer模型中的密集FFN层 10。这种设计使得模型总参数量可以达到万亿级别，但在处理任何单个输入时，只有一小部分参数被激活，从而在保持推理计算成本（FLOPs）相对恒定的同时，极大地提升了模型容量 11。

#### **2.1.1 架构概览**

在MoE模型中，每个MoE层都包含两个关键组件：一组专家网络和一个门控网络（Gating Network），后者也被称为“路由器”（Router）。当一个词元（token）的隐藏状态表示 $x$ 到达MoE层时，路由器会决定将这个词元发送给哪些专家进行处理。这种条件计算（conditional computation）机制是MoE与传统密集模型的根本区别 11。包括Mixtral和Grok-1在内的前沿模型都采用了这种架构以实现卓越的性能和效率 14。

#### **2.1.2 门控网络与路由机制**

门控网络是MoE架构的“指挥中心”，其工作流程可以被精确地数学化描述：

1. **计算路由权重**：路由器本身是一个小型的神经网络，通常是一个线性层，其参数为 $W\_r$。它接收词元的隐藏状态 $x$ 作为输入，并为该层的所有 $N$ 个专家计算出一个logit向量 $h(x) \= W\_r \\cdot x$ 15。  
2. **生成概率分布**：该logit向量随后通过一个Softmax函数进行归一化，生成一个概率分布 $p(x)$，其中第 $i$ 个元素 $p\_i(x)$ 代表将词元 $x$ 发送给第 $i$ 个专家的概率或“权重” 19。  
3. **Top-K专家选择**：路由器根据这个概率分布选择得分最高的 $K$ 个专家来处理该词元。例如，Switch Transformer为了追求极致的效率，采用了Top-1的路由策略 19。最终，该层的输出是这 $K$ 个专家输出的加权和。

#### **2.1.3 专家特化与路由模式**

MoE模型在训练过程中会自发地涌现出专家特化（expert specialization）的现象。不同的专家会学习处理不同类型的数据、语言模式或任务领域，从而实现功能上的多样性 11。研究表明，这种路由模式是结构化且可解释的。例如，在多语言模型中，位于网络早期和晚期的MoE层倾向于展现出与特定语言相关的路由模式，而中间层则处理更为抽象和跨语言的语义概念 10。

正是专家选择集 $S\_x \= \\text{TopK}(x)$ 的这种**可观测、离散且具有结构化语义**的特性，使其成为一个理想的、内在于模型计算过程的信号载体。与依赖于最终文本输出的水印不同，基于路由决策的水印直接利用了模型“思考过程”中的一个中间步骤，这为实现架构原生的鲁棒性奠定了基础。

### **2.2 不可感知性的理论：PMARK的可证明无失真水印**

PMARK框架为解决水印导致的文本质量下降问题提供了一个严谨的数学解决方案 1。其核心贡献在于设计了一种采样算法，该算法在理论上保证了加水印模型的平均输出分布与原始模型完全一致，即“可证明无失真”。

#### **2.2.1 代理函数（Proxy Function）**

PMARK理论的核心是\*\*代理函数（Proxy Function, PF）\**的概念。代理函数 $F: \\Sigma^* \\to \\mathbb{R}$ 是一个将任意一个完整的句子 $s$ 映射到一个实数值的函数。在PMARK的典型实现中，这个函数被定义为句子嵌入向量 $\\mathcal{T}(s)$ 与一个作为密钥的秘密随机向量 $v$ 之间的余弦相似度：$F(s) \= \\langle v, \\mathcal{T}(s) \\rangle$ 1。这个标量值可以被看作是句子在由密钥 $v$ 定义的某个语义维度上的“投影”。

#### **2.2.2 中位数分割采样算法**

PMARK的无失真特性是通过一种精巧的\*\*中位数分割采样（median-splitting sampling）\*\*算法实现的。其具体步骤如下：

1. **生成候选集**：在给定的上下文之后，模型首先生成一个包含多个高质量候选句子的集合。  
2. **评估与分割**：对每个候选句子应用代理函数 $F(s)$，计算出其得分。然后，找到所有这些得分的中位数（median）。根据这个中位数，将候选句子集合划分为得分高于中位数的“上半区”和得分低于或等于中位数的“下半区”。  
3. **密钥选择与采样**：使用一个秘密的二进制密钥位（例如，0或1）来决定是从“上半区”还是“下半区”进行最终的采样。一旦确定了采样区域，该区域内所有候选句子的原始概率将被重新归一化（renormalized），然后根据这个新的分布进行采样。

#### **2.2.3 数学保证**

该算法的无失真特性具有严格的数学证明。其逻辑可以简化如下：假设密钥空间 $\\mathcal{K}$ 由一个随机比特 $k \\in \\{0, 1\\}$ 构成，且 $P(k=0) \= P(k=1) \= 0.5$。对于任何一个给定的句子 $s$，其在加水印模型下的最终生成概率 $P\_{watermarked}(s)$ 是其在两个密钥选择下的概率的期望值 7。

* 假设句子 $s$ 的代理函数得分落在了由中位数定义的“下半区”。  
* 当密钥 $k=0$ 时（选择“下半区”），$s$ 的采样概率被重新归一化为 $P(s|k=0) \= \\frac{P\_{original}(s)}{P(\\text{下半区})} \= \\frac{P\_{original}(s)}{0.5}$。  
* 当密钥 $k=1$ 时（选择“上半区”），$s$ 不在采样空间内，因此其采样概率为 $P(s|k=1) \= 0$。  
* 对所有密钥求平均，我们得到：

  $$P\_{watermarked}(s) \= \\sum\_{k \\in \\{0,1\\}} P(k) \\cdot P(s|k) \= 0.5 \\cdot \\frac{P\_{original}(s)}{0.5} \+ 0.5 \\cdot 0 \= P\_{original}(s)$$

这个结论对于落在“上半区”的句子同样成立。因此，对于任何句子，其在加水印模型下的期望概率都与原始模型完全相同。AND-MoE框架的核心理论创新，正是要将这一强大的数学保证从外部的、对生成结果的约束，内化到模型生成每一个词元的内部计算过程中。

这种理论上的转变，从对连续、高维语义空间（句子嵌入）的操作，转向对离散、低维组合空间（专家索引集）的操作，是AND-MoE框架的基石。PMARK的代理函数依赖于高维向量空间的几何特性，例如随机向量之间近似正交 7。然而，一个专家索引集，如 $\\{5, 23\\}$，并不具备向量空间的结构。因此，为AND-MoE设计一个合适的代理函数 $\\mathcal{F}\_{MoE}$，使其能够在离散集合上产生良好分布的伪随机分值，是实现这一理论迁移的核心挑战，而不仅仅是一个简单的实现细节。

---

## **第三部分：AND-MoE框架：设计与实现**

本部分将详细阐述AND-MoE框架的具体设计和实现方案。我们将从其核心创新点——代理函数的内化——出发，逐步介绍实现该框架所需的三大关键技术，并对每个技术的实现要点和潜在挑战进行深入分析。

### **3.1 核心创新：代理函数的内化**

AND-MoE框架的根本性突破在于将水印约束从对\*\*“生成结果”**的后验评估，前移到对**“生成过程”\*\*的即时干预。我们不再等待一个完整的句子生成后去评估其语义嵌入，而是在生成每一个词元（token）的瞬间，利用模型内部的计算状态作为水印的载体。

具体而言，我们将代理函数的定义域从句子空间 $\\Sigma^\*$ 迁移到专家索引组合空间。在模型生成下一个词元时，对于任意一个潜在的候选词元，我们都可以预测它将激活的专家组合。我们将这个组合，即在第 $l$ 个MoE层为词元表示 $x$ 选择的专家索引集合，定义为模型的内部状态：$S\_{x, l} \= \\text{TopK}(x, l)$。

基于此，我们重新定义代理函数为 $\\mathcal{F}\_{MoE}(S\_{x,l}; k)$，它接收一个离散的专家索引集合 $S\_{x,l}$ 和一个秘密密钥 $k$ 作为输入，并输出一个实数值。这个过程将PMARK的无失真采样理论 7 完美地嵌入到MoE架构的单步词元生成循环中，从而在理论上实现了架构原生性与无失真性的统一。

### **3.2 关键技术一：动态路由候选集的构建与评估**

为了在生成过程中应用中位数分割，我们必须首先为模型下一步可能生成的多个词元构建一个“候选集”，并评估每个候选词元对应的内部状态（即专家路由组合）。

#### **3.2.1 “预计算（Dry Run）”机制**

该机制的算法流程如下：

1. **生成候选词元集**：在解码的每一步，模型首先根据其最终的logit层计算出整个词汇表的概率分布。我们不直接从中采样，而是选出概率最高的 $N$ 个词元，构成候选集 $\\{t\_1, t\_2,..., t\_N\\}$。  
2. **并行预计算路由**：对于候选集中的每一个词元 $t\_i$，我们将其输入嵌入（embedding）传入模型的所有MoE层，执行一次“预计算”或“试运行”（dry run）。这次计算只涉及门控网络，用于确定如果选择该词元，它将在每一层触发的专家组合 $S\_{t\_i, l}$。  
3. **构建评估对**：经过预计算，我们得到一个包含 $N$ 个评估对的集合：$\\{(t\_1, \\{S\_{t\_1, l}\\}\_{l=1}^{L}), (t\_2, \\{S\_{t\_2, l}\\}\_{l=1}^{L}),..., (t\_N, \\{S\_{t\_N, l}\\}\_{l=1}^{L})\\}$，其中 $L$ 是模型中MoE层的总数。这个集合构成了后续水印嵌入操作的基础。

#### **3.2.2 可行性与延迟分析**

这一步骤是AND-MoE框架在实际应用中面临的最大挑战，因为它直接关系到生成效率。门控网络本身的计算，以及在分布式系统中可能涉及的All-to-All通信，是MoE模型推理延迟的重要组成部分 23。我们的“预计算”机制将这一开销放大了 $N$ 倍。

这是一个根本性的矛盾：MoE架构的设计初衷是为了通过稀疏激活来提升计算效率 16，而我们的水印方案却引入了额外的密集计算步骤。这不仅仅是一个性能优化问题，而是关乎整个框架实用性的核心瓶颈。

为了缓解这一问题，必须采取高效的实现策略。幸运的是，现代深度学习框架和硬件（如GPU）非常擅长并行处理。门控网络的计算本质上是一系列矩阵乘法，可以对 $N$ 个候选词元的嵌入进行批量处理（batching），从而在一次或几次集中的计算中得出所有候选词元的路由决策。尽管总计算量增加了，但通过并行化，实际增加的“墙上时间”（wall-clock time）可以被有效控制。最终的可行性取决于 $N$ 的大小、模型中MoE层的数量、硬件的并行计算能力以及对生成延迟的容忍度之间的权衡。例如，可以探索使用一个计算成本更低的、蒸馏过的“代理路由器”来近似预测路由决策，或者在保证统计有效性的前提下，尽可能选择一个较小的 $N$ 值。

### **3.3 关键技术二：可组合的、带密钥的路由代理函数**

代理函数 $\\mathcal{F}\_{MoE}(S; k)$ 是连接内部状态和无失真采样的桥梁。其设计必须满足一系列严格的数学和工程要求。

#### **3.3.1 设计要求**

一个合格的路由代理函数需要具备以下特性：

1. **确定性（Deterministic）**：对于相同的输入 $(S, k)$，必须产生相同的输出。  
2. **高效性（Efficient）**：计算速度必须极快，以避免在解码循环中引入显著延迟。  
3. **均匀分布性（Uniform Distribution）**：对于随机的输入，其输出值应近似均匀分布。这是中位数分割能够有效将候选集划分为两个概率质量相等的子集的统计学基础 26。  
4. **顺序无关性（Order-Invariant）**：函数的输出不应依赖于专家索引在集合 $S$ 中的顺序，即 $\\mathcal{F}(\\{i\_1, i\_2,..., i\_k\\}; k) \= \\mathcal{F}(\\{i\_2, i\_1,..., i\_k\\}; k)$ 29。  
5. **密钥依赖与抗碰撞性（Key-Dependent & Collision-Resistant）**：在不知道密钥 $k$ 的情况下，从输出值反推输入集合 $S$ 应该是计算上不可行的。不同的密钥应产生完全不同的、不可预测的映射。

#### **3.3.2 候选函数族**

多种函数设计方案可以满足上述要求，它们在安全性、速度和实现复杂度上各有取舍。

* **加密哈希函数（如HMAC-SHA256）**：提供最高的安全保证和优秀的统计特性。HMAC结构通过引入密钥来防止标准哈希函数的长度扩展等攻击 31。然而，其计算成本相对较高，对于需要每步解码都调用的场景，可能会成为性能瓶颈 33。  
* **非加密哈希函数（如SipHash, MurmurHash）**：这类函数为性能而生，计算速度极快 33。特别是SipHash，它被设计用来抵御哈希表碰撞攻击（Hash Flooding DoS attacks），因此具备一定的抗恶意输入攻击的能力，使其成为一个在速度和安全性之间取得良好平衡的有力候选者 31。  
* **带密钥的随机投影（Keyed Random Projection）**：这是一种理论上非常优雅的方法。首先，将专家索引集合 $S \= \\{i\_1,..., i\_k\\}$ 表示为一个高维稀疏二元向量 $v\_S$（在对应专家索引的位置为1，其余为0）。然后，将这个向量投影到一个由密钥生成的随机向量 $v\_k$ 上，得到标量得分 $\\mathcal{F}(S; k) \= \\langle v\_S, v\_k \\rangle$。这种方法的统计特性（如分布的正态性）得益于Johnson-Lindenstrauss引理等理论支持，并且计算上非常高效（本质上是几次加法）35。这种思想类似于用户查询中提到的CES（Contrastive Expert Selection）方法。

下表对这些候选函数进行了系统性的比较，为工程实现提供了决策依据。

**表2：带密钥路由代理函数 ($\\mathcal{F}\_{MoE}$) 的设计权衡**

| 函数族 | 计算成本 | 加密安全性 | 抗碰撞性 | 对AND-MoE的适用性 |
| :---- | :---- | :---- | :---- | :---- |
| **HMAC-SHA256** | 高 | 非常高 | 非常高 | 提供最大程度的安全性，但可能在“预计算”循环中引入不可接受的延迟。可能存在性能过剩问题 31。 |
| **SipHash** | 非常低 | 高（抗DoS攻击） | 高 | 极佳的平衡选择。专为哈希表等性能关键应用设计，同时能抵御恶意攻击，非常适合本框架的需求 31。 |
| **MurmurHash** | 非常低 | 无 | 良好（非对抗场景） | 速度最快，但如果攻击者能影响输入以产生哈希碰撞，则可能存在安全风险 33。 |
| **带密钥的随机投影** | 低 | 中到高 | 概率性高 | 理论上优雅。其安全性和分布特性在数学上有充分的理解和保证，是一种非常有前景的方案 35。 |

### **3.4 关键技术三：多通道与多层级的约束叠加**

为了构建一个足够密集、能够抵御强力攻击的水印，单一的约束是远远不够的。AND-MoE框架通过“叠加”多个独立的无失真水印约束来极大地增强信号强度和鲁棒性。

#### **3.4.1 横向叠加（多通道）**

在单个MoE层内部，我们可以借鉴PMARK的多通道设计 1。具体做法是，我们不使用单个密钥 $k$，而是生成一组在某种意义上“正交”的密钥 $\\{k\_1, k\_2,..., k\_b\\}$。这些密钥定义了 $b$ 个独立的代理函数 $\\mathcal{F}\_1, \\mathcal{F}\_2,..., \\mathcal{F}\_b$。

在进行词元选择时，候选集将依次通过这 $b$ 个“过滤器”。首先，使用 $\\mathcal{F}\_1$ 和密钥位 $c\_1$ 对 $N$ 个候选词元进行第一次中位数分割，将候选范围缩小到约 $N/2$。然后，对这 $N/2$ 个幸存的候选词元，再使用 $\\mathcal{F}\_2$ 和密钥位 $c\_2$ 进行第二次中位数分割，范围进一步缩小到约 $N/4$，以此类推。这个过程将一个 $b$ 比特的秘密信息嵌入到了单次词元生成的选择中，极大地增强了单步水印的证据强度。

#### **3.4.2 纵向叠加（多层级）**

现代LLM通常包含数十个MoE层。AND-MoE框架将模型中的**每一个MoE层**都视为一个独立的“宏通道”。在生成每个词元时，上述的多通道水印嵌入过程会在模型的每一层（或部分关键层）独立进行。

这将水印信号从一个单点信息，扩展成了一个贯穿模型计算深度的\*\*“水印轨迹”**或**“路由签名”\*\*。一个词元的水印证据不再是一个比特或一个分数，而是一个维度为（层数 $\\times$ 通道数）的证据矩阵。一个长度为 $T$ 的文本序列，其水印证据将构成一个三维张量。这种高维度的信号结构使得水印异常密集和丰富，为抵御攻击提供了前所未有的可能性。攻击者若想抹除水印，必须找到一段语义相近的文本，其在生成过程中产生的整个高维轨迹张量都与原始水印不匹配，且同时看起来又像是随机噪声，这在计算上是一个极其困难的挑战。

#### **3.4.3 信号聚合与检测**

由于水印信号由大量微弱的统计偏差构成 38，检测算法必须能够有效地聚合这些信号。检测过程不再是简单的计数或二元判断，而是收集每个决策点的“软信息”（soft information）。

例如，对于在第 $l$ 层、第 $j$ 个通道做出的选择，我们可以计算所选词元的代理函数得分与该轮分割中位数之间的归一化距离。这个距离本身就是一个带有符号的统计量。检测时，我们将收集到的、来自所有词元、所有层、所有通道的成千上万个这样的统计量。

**加权Z检验（Weighted Z-test）**，有时也被称为soft-z-test，是聚合这些独立证据的理想统计工具 40。该方法可以将来自不同来源（此处即不同层和通道）的Z分数，根据其各自的置信度或预期信号强度（权重）进行加权组合，最终计算出一个总体的、具有极高统计显著性的Z分数和p值。这使得AND-MoE框架能够从极短的文本片段中（例如几十个词元）可靠地检测出水印的存在。

---

## **第四部分：鲁棒性与安全性深度分析**

一个水印框架的最终价值取决于其在真实世界对抗环境下的生存能力。AND-MoE框架通过其独特的设计，在理论上对两类主要攻击——内容修改攻击和模型篡改攻击——展现出强大的防御潜力。

### **4.1 对释义和内容修改的天然免疫力**

传统水印方法的主要弱点在于其信号载体是文本本身或其语义表示 2。因此，任何保留语义但改变文本形式的攻击，如释义、翻译、摘要等，都可能破坏或移除水印。

AND-MoE框架从根本上规避了这个问题。它的水印信号并非嵌入在\*\*“生成了什么内容”**之中，而是嵌入在**“模型是如何生成这些内容的”\*\*这一过程中。水印的证据是专家路由决策序列，这是一个与模型内部计算图紧密绑定的“元数据”。

当攻击者使用另一个模型（无论是否加水印）对AND-MoE生成的内容进行释义时，新的文本是通过一个完全不同的模型、遵循一条完全不同的内部计算路径生成的。因此，新文本中自然不会包含原始模型的专家路由签名。检测器在分析这段释义文本时，会发现其内部生成逻辑与任何已知的AND-MoE密钥都不匹配，从而得出“非该模型生成”的结论。这不仅是水印的检测，更是一种强有力的\*\*来源归属（provenance）\*\*证明。因为只有原始的、持有密钥的水印模型，才能生成同时满足内容和内部计算路径双重约束的文本。

### **4.2 抵御模型篡改与微调攻击**

将水印与模型架构深度绑定，既是其力量的源泉，也带来了新的、针对架构本身的攻击面。其中，最严峻的威胁来自于模型的微调（fine-tuning）。

#### **4.2.1 路由漂移（Routing Drift）的威胁**

微调是根据特定任务或数据集调整预训练模型参数的常见且合法的操作 42。然而，研究表明，即使是良性的微调，也可能导致MoE模型内部的\*\*“路由漂移”\*\*——即模型学会了用与预训练阶段不同的方式将输入路由到专家 45。

这种现象为攻击者提供了可乘之机。一个恶意的攻击者可以通过精心设计的微调任务，系统性地改变模型的路由行为，从而达到破坏或篡改水印的目的。例如：

* **水印擦除攻击**：通过微调，使模型在生成文本时，其路由决策变得更加随机，从而破坏水印信号的统计规律。  
* **水印伪造攻击**：通过微调，强迫模型遵循一个攻击者指定的、与另一个合法密钥对应的路由模式，从而实现水印的“嫁祸”。

这种攻击的危险性在于，它利用了模型生命周期中一个完全正常且必要的操作。因此，任何防御措施都不能简单地禁止微调。

#### **4.2.2 对策：基于正则化的路由稳定性增强**

应对路由漂移威胁的思路，是在微调过程中引入额外的约束，以“锚定”模型的路由行为。借鉴在MoE模型安全性研究中的成果，例如SafeMoE框架 46，我们可以设计一种\*\*路由正则化（Routing Regularization）\*\*机制。

该机制的核心思想是，在微调的损失函数中增加一个正则化项。这个正则化项的作用是惩罚微调后模型与原始模型在路由决策上的差异。具体实现上，可以：

1. 维护一个私有的、包含多种类型输入的“金丝雀（canary）”提示集合。  
2. 在微调的每个步骤，将这些金丝雀提示输入到当前正在微调的模型和原始的、未修改的模型中，分别获取它们在每个MoE层产生的专家路由概率分布。  
3. 计算这两个概率分布之间的KL散度（Kullback-Leibler divergence），并将其作为一个惩罚项加入到主任务的损失函数中。  
   $$ \\mathcal{L}{total} \= \\mathcal{L}{task} \+ \\lambda \\cdot \\sum\_{l=1}^{L} D\_{KL}(P\_{original}(S\_l | \\text{canary}) |

| P\_{finetuned}(S\_l | \\text{canary})) $$  
其中 λ 是一个超参数，用于平衡任务性能和路由稳定性。  
通过这种方式，我们可以在允许模型为新任务学习新知识的同时，强制其保留在水印相关的基础路由模式上的“记忆”，从而有效保护水印的完整性。这表明，AND-MoE水印的部署，必须与一个配套的\*\*“安全微调协议”\*\*紧密结合，两者共同构成了模型全生命周期的可信保障体系。

#### **4.2.3 对专家篡改的鲁棒性**

AND-MoE框架的高密度特性为其抵御直接的模型参数篡改（如移除或修改少数专家）提供了天然的鲁棒性。由于水印信号分布在模型的所有MoE层和每个层的多个通道中，攻击者即使成功破坏了少数几个MoE层的信号，来自其余数十个完好层级的海量统计证据，仍然足以让检测算法做出高置信度的判断 40。信号的冗余和分布式特性使得单点故障几乎不可能摧毁整个水印系统。

---

## **第五部分：结论：迈向可信人工智能的新纪元**

本文提出的架构原生无失真混合专家水印（AND-MoE）框架，是对当前人工智能内容归属领域核心挑战的一次系统性回应。通过将MoE架构的内部计算特性与PMARK框架的无失真理论进行创新性融合，AND-MoE为解决长期困扰水印技术的“不可能三角”困境提供了一条清晰且可行的路径。

### **5.1 破解水印技术的三难困境**

AND-MoE框架通过其精巧的设计，协同实现了三大核心目标，展示了其突破性的潜力：

* **鲁棒性**：通过将水印信号的载体从易变的文本内容转移到模型的内在计算路径——专家路由决策，AND-MoE获得了对释义、翻译等内容修改攻击的根本性免疫力。其多层级、多通道的信号叠加机制构建了一个高维度的水印轨迹，使得针对模型篡改的攻击也变得异常困难。  
* **不可感知性**：通过严格遵循并内化PMARK框架的中位数分割采样理论，AND-MoE在数学上保证了其在所有密钥下的平均输出分布与原始模型完全一致。这一“可证明无失真”的特性，确保了水印的引入在统计上是“隐形”的，不会对模型的性能、流畅度或客观性产生任何负面影响。  
* **效率**：尽管“预计算”步骤带来了额外的计算开销，对延迟构成了挑战，但该框架的核心计算成本仍然受益于MoE架构的稀疏激活特性，避免了与密集模型同等参数规模下的指数级计算增长。通过高效的并行化批处理，这一延迟开销有望被控制在可接受的范围内，从而在实践中保持了较高的效率。

### **5.2 开放挑战与未来研究方向**

尽管AND-MoE框架在理论上展现出巨大潜力，但将其从理论蓝图转化为可广泛部署的工业级技术，仍面临一系列开放性挑战，这也为未来的研究指明了方向：

* **延迟开销的极致优化**：“预计算”机制是当前框架实用性的主要瓶颈。未来的研究可以探索更先进的延迟优化技术，例如：使用轻量级的、蒸馏出的“代理路由器”来近似预测路由决策；将水印嵌入与推测式解码（speculative decoding）等高效推理技术相结合；或者研究非对称机制，即在嵌入时付出较高代价，但在检测时实现极速验证。  
* **自适应代理函数**：当前的代理函数 $\\mathcal{F}\_{MoE}$ 是静态的。可以研究设计动态的、上下文感知的代理函数，使其能够根据不同的层级（例如，对处理抽象概念的中间层赋予更高权重）或不同的输入内容动态调整，从而进一步提升水印信号的质量和鲁棒性。  
* **鲁棒性的理论边界**：在信息论和密码学的框架下，对AND-MoE的多维水印轨迹进行严格的理论分析，量化其在面对拥有模型访问权限的最优攻击者时的鲁棒性边界。这将为水印的安全性提供可量化的保证。  
* **框架的泛化与扩展**：探索将“对内部计算选择进行水印”这一核心思想推广到MoE之外的其他条件计算架构，例如具有动态网络结构或自适应计算深度的模型。这将为更广泛的人工智能系统提供可信的来源追溯能力。

综上所述，AND-MoE框架不仅是一种具体的水印技术方案，更代表了一种全新的设计哲学：将内容的来源证明根植于模型的“思维过程”本身。若能成功实现，它将为构建一个更加透明、负责和可信的人工智能生态系统奠定坚实的技术基础。

#### **Works cited**

1. PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints \- ResearchGate, accessed October 20, 2025, [https://www.researchgate.net/publication/395848683\_PMark\_Towards\_Robust\_and\_Distortion-free\_Semantic-level\_Watermarking\_with\_Channel\_Constraints](https://www.researchgate.net/publication/395848683_PMark_Towards_Robust_and_Distortion-free_Semantic-level_Watermarking_with_Channel_Constraints)  
2. Watermarking for Large Language Models: A Survey \- ResearchGate, accessed October 20, 2025, [https://www.researchgate.net/publication/391257786\_Watermarking\_for\_Large\_Language\_Models\_A\_Survey](https://www.researchgate.net/publication/391257786_Watermarking_for_Large_Language_Models_A_Survey)  
3. Watermarking for Large Language Models: A Survey \- MDPI, accessed October 20, 2025, [https://www.mdpi.com/2227-7390/13/9/1420](https://www.mdpi.com/2227-7390/13/9/1420)  
4. Watermarking Techniques for Large Language Models: A Survey \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2409.00089v1](https://arxiv.org/html/2409.00089v1)  
5. \[2301.10226\] A Watermark for Large Language Models \- arXiv, accessed October 20, 2025, [https://arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226)  
6. A Watermark for Large Language Models \- arXiv, accessed October 20, 2025, [https://arxiv.org/pdf/2301.10226](https://arxiv.org/pdf/2301.10226)  
7. PMARK: TOWARDS ROBUST AND DISTORTION ... \- OpenReview, accessed October 20, 2025, [https://openreview.net/pdf/9ca3f86b48f3477620489f6152d6428d3446bf6f.pdf](https://openreview.net/pdf/9ca3f86b48f3477620489f6152d6428d3446bf6f.pdf)  
8. \[2509.21057\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints \- arXiv, accessed October 20, 2025, [https://arxiv.org/abs/2509.21057](https://arxiv.org/abs/2509.21057)  
9. A Survey of Text Watermarking in the Era of Large Language Models \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2312.07913v4](https://arxiv.org/html/2312.07913v4)  
10. Multilingual Routing in Mixture-of-Experts \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2510.04694v1](https://arxiv.org/html/2510.04694v1)  
11. Mixture of Experts in Large Language Models †: Corresponding author \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2507.11181v1](https://arxiv.org/html/2507.11181v1)  
12. PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2509.21057v1](https://arxiv.org/html/2509.21057v1)  
13. \[2507.11181\] Mixture of Experts in Large Language Models \- arXiv, accessed October 20, 2025, [https://arxiv.org/abs/2507.11181](https://arxiv.org/abs/2507.11181)  
14. Mixture of Experts for Faster, More Efficient LLMs | by Hamza El Fergougui | Medium, accessed October 20, 2025, [https://medium.com/@hamzafergougui/mixture-of-experts-for-faster-more-efficient-llms-426351d942e6](https://medium.com/@hamzafergougui/mixture-of-experts-for-faster-more-efficient-llms-426351d942e6)  
15. Mixture of experts \- Wikipedia, accessed October 20, 2025, [https://en.wikipedia.org/wiki/Mixture\_of\_experts](https://en.wikipedia.org/wiki/Mixture_of_experts)  
16. The Rise of Mixture-of-Experts for Efficient Large Language Models \- Unite.AI, accessed October 20, 2025, [https://www.unite.ai/the-rise-of-mixture-of-experts-for-efficient-large-language-models/](https://www.unite.ai/the-rise-of-mixture-of-experts-for-efficient-large-language-models/)  
17. Scaling to Trillion Parameter Models With Switch Transformers | by Zia Babar \- Medium, accessed October 20, 2025, [https://medium.com/@zbabar/scaling-to-trillion-parameter-models-with-switch-transformers-88ca5fb95e5c](https://medium.com/@zbabar/scaling-to-trillion-parameter-models-with-switch-transformers-88ca5fb95e5c)  
18. What is mixture of experts? | IBM, accessed October 20, 2025, [https://www.ibm.com/think/topics/mixture-of-experts](https://www.ibm.com/think/topics/mixture-of-experts)  
19. Switch Transformers: Scaling to Trillion Parameter Models with ..., accessed October 20, 2025, [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)  
20. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity – Related Work – Interesting papers \- Alastair Reid, accessed October 20, 2025, [https://alastairreid.github.io/RelatedWork/papers/fedus:arxiv:2021/](https://alastairreid.github.io/RelatedWork/papers/fedus:arxiv:2021/)  
21. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, accessed October 20, 2025, [https://www.researchgate.net/publication/348403003\_Switch\_Transformers\_Scaling\_to\_Trillion\_Parameter\_Models\_with\_Simple\_and\_Efficient\_Sparsity](https://www.researchgate.net/publication/348403003_Switch_Transformers_Scaling_to_Trillion_Parameter_Models_with_Simple_and_Efficient_Sparsity)  
22. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity \- Journal of Machine Learning Research, accessed October 20, 2025, [https://jmlr.org/papers/volume23/21-0998/21-0998.pdf](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)  
23. Toward Efficient Inference for Mixture of Experts \- University of Pennsylvania, accessed October 20, 2025, [https://www.seas.upenn.edu/\~leebcc/documents/huang24-neurips.pdf](https://www.seas.upenn.edu/~leebcc/documents/huang24-neurips.pdf)  
24. Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference \- SciSpace, accessed October 20, 2025, [https://scispace.com/pdf/towards-moe-deployment-mitigating-inefficiencies-in-mixture-362iljkj.pdf](https://scispace.com/pdf/towards-moe-deployment-mitigating-inefficiencies-in-mixture-362iljkj.pdf)  
25. Applying Mixture of Experts in LLM Architectures | NVIDIA Technical Blog, accessed October 20, 2025, [https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)  
26. CS 312 Lecture 21 Hash functions, accessed October 20, 2025, [https://www.cs.cornell.edu/courses/cs312/2008sp/lectures/lec21.html](https://www.cs.cornell.edu/courses/cs312/2008sp/lectures/lec21.html)  
27. What are Hash Functions and How to choose a good Hash Function? \- GeeksforGeeks, accessed October 20, 2025, [https://www.geeksforgeeks.org/dsa/what-are-hash-functions-and-how-to-choose-a-good-hash-function/](https://www.geeksforgeeks.org/dsa/what-are-hash-functions-and-how-to-choose-a-good-hash-function/)  
28. Hash table \- Wikipedia, accessed October 20, 2025, [https://en.wikipedia.org/wiki/Hash\_table](https://en.wikipedia.org/wiki/Hash_table)  
29. Hash function \- Wikipedia, accessed October 20, 2025, [https://en.wikipedia.org/wiki/Hash\_function](https://en.wikipedia.org/wiki/Hash_function)  
30. Order-invariant hash in Python \- Stack Overflow, accessed October 20, 2025, [https://stackoverflow.com/questions/42306656/order-invariant-hash-in-python](https://stackoverflow.com/questions/42306656/order-invariant-hash-in-python)  
31. HMAC-SHA256 vs SipHash \- A Comprehensive Comparison \- MojoAuth, accessed October 20, 2025, [https://mojoauth.com/compare-hashing-algorithms/hmac-sha256-vs-siphash/](https://mojoauth.com/compare-hashing-algorithms/hmac-sha256-vs-siphash/)  
32. What's the difference between HMAC-SHA256(key, data) and SHA256(key \+ data), accessed October 20, 2025, [https://security.stackexchange.com/questions/79577/whats-the-difference-between-hmac-sha256key-data-and-sha256key-data](https://security.stackexchange.com/questions/79577/whats-the-difference-between-hmac-sha256key-data-and-sha256key-data)  
33. HMAC-SHA256 vs MurmurHash \- SSOJet, accessed October 20, 2025, [https://ssojet.com/compare-hashing-algorithms/hmac-sha256-vs-murmurhash/](https://ssojet.com/compare-hashing-algorithms/hmac-sha256-vs-murmurhash/)  
34. What is difference between HMAC-SHA256 vs MurmurHash \- Compile7, accessed October 20, 2025, [https://compile7.org/compare-hashing-algorithms/what-is-difference-between-hmac-sha256-vs-murmurhash/](https://compile7.org/compare-hashing-algorithms/what-is-difference-between-hmac-sha256-vs-murmurhash/)  
35. Random projections of linear and semidefinite problems with linear inequalities \- arXiv, accessed October 20, 2025, [https://arxiv.org/pdf/2007.00242](https://arxiv.org/pdf/2007.00242)  
36. Random projection \- Wikipedia, accessed October 20, 2025, [https://en.wikipedia.org/wiki/Random\_projection](https://en.wikipedia.org/wiki/Random_projection)  
37. 7.6. Random Projection — scikit-learn 1.7.2 documentation, accessed October 20, 2025, [https://scikit-learn.org/stable/modules/random\_projection.html](https://scikit-learn.org/stable/modules/random_projection.html)  
38. An Ensemble Framework for Unbiased Language Model Watermarking \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2509.24043v1](https://arxiv.org/html/2509.24043v1)  
39. A Watermarking Method Based on Optimization Statistics \- ResearchGate, accessed October 20, 2025, [https://www.researchgate.net/publication/274477457\_A\_Watermarking\_Method\_Based\_on\_Optimization\_Statistics](https://www.researchgate.net/publication/274477457_A_Watermarking_Method_Based_on_Optimization_Statistics)  
40. Optimally weighted Z-test is a powerful method for combining probabilities in meta-analysis \- PMC, accessed October 20, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3135688/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3135688/)  
41. Combining information \- PMC, accessed October 20, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2898213/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2898213/)  
42. A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2503.07137v1](https://arxiv.org/html/2503.07137v1)  
43. Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2508.02587v1](https://arxiv.org/html/2508.02587v1)  
44. Communication-Efficient MoE Fine-Tuning with Locality-Aware Expert Placement \- iQua Group, accessed October 20, 2025, [https://iqua.ece.toronto.edu/papers/chenghao-icdcs25.pdf](https://iqua.ece.toronto.edu/papers/chenghao-icdcs25.pdf)  
45. Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2510.14853v1](https://arxiv.org/html/2510.14853v1)  
46. Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2509.22745v1](https://arxiv.org/html/2509.22745v1)  
47. Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment \- arXiv, accessed October 20, 2025, [https://www.arxiv.org/pdf/2509.22745](https://www.arxiv.org/pdf/2509.22745)  
48. AN ENSEMBLE FRAMEWORK FOR UNBIASED LAN- GUAGE MODEL WATERMARKING \- OpenReview, accessed October 20, 2025, [https://openreview.net/pdf/d1d3bf6c2378ca90707046110b52d184d7f87a2b.pdf](https://openreview.net/pdf/d1d3bf6c2378ca90707046110b52d184d7f87a2b.pdf)
# SE-VLM3DQA — Design 1

## Video-only 对抗证据评价器与完整后训练实验方案

**版本：** 1.1 · 2026-09-25  
**交付对象：** Shi 与 Codex  
**当前目标：** 完整执行 **pre-supervise → train → test**，评价新 evaluator 产生的监督能否改善学生的 3D 视频质量评价能力。  
**输入设定：** 已有 synthetic / 3D-rendered RGB videos；合成过程信息与额外 3D 信息均置于当前算法输入边界之外。  
**文档层级：** 固定研究问题、完整顶层架构、阶段接口、实验对照、scaling variables 和结果要求。模型规模、数据量、agent 数量、具体公式、工程实现与资源配置由 Shi 和 Codex 确定。  
**证据约定：** `[R#]` 指文末的论文、作者资源或官方文档；模块组合、推荐优先级与预期作用属于设计建议。本文是一份设计稿，尚未执行下载、训练或评测。

> **给 Codex 的执行约定**  
> 本任务包含学生后训练和独立测试。Evaluator 独立评价属于同一完整实验中的诊断模块。资源不足时缩小模型、数据或 agent 数量，仍保留至少一组可解释的端到端对照。本文出现的数值集合仅作为 scaling 候选点，实际运行前与 Shi 确认。检查现有仓库、模型与数据的真实可用性后，形成运行配置；任何改变输入权限、监督来源或研究问题的选择，先向 Shi 澄清。

---

# 1. 背景与目的

## 1.1 项目背景

根据 Shi 转述的博士生说明，项目希望使用其他来源的 synthetic 3D videos 做 VLM post-training，最终在有人类主观分数的 3D 质量基准上测试泛化。博士生报告已有接近现有方法水平的初步结果；其真实 pipeline、比较协议与实现成熟度需要由现有代码和实验记录确认。

本稿固定如下学习任务：

```text
已渲染好的 source 视频
        ↓
Evaluator 比较 A/B，只依据视频内容生成偏好与可靠度
        ↓
伪监督数据集
        ↓
学生 VLM 的质量评分后训练
        ↓
独立 target 视频 → 学生单视频质量分数
        ↓
评测脚本对照人类 MOS → SRCC / PLCC 等结果
```

**主要研究问题：** 在相同学生初始化与训练条件下，对抗证据搜索、视频核验、层级裁决及有限递归，能否比 EvoQuality 式自一致性投票产生更有效的后训练监督？

本项目最终评价对象是**学生的感知质量预测能力**。教师自己的满意度、训练 reward 或内部一致性承担诊断角色。

## 1.2 相对 EvoQuality 的方法位置

EvoQuality 原任务为图像质量评价。它在离线比较中已使用 reasoning，通过多数票形成相对标签，再以 fidelity reward 和 GRPO 训练单图评分能力。[R1]

本设计研究怎样**检查并利用 reasoning 中的视觉依据**。在视频任务上运行的对照统一称为 **EvoQuality-style video adaptation**，并披露输入适配方式。

| 层面 | EvoQuality-style 对照 | Design 1 候选机制 |
|---|---|---|
| 比较意见 | 多次自由比较 | 对称的 A 方 / B 方证据搜索，加独立中立观察 |
| 理由处理 | 汇总答案投票 | 结构化主张、证据定位、来源与分歧记录 |
| 标签裁决 | 多数票 | 中立 hyperjudge 结合视频与核验结果裁决 |
| 观察方式 | 预先确定的视觉输入 | 按争议补看已有视频中的片段或区域 |
| 递归 | 可跨轮次重新生成标签 | 单次标签生成内部的局部证据递归 |
| 输出 | 相对偏好 | 偏好、可靠度、证据记录、接近或暂缓状态 |
| 学习接口 | 伪排序驱动 GRPO | 同类 reward，加标准化后的可靠度加权 |

对抗上层裁判、推理审查、视频专家路由和标准化后的 confidence weighting 各自已有研究基础。[R13–R16] 候选创新落在**video-only 的证据核验机制、监督质量及端到端学习收益**；原创性与性能结论由文献对照和实验共同支持。

## 1.3 完整实验的三个必要产物

1. **Pre-supervise：** 固定 source 视频对上，各 evaluator 产生的可追溯伪标签与可靠度。
2. **Train：** 用这些监督训练得到的真实学生 checkpoint，以及可复现训练记录。
3. **Test：** 各学生在独立 target 上的逐视频预测、与 MOS 对照的指标、成本与不确定性报告。

完成 smoke test 仅表示链路可执行。研究结论要求覆盖完整的基线、新标签、可靠度与独立测试比较。

---

# 2. 当前数据与信息边界

## 2.1 “只有 synthetic 3D videos”的操作定义

**算法可访问的任务数据仅为已有 RGB 视频及其可直接解码的信息。** 预训练 VLM 的通用能力照常使用。

| 信息或动作 | 当前定位 |
|---|---|
| RGB 视频、原始帧、视频内时间戳、帧率、分辨率、时长 | 可用；用于观察、定位与复现 |
| 同一视频中的连续片段、加密抽帧、空间裁剪 | 可用；保留到原视频的映射 |
| 人类 MOS / 主观标签 | 独立评测脚本的参照；与 evaluator、学生训练隔离 |
| Mesh、Gaussian primitives、点云、相机内外参、深度、合成分割或光流标注 | 当前算法输入之外 |
| 原始 pristine/reference 视频 | 当前模型输入之外；主协议为 rendered-video no-reference QA |
| 合成方法、退化参数、严重程度、重建迭代数 | 当前监督依据之外 |
| 重渲染、新视角生成、几何干预、参考辅助反事实 | 当前架构之外 |
| 修复、生成式超分、插帧产生“新证据” | 当前证据接口之外 |
| 目标集视频用于学生训练或 target-time 更新 | 当前主协议之外 |

**测试参照说明：** 本文将“唯一数据源”理解为模型可用输入与伪监督依据为视频；完整效果测试仍使用项目原定的 3DGS-QA / MUGSQA 等基准已有 MOS。MOS 仅在预测完成后由独立脚本读取。若 Shi 希望连测试侧也只提供无标签视频，应先澄清评价依据：链路可完成运行，但 human-alignment、SRCC、PLCC 需要相应的人类参照。

## 2.2 信息隔离的实际含义

- 面向模型的身份使用匿名 `video_id`；文件路径里可能含重建方法、质量级别和分数，路径文本留在数据管理层。
- 同场景关系仅在有可靠的管理分组或视频内容证据时使用。当前设计本身独立于生成过程标签。
- 仅凭视频推断的几何或伪影成因，保存为**解释假设**。观察到的现象与解释分别记录。
- 对来源不同或轨迹不同的 A/B，各视频内分别定位证据。时间戳相同仅表示相同播放时间；几何对应关系需要可见证据支持。
- 裁剪和慢速检查服务于局部核验；整体感知权衡回到正常播放条件。
- 原始解码帧是证据基准。重编码、resize、颜色变换和抽帧会影响观测，全部记录，并在对照间统一。

## 2.3 本轮暂缓的设计

本轮暂缓：新增人类锚点库、主动合成偏差挑战样本、候选教师审核/自动回滚、跨轮次标签继承治理、长期能力记忆。它们由后续结果决定。

本轮保留：数据隔离、单次评价证据来源记录、有限终止条件、接近/暂缓区分、正常实验复现。已有模型的单次视觉核验属于核心方法。

---

# 3. 测试模型推荐与角色配置

## 3.1 推荐模型

以下为核对到的公开模型与作者资源。推荐理由是与本任务的接口、基线关联和可控比较；具体 QA 能力由本项目测试。[R1][R2][R3–R6]

| 模型 / 候选系列 | 推荐用途 | 选择理由与使用条件 |
|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | **优先基线锚点；学生和教师均可用** | EvoQuality 官方训练说明采用该底座；官方支持视频输入。最适合隔离 evaluator 架构变化。[R2][R3] |
| Qwen2.5-VL 的较小或较大公开尺寸 | 同系列容量 scaling | 各尺寸检查对应 checkpoint 和许可；大小变化与架构变化分开。[R3] |
| `Qwen/Qwen3-VL-4B-Instruct`、`Qwen/Qwen3-VL-8B-Instruct` | 新一代底座上的完整复核 | 官方提供视频使用方法；4B/8B 可构成独立容量轴。是否采用取决于当前训练链路兼容性。[R4] |
| `Qwen/Qwen3-VL-32B-Instruct` | 较强 subjudge / auditor / hyperjudge 候选 | 用于教师容量轴；更强教师带来的收益单列为 teacher-capacity 效应。[R4] |
| `OpenGVLab/InternVL3_5-8B` | 异构视觉判断分支 | 作者提供多帧视频调用示例；适合检验同构与异构来源。其训练与 processor 适配单独确认。[R5] |
| `ByteDance/EvoQuality` | 现有 QA checkpoint 的零样本对照 | 官方已发布权重；原训练历史与基础 VLM 有别，作为独立 checkpoint 对照记录。[R2] |

**建议主线：** 首个架构比较让所有角色和学生从同一个可训练 checkpoint 出发；随后再独立改变教师大小、hyperjudge 大小、学生大小和模型家族。

使用 Instruct / Thinking 变体时，把变体与 reasoning token 预算作为独立配置。公开模型的通用视频能力只支持候选资格，最终 QA 表现需要实测。

## 3.2 教师与学生的关系

| 模式 | 定义 | 结果应怎样表述 |
|---|---|---|
| **Self-teacher** | 教师各角色与学生起点使用相同 checkpoint；教师以更充分观察与多调用生成监督 | 当前模型能力的自生成监督与计算蒸馏 |
| **Larger-teacher** | 教师、auditor 或 hyperjudge 使用更大模型 | 教师容量提升与架构收益分别报告 |
| **Heterogeneous-teacher** | 多个公开模型家族参与评价 | 多源监督；记录来源与相关性 |
| **QA-specialized teacher** | 使用已有 QA 后训练 checkpoint | 额外 QA 训练历史作为信息条件披露 |

在同一轮 `pre-supervise` 中，教师权重固定。学生训练期间读取已冻结伪标签。正式 test 时各学生使用相同单视频评分协议。

## 3.3 工程选型的建议边界

优先检查 EvoQuality 官方实现中的 pairwise reward 与评分链路；其任务是图像，视频与新权重接口仍需适配。[R2]

TRL 文档提供 GRPO/VLM 与 PEFT 接口，可作为另一实现基础。[R17][R18] 文档示例的多模态支持范围与本项目的**双视频 reward 配对、视频 rollout、标准化后 sample weighting**分别核验。把处理器可读视频、推理框架可生成、训练框架可反向传播三项分别做通。

模型实际 revision、tokenizer、processor、视觉模板、量化配置与训练精度全部写入配置。本文保持框架、LoRA/全参数和并行方式开放。

---

# 4. Synthetic 3D videos 数据源推荐

## 4.1 优先级与适配判断

优先选用**已提供渲染视频、具有可见质量差异的 3D QA 数据**。通用合成运动视频提供内容与时序覆盖，其原始发布任务通常是跟踪或场景理解；能否提供足够的 QA 差异需要检查。

| 来源 | 公开资源与可用内容 | 本设计推荐用途 | 关键注意点 |
|---|---|---|---|
| **博士生已有 source 视频包** | 具体文件与覆盖由 Shi/Codex 盘点 | 项目主 source 首选 | 当前只读取 RGB；合成实现背景保持未知，MOS 与参数信息隔离 |
| **Explicit-NeRF-QA** | 作者仓库说明发布 22 个合成场景的 440 条 rendered videos，其中 420 条有 MOS；另有作者链接的 HF 数据页。[R7] | **公开 source 优先候选**；读取视频，把训练部分 MOS 隐藏；按内容分组留出诊断部分 | 有实际重建质量变化，较贴近本任务；NeRF 与 GS 的域差异明确记录。下载时选真实视频包，HF 默认图片预览不能替代视频 |
| **Kubric MOVi-C / D / E 的已发布数据** | 官方提供预生成多物体合成视频和 TFDS 入口；典型样本为短序列。[R8] | source 内容扩展、低成本端到端链路、教师 scaling 辅助 | 仅提取 RGB video；质量范围与 3D 重建伪影范围需要盘点。原生帧数限制观察 scaling |
| **PointOdyssey 的已发布 RGB 序列** | 作者提供长合成视频/RGB 帧序列及下载入口。[R9] | 较长时序 source 扩展 | 只取 RGB，保留顺序与真实播放时间；动态人物与物体是正常内容。跟踪/depth/camera 标注留在模型输入外 |
| **3DGS-QA** | 作者给出 225 个退化重建样本、MOS 与公开数据入口。[R10] | **锁定 target 测试优先** | 确认下载包含可评测的 rendered videos 与正确 MOS 映射。Native 3D prediction method 的资源与视频测试分开处理 |
| **MUGSQA** | 作者仓库说明有 2,414 个重建样本、对应 `.mp4`、MOS 以及 benchmark 脚本。[R11] | **锁定 target 测试优先** | 只给模型 `.mp4`；reference、`.ply` 和生成参数置于模型输入外；main/additional 的来源关系需检查 |

**建议数据组合：** 已有 source 或 Explicit-NeRF-QA 提供主要 QA 差异，MOVi / PointOdyssey 作为可选内容扩展；3DGS-QA 和 MUGSQA 留作独立 target。实际组合由数据可用性、近重复与内容重叠检查决定。

若博士生 source 已足够，首轮可以直接用它完成闭环，公开 source 作为后续扩展。

## 4.2 数据可用性的核对范围

本稿核对了作者页面/README 中的发布说明和入口，尚未验证完整压缩包、数据总字节数、逐视频可解码性或下载凭证。实施时先做小样本盘点，再确定有效规模。

- Explicit-NeRF-QA 的多视角图片、模型与视频是不同内容；当前需要发布的**观看视频**。
- MOVi 使用已发布序列，PointOdyssey 使用已发布 RGB 帧；当前运行包含视频读取，3D synthesis 为独立任务。
- 3DGS-QA 作者页当前提供数据入口，其预测模型代码仍标注准备中。主实验可独立完成逐视频推断与 MOS 对照。[R10]
- MUGSQA 官方脚本包含 target 内微调和 logistic fitting 路径。本项目只复用符合已声明协议的读取与相关性计算部分。[R11]
- 下载前核对数据、底座模型及附属素材的许可；发布模型、伪标签和视频时分别检查权限。

## 4.3 当前设定下的数据不足怎样处理

如果某 source 的视频几乎同质量，只扩大 pair 数量会得到大量接近判断。应报告原始视频与可见差异覆盖，再选择更合适的**已有渲染视频来源**。

本文的默认 source 保持为已提供的视频。额外人工退化、视频合成或重渲染应先作为新的数据实验与 Shi 确认，其参数也应与教师输入隔离。通用模拟视频上的伪标签收益与真实 3DGS 伪影迁移分开解读。

Objaverse/ShapeNet 等原始资产，以及仅给稀疏多视角图像的资源，属于未来渲染管线候选；当前 video-only 主实验使用直接可读取的连续视频。

---

# 5. 完整顶层架构

## 5.1 三阶段总图

```text
SOURCE：已有 synthetic / rendered RGB videos
       │
       ├─ 分组划分、匿名化、配对 → 固定 source pair manifest
       │                             │
       │                      PRE-SUPERVISE
       │                             │
       │      中立观察 + A/B 双边证据搜索
       │                             ↓
       │                 结构化证据与来源记录
       │                             ↕
       │            事实核验 + 视频内主动观察
       │                             ↕
       │               有限的局部证据递归
       │                             ↓
       │              多个独立 neutral hyperjudge
       │                             ↓
       │          最终 pseudo-label + reliability + trace
       │                             │
       └─────────────────────────────┤
                                     ↓
                                   TRAIN
                                     ↓
                    学生单视频评分 rollout groups
                                     ↓
                    伪排序 → fidelity/ranking reward
                                     ↓
                    GRPO 标准化 → reliability 加权
                                     ↓
                           学生 checkpoint
                                     ↓
                                    TEST
                                     ↓
             独立 target RGB videos → 单视频质量分数
                                     ↓
                独立脚本读取 MOS → 泛化与成本结果
```

## 5.2 M0：统一评价规范

固定主目标：**在指定观看条件下，预测视频的总体人类感知质量排序。** 空间质量、时序稳定性和可见结构问题承担证据维度。

原生 source/target 的观看协议与 frame sampling 记录清楚。学生输出单一 scalar 的尺度在运行前固定；评价重点为同协议内的排序与相关性。

本轮依据为视频中可见质量。美学题材、文本流畅度、背景音乐与内容题目完成度另属其他评价目标。

## 5.3 M1：独立中立观察

中立观察者先读取 A/B、视频时间信息和评价规范，独立记录初判、关键观察与覆盖局限。完成后冻结这份记录。

它提供直接视觉路径，用于分析后续裁决为什么改变。中立初判可以被新核验证据修正。

## 5.4 M2：对称、初始隔离的 advocates / subjudges

以 `n_adv_pairs` 组组织；每组一个搜索 A 优势，一个搜索 B 优势。**例如 10 组对应 10 个支持 A 的任务和 10 个支持 B 的任务，总计 20 个 advocates。** 它们的固定立场计数是预算分配，最终偏好来自中立裁决。

同组双方拥有对称的初始视觉预算。各组可分配全局、空间、连续时间片段等观察重点。初始上下文互相隔离，随后才提交报告。

每位 advocate 产生：

| 报告内容 | 要求 |
|---|---|
| 具体视觉观察 | 对应实际读取的输入 |
| 定位 | 视频身份、时间/帧、可选区域 |
| 影响 | 与规定质量维度的联系 |
| 支持方向与强度 | 可说明证据较弱或质量接近 |
| 局限与相反观察 | 保留尚待核验的内容 |
| 自报 confidence | 记录为辅助信号；语义优先为该观察的清晰度 |

Reasoning 使用**可输出的简要视觉证据说明**。报告长度是 scaling variable。获取系统隐藏内部推理的能力位于本项目接口之外。

可选的“接近假设”和“覆盖检查”属于同一搜索阶段的额外职责，规模单独记录。

## 5.5 M3：证据图 / evidence ledger

将报告组织为可追溯状态，区分：

- **观察：** 某区域亮度发生变化。
- **解释：** 可能涉及闪烁或正常视角变化。
- **感知影响：** 正常观看中有多明显、持续多久。

图中保存支持、冲突、引用、重复来源、待核验和修订关系。同一现象被多人描述时，保留共享来源；少数但可核验的关键观察保留完整索引。

本稿要求逻辑关系，具体图结构、字段与存储方式由 Codex 决定。

## 5.6 M4：视频事实核验

Auditor 回到原视频，核验决定性主张。优先隐藏 advocate 身份与自报 confidence，减少社会线索影响。

核验器可执行：读取连续片段、在现有帧中加密采样、读取原分辨率区域、回到全局正常播放语境。每次获取的信息可回溯到原视频。

输出为“有支持 / 有相反证据 / 观察待补充”及其位置与局限。模型的核验结论仍属于待评估判断。

**当前 video-only 限制：** 核验器以已有像素证据判断；参考渲染、相机标注、3D 结构和合成参数均排除在当前接口。视频未覆盖的视角保留为观察边界。

## 5.7 M5：调度与有限 recursion

触发原则：争议可能改变最终标签，追加已有视频观察有机会澄清。

每个子调用绑定一个局部问题，例如“这个区域的跳变是否在连续帧中可见”；返回核验记录。根裁判仍负责总体感知权衡。

调度器可以追加观察、调用专项核验、复用缓存、返回或结束。配置必须同时定义最大递归深度、分支数、节点数、调用数或 token/视觉预算上限，以及终止原因。

同一主张与同一观测结果缓存复用。递归中的参数固定；外层学生训练轮次用另一个变量管理。

## 5.8 M6：中立 hyperjudges 与最终裁决

多个 hyperjudge 在独立上下文中读取原视频/已声明观察、评价规范、匿名证据图与核验记录。先分别给结论，再聚合。

综合维度包括现象强度、持续时间、可见面积、显著性及正常播放体验。支持材料保留少数意见与未解决的争议。

最终聚合方法由 Shi/Codex 固定并做对照：简单多数票、显式证据汇总、其他可靠度感知聚合。研究对象始终是**标签方向与其依据**。跨层投票数只在相应中立判断层统计。

## 5.9 M7：独立功能的可靠度估计

可靠度模块可以是规则函数、轻量模块或独立提示；本轮主线保持 label-free，学习一个额外校准器属于另外声明的监督设定。

输入候选：核验支持程度、覆盖、关键争议解决情况、中立通道稳定性、hyperjudge 分歧、A/B 顺序交换后的稳定性。原始分数分别保存。

**方向、感知差距、标签可靠度分别表示。** 同组 A/B advocate confidence 的归一化仅是候选相对特征；原始强度仍保存。输出默认称 `reliability_score`，与人类正确率的概率解释需要额外校准依据。

## 5.10 M8：结构化监督包

至少输出以下逻辑内容：

| 字段类别 | 内容 |
|---|---|
| 身份 | pair、A/B canonical video IDs、展示顺序、source/split |
| 标签 | `A_better / B_better / near_equal / abstain` |
| 训练目标 | 声明好的偏好编码；接近与弃权分离 |
| 可靠度 | score、映射后的 weight、形成依据与版本 |
| 证据 | 简短主张、视频内定位、核验与来源依赖 |
| 决策 | hyperjudge 原始结果、最终聚合及剩余分歧 |
| 观察 | 原帧时间、裁剪映射、实际读取量 |
| 运行 | checkpoint、配置哈希、种子、成本、递归与失败状态 |

`near_equal` 表示已有证据支持感知接近；`abstain` 表示当前依据尚不足以用于排序训练。解析失败、读取失败与预算终止分别记录。

---

# 6. 数据组织与完整 data flow

## 6.1 数据分工

| 集合 | 用途 | 标签与模型访问 |
|---|---|---|
| `source_train` | 生成伪标签、后训练学生 | RGB + 自动伪监督 |
| `source_dev_unlabeled` | 运行诊断、稳定性与固定训练协议检查 | RGB；伪 reward 仅作内部诊断 |
| `source_diag_labeled`（具备时） | 独立 evaluator 质量和可靠度诊断 | 模型只看视频；脚本读取保留的已有 MOS |
| `target_test_3dgsqa` / `target_test_mugsqa` | 最终跨来源效果测试 | 冻结学生盲推断；脚本随后读取 MOS |

若 `source_diag_labeled` 的结果用于调 prompt、阈值、选择模型/超参，报告为**使用 source 人类开发标签**。严格 label-free 开发模式采用预先固定设置和预先规定的 checkpoint；带分数诊断在设置锁定后运行。

当前主线保持 target 视频与标签均处于训练之外。已有标签的离线测试与本轮暂缓的“人类锚点训练”是不同职责。

## 6.2 划分单位

优先按同一原始视频及其派生片段、可识别场景/物体分组划分。管理层若有可信的分组 ID，可仅用于数据隔离；模型输入保持匿名。

只有视频时，依据视觉近重复、序列关系和内容聚类做保守分组，并记录其为**估计分组**。证据不足时，泛化表述限于已验证的 clip/cluster/source 隔离程度。

跨数据源检查共同物体与近重复；不同数据集名称本身只证明发布来源不同。同一长视频切出的多个片段放在同一分区。目标 MOS 绑定哪个完整观看刺激，测试就恢复哪个刺激的预测，子片段沿用原 MOS 的做法需独立论证。

## 6.3 Pair 构造

先冻结 `pair_manifest`，再让各 evaluator 处理同一候选池。配对依据限定为视频身份、可识别内容、视频自身导出的特征与预先固定采样规则。

可以包含：同内容的可识别质量变体、视觉内容相近的视频、跨内容覆盖桥接 pair。各类分别统计。

如果只在各自物体内部排序，学生的跨物体绝对评分尺度可能缺少连接。主实验需记录比较图连通性，可加入盲于 MOS 的跨内容 pair；比较图设计与标签生成机制分开消融。

MOS 差距、GT degradation severity 和重建参数留在配对策略之外；使用它们分层分析时仅由评测侧在预测后进行。

A/B 顺序交换属于同一个 canonical pair 的重复呈现，数据规模同时报告 unique pair 与实际评估次数。

---

# 7. 实验阶段 I：Pre-supervise

## 7.1 输入与输出

**输入：** 固定的视频 manifest、split、pair manifest、教师 checkpoint、评价规范和 evaluator 配置。  
**输出：** 各方法的冻结监督包、逐 pair 审查轨迹、覆盖与成本统计。

## 7.2 阶段流程

1. **数据盘点与盲化。** 验证视频解码、分辨率、时间信息、去重和分区；把隐藏标签与可泄漏路径文本移到独立评测索引。
2. **建立共同的 source candidate pairs。** 保存随机种子和内容分组；注册所有教师配置。
3. **运行基线。** 单次中立比较、EvoQuality-style 自由 reasoning + voting，以及后文的结构对照。
4. **运行 Design 1。** 中立观察、对称 advocates、证据记录、视觉核验、按需递归、中立 hyperjudges、可靠度输出。
5. **冻结 pseudo-label bundle。** 保留全部候选的状态、原始 confidence、最终标签、weight 与实际成本。
6. **生成训练视图。** 所有方法采用声明好的接近、弃权、失败与保留策略；同时生成共同覆盖子集与各自可用全集。
7. **做独立教师诊断。** 在已划分的诊断视频上重复盲推断，由脚本对照保留 MOS；诊断输出与训练监督文件分开。

Pre-supervise 中比较对象是原视频；预训练模型可生成简短证据报告。教师推理轨迹的长度或措辞只属于诊断，默认训练 target 为偏好与可靠度。

## 7.3 关键语义

首轮主比较推荐使用同一偏好编码族：A、B、明确接近分别映射到固定的 pairwise 目标；`abstain` 独立掩码。精确数值、soft target 与 reward 映射由 Codex 和 Shi 固定。

教师 confidence 表示标签依据，学生 rollout reward 表示输出与伪监督的一致程度。二者分别存储。

全 pair 成本包括被弃权、解析失败和递归到预算上限的调用；有效标签数与原始候选数同时报告。

---

# 8. 实验阶段 II：Train

## 8.1 训练目标：单视频质量评分学生

**主学生输入：单个 RGB 视频。**  
**主学生输出：可解析的一个质量 scalar，可附统一长度上限的简短说明。**

训练过程使用 A/B 关系定义 reward，但学生评分提示本身只看到被评分的视频。这样正式部署与 test 可以独立给每条视频打分。

Pairwise 直接输出 A/B 的学生可以作为独立模型变体；若采用它，需要另行定义其全局评分恢复与成本，结果单列。

## 8.2 主训练路线

沿用 EvoQuality / VisualQuality-R1 类的质量评分到 pairwise ranking reward 思路。[R1][R12]

```text
冻结的 pair 标签 (A,B) 与可靠度
             ↓
学生分别对 A、B 产生多次单视频评分
             ↓
评分及组内统计 → 相对质量预测
             ↓
与冻结伪偏好比较 → fidelity / ranking reward
             ↓
按声明的 rollout group 计算并标准化 advantage
             ↓
应用来自教师的可靠度 weight
             ↓
GRPO policy 更新；KL 与其他约束独立管理
```

保持基线和 Design 1 的 reward family、学生输出形式、初始化与优化设置相同，首先隔离**监督来源**的变化。

**实现前需要固定一个接口选择：**

- `pair-centric`：每个 pair 含 A/B 两个单视频 rollout group，关联同一 pair 监督及其可靠度。
- `multi-partner`：一个视频的 reward 汇总多个比较对象，沿用 EvoQuality 式聚合；pair 权重与组级可靠度分别定义。

两种选择均可形成完整训练，但同一主比较统一采用一种，并报告与原文的适配差异。

## 8.3 Reliability 的接入约束

本方法的关键是**在最终标准化后保留样本间权重差异**。一个抽象接口为：

```text
standardized_advantage = group_normalize(rollout_rewards)
weighted_policy_term = teacher_reliability × clipped_policy_term(standardized_advantage)
loss = aggregate(weighted_policy_term) + separately_managed_KL
```

这里 `teacher_reliability` 来自冻结监督端；`group_normalize` 描述训练组内相对优劣；`clipped_policy_term` 是相应 GRPO policy 更新项。符号仅定义接口与作用位置，具体公式、数值稳定项及归约方式由 Codex 核对。标准化后置信度加权有 AQA-TTRL 的相关先例。[R16]

必须检查：

- 同组公共权重在最终标准化之后接入；后续再标准化或逐组除以权重总和时检查是否抵消。
- Pair 内证据融合权重与整个 rollout group 的训练权重分别管理。
- 全局平均 weight、有效 batch、KL 相对强度及学习率分别记录；设置等全局训练强度的对照。
- 零/近零 reward 方差、全接近 pair、格式失败与越界评分有明确处理，保留发生率。
- 原始学生输出中的自报 confidence 保留为文本，训练权重由教师文件读取。
- 自动按 reward function 加权的框架接口，与本设计的逐样本 policy 权重分别核对。

## 8.4 训练比较矩阵

| 编号 | 监督来源 | 训练权重 | 要隔离的因素 |
|---|---|---|---|
| **T0** | 相同初始学生 | — | 训练前单视频 QA 水平 |
| **T1** | EvoQuality-style 伪标签 | 全部为 1 | 基础 self-training 收益 |
| **T2** | Design 1 伪标签 | 全部为 1 | 新标签本身的收益 |
| **T3** | 与 T2 相同的伪标签 | 审查可靠度 | 完整方案 |
| **T4a** | 与 T2 相同的伪标签 | Hyperjudge 自报 confidence | 便宜权重参照 |
| **T4b** | 与 T2 相同的伪标签 | 中立 hyperjudge 一致性 | 分歧信号参照；advocate 席位计数保持预算语义 |
| **T5** | 与 T2 相同的伪标签 | 打乱样本对应关系的审查可靠度 | 相同权重分布下的样本匹配价值 |
| **T6**（扩展） | 与 T2 相同 | 硬筛选；保留样本权重为 1 | 筛选与连续加权的区别 |
| **T7**（扩展） | 强教师 / 异构教师生成 | 同一训练协议 | 教师容量或来源效应 |

**完整主结果至少包含 T0、T1、T2、T3。** 可靠度机制的完整解释还需要简单 confidence/一致性及 T5 对照。预算限制时降低每组规模，并报告实际完成的矩阵。

共同覆盖子集用于严格比较；各方法可用全集用于衡量整体产出。分别报告两种视图，控制弃权、数据数量和容易样本偏好。

## 8.5 更新方式与其他训练分支

LoRA / 全参数、视觉编码器是否更新、训练步数、rollout 数量等作为配置变量。每组实验保持相同初始化和可训练模块。

SFT、DPO、其他 ranking objective 可以作为后续训练算法轴。首个核心对照优先保持 GRPO 与 reward family，防止多处改变掩盖 evaluator 的作用。

默认完整运行包含**一次 pre-supervise 和一次学生后训练**。外层 self-training round 数可作为 scaling 变量扩展；每轮标签独立缓存，且与单次推理 recursion 深度分开记录。跨轮次风险控制仍按 Shi 的要求暂缓。

## 8.6 Checkpoint 选择

主协议采用预先规定的训练长度与 checkpoint 规则。可用的 source 开发指标及其标签权限在运行前声明。

Target MOS 保持在最终比较阶段。若使用有标签 source 开发集选 checkpoint，结果明确记为 source-supervised model selection。纯伪 reward 上升只说明对教师监督更一致，最终能力由独立 test 判断。

---

# 9. 实验阶段 III：Test

## 9.1 两类测试分别报告

### A. 教师 / evaluator 测试

在独立诊断视频对上，比较不同 evaluator 的偏好质量、可靠度、覆盖率和成本。可在所有设置锁定后运行一组 target pair 诊断，并与学生 test 分表。

### B. 学生完整测试——主结果

```text
锁定 checkpoint 与推断协议
          ↓
逐个 target video 输入学生
          ↓
输出 scalar quality score，先保存完整预测
          ↓
独立脚本按 ID 连接已有 MOS
          ↓
每个 dataset 的 SRCC / raw PLCC / 区间与覆盖
```

学生主结果使用相同单视频推断预算。额外启用多 agent 教师在 test-time 帮助评分的结果单列，避免把推断计算收益记成训练收益。

## 9.2 评价指标

| 对象 | 指标与解释 |
|---|---|
| 学生 | **SRCC**：预测与 MOS 的排序相关；**raw PLCC**：原始分数与 MOS 的线性相关 |
| 学生 | 可附 KRCC；采用 RMSE 时先说明分数尺度和映射 |
| 教师 | 与同协议 MOS 排序的一致率；明显差距与接近 pair 分别统计 |
| 可靠度 | risk–coverage：保留不同份额高可靠度标签后的错误率；概率定义充分时再报告校准指标 |
| 标签变化 | 相对基线纠正的错误、改错的正确判断及净变化 |
| 系统 | 原始候选数、有效标签数、弃权、失败、实际调用与成本 |

MOS 是平均主观分数，MOS 差值生成的是平均评分排序参照。把它解释成人类成对投票比例需要相应的逐人偏好数据。

接近 pair 的阈值或不确定区间在开发协议中固定；最终给出阈值敏感性。A/B 硬排序、接近和 abstain 保持分离。

MUGSQA 官方相关性脚本提供 logistic fitting 选项。[R11] 主表采用 raw PLCC；若报告 target 上拟合后的 PLCC，作为 benchmark-specific 后处理另列，披露拟合位置与其信息条件。

## 9.3 泛化、统计与正确比较

- 每个 target 单独计算指标。不同 MOS 量表跨数据集合并时另立协议。
- 以物体/场景或可确认的内容组估计结果波动；同一物体产生大量 pairs 时避免当成等量独立观测。
- 同时报告按 sample 和按内容组的视图，并记录随机种子/重复次数。
- 视频读取或评分解析失败保留在覆盖率中，分表报告共同可评估子集。
- Target 内 train/test split 的已发表结果作为**使用 target 监督的上下文参照**；同视频信息、同监督条件的基线放在主比较表。
- Native 3D、reference-based 方法所用信息不同，单独标记；当前方法的模型输入为 RGB video。

**测试标签缺失时：** 写明缺失的是哪一指标的参照，输出可执行的预测和内部诊断，先向 Shi 确认可获得的评分来源。教师自己生成的标签继续承担伪监督角色。

---

# 10. 具体实验模块与机制对照

## 10.1 E：Evaluator 架构主比较

| 编号 | 配置 | 作用 |
|---|---|---|
| E0 | 单次中立视频比较 | 直接判断基线 |
| **E1** | 自由 reasoning + majority voting | EvoQuality-style video 主基线 |
| E2 | 自由报告 + 中立 hyperjudge | 利用理由汇总 |
| E3 | 对称 A/B advocates + 平铺中立裁判 | 双边证据搜索 |
| E4 | E3 + 证据记录 + 原视频 auditor + 中立独立路径 | 核验与信息组织 |
| **E5** | 完整 Design 1，含主动观察、有限 recursion 和可靠度 | 完整架构 |
| EP | 博士生现有 prototype | 取得真实代码后复现其 evaluator |
| ER | VISTA 式 normal/adversarial/meta 对照 | 与已有相近层级结构比较；采用本项目 QA 规范。[R13] |

E1 也保留 reasoning；教师模型与输入质量公平配置。方法复杂度与模型大小分别改变。

## 10.2 A：结构消融

至少保存以下开关：自由/指定立场、vote-only/证据报告、原始汇总/结构化 ledger、中立初判开关、auditor 开关、静态/主动观察、递归深度、hyperjudge 数量、可靠度类型。

可先采用增加模块路径，再对完整配置单项移除。Codex 应支持模块组合，实际运行子集与 Shi 确定。

## 10.3 V：视觉与 reasoning 使用诊断

对固定视频实施受控输入变化：

| 变化 | 观察什么 |
|---|---|
| 交换 A/B 展示顺序 | 映射回 canonical 身份后是否稳定 |
| 修改报告措辞/长度，保留事实 | 文本风格对裁决的影响 |
| 同一证据重复多次 | 重复引用与新增独立信息的区别 |
| 插入错配时间/区域的报告 | 原视频核验对无依据理由的响应 |
| 加入可核验的少数证据 | 少数信息能否纠正初判 |
| 补充真实连续片段 | 结论改变是否关联新观察 |

仅有视频与 MOS 时，可以客观检查总体标签变化；对“某个精细伪影定位是否正确”的断言，需要相应独立审核。伪造报告只用于标记清楚的诊断集。

## 10.4 R：Recursion 与额外观察

比较：固定单轮、同预算更多独立采样、相同新增视觉材料的平铺审查、有限深度递归。

保存触发问题、实际新增帧/片段、子返回与根决策变化。Replay 对照只接收该次调用获得的视觉材料和同一任务条件，教师最终答案隔离。

## 10.5 W：可靠度与训练接口

教师侧比较不同 confidence 和 audit reliability 的 risk–coverage。学生侧运行 T2/T3/T4/T5，匹配标签、样本与整体更新强度。

保留 raw weight 分布、零权重比例、全局均值、组内 reward 方差和有效更新覆盖。数值权重在归约后是否仍影响样本相对贡献，应有独立测试。

## 10.6 S：Scale 与资源分配

规模研究包含模型、数据、agent、视觉观察、reasoning、recursion、学生训练以及推断预算。第 11 节给出变量清单。

分别报告：

- **固定教师计算预算：** 更多 pair 与更深审查之间的分配。
- **固定学生训练预算：** 更好标签带来的能力变化。
- **固定总预算：** 教师生成 + 学生训练 + 测试的整体收益。
- **固定观察信息：** 相同实际视觉证据下，组织方式与递归的作用。

## 10.7 主结论与扩展结论

主结论由 E1 与完整 evaluator 产生的 T1/T2/T3 等学生差异建立。模块消融解释机制。Scaling 解释何种资源条件下方法有价值。

规模筛选可以先利用便宜的诊断，但进入报告的代表性规模点需要完成对应的 pre-supervise、train、test。每个规模点记录完成范围，区分 evaluator-only 诊断与端到端结果。

---

# 11. Scaling variables：定义、单位与建议候选轴

以下全部为**待选择的配置维度**。示例值是讨论用网格，既非数据集保证容量，也非固定执行规模。实际候选点由 Shi/Codex 根据显存、数据、训练链路和预算确定。

## 11.1 模型与来源

| 变量 | 含义 | 候选轴 / 记录方式 |
|---|---|---|
| `student_model_id` / `student_size` | 学生模型和参数规模 | Qwen2.5-VL 或 Qwen3-VL 各公开尺寸；保持同系列比较 |
| `advocate_model_id` / `advocate_size` | 第一层视觉论证能力 | 与学生同模型 / 更大同系列 / 异构 |
| `auditor_model_id` / `auditor_size` | 事实核验能力 | 独立容量轴 |
| `hyperjudge_model_id` / `hyperjudge_size` | 综合裁判能力 | 独立容量轴；如 8B 与 32B 的候选比较 |
| `neutral_model_id` | 中立观察底座 | 与主体同构或异构 |
| `model_family_count` | 实际使用的不同模型家族数量 | 1、2、3 等；角色数量另计 |
| `model_variant` | Instruct / Thinking / QA checkpoint | 训练历史与文本预算披露 |
| `weight_precision` / `train_precision` | 教师权重与学生计算精度 | 量化与高精度分开记录 |
| `trainable_capacity` | 可训练模块及 LoRA/全参规模 | LoRA rank、vision/projector/text 是否更新 |

Dense/MoE 若同时出现，记录总参数、活跃参数与实测成本；名义参数量保持具体解释。

## 11.2 数据与配对

| 变量 | 含义 | 候选轴 / 记录方式 |
|---|---|---|
| `n_source_videos` | 唯一 source 视频数 | 实际有效数据的 25% / 50% / 100% 等嵌套子集 |
| `n_source_groups` | 独立/估计内容组数 | 与视频数分别报告 |
| `n_source_domains` | 不同 source 来源数 | QA rendered source；再加入 MOVi / PointOdyssey |
| `source_mix` | 不同来源的采样比例 | 固定总视频数时改变配比 |
| `n_unique_pairs` | 唯一无序视频对数 | 嵌套 pair 子集；与调用次数分开 |
| `partners_per_video` | 每个视频的比较对象数 | 如 2 / 4 / 8；受候选池与图结构限制 |
| `within_group_pair_fraction` | 同可识别内容 pair 比例 | 与跨内容桥接比例联合记录 |
| `pair_graph_connectivity` | 比较图连接程度 | 连通分量、度分布等描述 |
| `pair_presentations` | 每个 canonical pair 的展示次数 | A/B 交换和随机重复均计入 |
| `train_label_coverage` | 原始 candidate 中实际参与训练份额 | 与弃权、失败、筛选分别报告 |
| `n_test_videos` / `n_diag_pairs` | 独立测试与诊断规模 | 按公开数据实际有效规模选择 |

扩大 pair 数与扩大独立场景覆盖分别解释。同一视频切出更多 clips 单独记为 clip scaling。

## 11.3 Judge 与层级规模

| 变量 | 含义 | 建议讨论网格 |
|---|---|---|
| `n_adv_pairs` | 独立 A/B 对抗组数 | **1 / 3 / 5 / 10** |
| `n_advocates` | advocate 总数 | 每组 A、B 各一位时为 `2 × n_adv_pairs` |
| `advocate_samples_per_role` | 每位角色的随机重复采样 | 1 / 2 / 4 等 |
| `n_neutral_observers` | 中立初判任务数 | 1 或更多；单项 scaling |
| `n_auditors_per_claim` | 每条主张的核验重复/来源数 | 1 / 2 / 3 等 |
| `n_hyperjudges` | 独立上层裁判数 | **1 / 3 / 5** |
| `n_lenses` | 观察重点数量 | 全局、空间、时序等 |
| `n_claims_per_role` | 单份报告保留的主张数量 | 与文本长度、证据覆盖共同控制 |
| `baseline_votes_K` | EvoQuality-style 自由投票次数 | 与实际计算预算对齐；独立于 `n_adv_pairs` |
| `aggregation_mode` | 上层结果汇总方式 | 多数票 / 证据综合等，具体计算另定 |

同底座的多次调用是角色/采样增加；与真正模型多样性分别比较。

## 11.4 视觉与 reasoning 预算

| 变量 | 含义 | 候选轴 / 约束 |
|---|---|---|
| `input_mode` | Native video / 有序多帧 | 适配方式固定或单独对照 |
| `initial_unique_frames` | 初始看到的原生帧数 | 如 8 / 16 / 32 / 64；以视频实际帧数为上限 |
| `fps_sampling` | 时间采样密度 | 保留原时间；与总帧预算的优先级明确 |
| `frame_resolution` / `pixel_budget` | 每帧可见细节 | 以模型实际 resize 后数值记录 |
| `clip_duration` | 每个观察片段覆盖的真实时长 | 全局覆盖与局部片段分开 |
| `n_reinspection_clips` | 主动补看的片段数量 | 固定总视觉预算时比较分配 |
| `crop_count` / `crop_resolution` | 局部空间核验量 | 从已有帧提取，并保留全局语境 |
| `visual_token_budget_per_pair` | 每对视频总视觉处理量 | 所有角色与重复观察均计入 |
| `report_token_budget` | advocate / auditor / hyperjudge 输出上限 | 按角色分别记录 |
| `evidence_context_budget` | 上层能接收的报告/证据长度 | 截断、去重、摘要策略必须记录 |
| `sampling_temperature` / `top_p` | 生成随机性 | 各方法公平控制 |

重复帧计入处理成本，独立视觉信息覆盖按 unique frames 记录。MOVi 的短序列会对帧数 scaling 形成实际限制。[R8]

## 11.5 递归与主动调度

| 变量 | 含义 | 候选轴 |
|---|---|---|
| `recursion_depth` | 根层之下的最大子问题深度 | 0 / 1 / 2 等有限值 |
| `branching_factor` | 每个子问题展开的最大分支 | 1 / 2 / 3 等 |
| `max_evidence_nodes` | 单个 pair 可展开节点总数 | 与调用/视觉预算联合设置 |
| `max_calls_per_pair` | 总模型调用上限 | 包含正常及重试调用 |
| `trigger_policy` | 哪些争议触发补看 | 固定规则/可配置调度，学习调度另立分支 |
| `stop_rule` | 澄清、接近、证据耗尽、预算结束 | 输出终止类型 |
| `cache_reuse_policy` | 相同观测/问题的缓存复用 | 成本统计区分读取与实际新推理 |

递归深度、agent 数和观看量相互耦合。至少同时提供 matched-compute 和 matched-observation 视图。

## 11.6 学生训练与最终推断

| 变量 | 含义 | 候选轴 / 约束 |
|---|---|---|
| `rollouts_per_video_G` | 每个学生输入的评分输出组大小 | 如 4 / 8 / 16；与教师投票 K 分开 |
| `effective_batch_size` | 每次有效训练的输入/组数 | 声明 pair、video 和 token 三种口径 |
| `train_steps` / `train_token_budget` | 学生优化量 | 与 unique labels 和实际见过样本量分别记录 |
| `learning_rate` / `optimizer` | 更新设置 | 主方法对照固定；超参搜索预算相同 |
| `lora_rank` / `trainable_modules` | 参数更新容量 | 同初始化、同模块对比 |
| `score_output_protocol` | 单视频评分尺度、格式与可选简短说明 | 训练与 test 对齐 |
| `reward_mapping` | 评分到比较 reward 的映射 | 主对照同一 family；精确计算与 Shi/Codex 固定 |
| `weight_mapping` | reliability 到训练权重的映射 | raw 分布保留；聚合位置受第 8 节约束 |
| `kl_strength` / `clipping` | 更新约束 | 独立于监督 weight 管理 |
| `self_training_rounds` | 外层伪标签刷新与训练轮次 | 1 为完整单轮；2 / 3 可作扩展 |
| `test_frames` / `test_repeats` | 学生 test 视觉预算与重复次数 | 主比较固定；推断 scaling 单独作图 |
| `n_seeds` | 教师与训练随机重复 | 如 1 / 3 / 5；分别记录两个随机性来源 |

## 11.7 怎样选择规模组合

先固定一个实际可训练底座和一套有效 source；选择代表性的 evaluator 规模完成整条链。随后分轴改变规模，避免同时扩大教师、数据、学生和 test 预算。

建议保留三种配置模板：

- `smoke_e2e`：最小可执行完整闭环，用于接口正确性。
- `main_e2e`：固定底座与观察协议，完成 T0/T1/T2/T3 及可靠度参照。
- `scaling_e2e`：沿独立资源轴改变规模，并输出相应学生 test 结果。

模板中实际参数由 Shi/Codex 写入。任何 evaluator-only 子运行在名称和报告中明确标记。

---

# 12. Codex 应交付的运行产物与验收条件

## 12.1 建议的逻辑产物

```text
run_config / environment_manifest
video_manifest / split_manifest / pair_manifest
pre_supervise/
    baseline_labels
    design1_labels
    evidence_traces
    reliability_records
    cost_and_failure_records
train/
    training_manifests
    optimizer_and_reward_logs
    checkpoints
    normalization_and_weight_tests
test/
    frozen_protocol
    per_video_predictions
    evaluator_pair_diagnostics
    per_dataset_metrics
    coverage_cost_uncertainty
report/
    e2e_comparison
    ablations
    scaling
    limitations_and_open_questions
```

这是逻辑内容清单，具体目录、格式和类设计由 Codex 决定。

## 12.2 正确性检查

- 视频 A/B 交换后，canonical label 与证据索引映射正确。
- 视觉定位能回到真实读取的原始帧，递归调用在固定上限内终止。
- Teacher、train、test 的数据权限隔离；MOS 和路径中的标签信息只进入对应评测索引。
- 学生 train 实际产生梯度、参数变化和可重新载入的 checkpoint。
- 全部 weight 为 1 时恢复共同基础训练接口；weight 为 0 时对应 policy 贡献归零，KL 按独立策略处理。
- 同组 confidence 缩放在最终标准化后仍影响该组相对训练贡献。
- Zero-variance reward、near-equal、abstain、读取失败和格式失败的行为可复现。
- 目标 test 逐视频预测先保存，之后连接 MOS；原始预测和 metric 可追溯。

## 12.3 完整交付标准

**必须有：冻结伪标签、实际后训练 checkpoint、独立 test 预测，以及 T0/T1/T2/T3 的端到端比较。** 可靠度与结构结论注明对应完成的消融；正面、相近、退化和证据不足的结果均保留。

报告同时回答：

1. 新 evaluator 产生了怎样的标签变化？
2. 这些变化是否转化成学生泛化？
3. Weight 的价值来自样本匹配、筛选，还是整体更新强度？
4. 额外教师计算是否划算，收益随哪些 scaling variables 改变？
5. Video-only 条件下，哪些证据缺口仍影响方法？

---

# 13. 实施前向 Shi 澄清的具体事项

**完整 pre-supervise → train → test 已经确定。** 下表只保留会改变实际运行配置的事项：

| 事项 | Codex 应先提供的信息 |
|---|---|
| 博士生现有 source 视频、repo 和 prototype 是否可访问 | 可用路径、真实数据/代码清单、复用点 |
| 首选 source 与 test 组合 | 已下载内容、RGB/MOS 对应、许可、内容重叠检查 |
| Source 开发是否允许使用已有保留 MOS | 严格 label-free 开发与 source-label-assisted selection 的区别 |
| 测试 MOS 是否可获得 | 逐数据集的真实映射状态；只剩无标签视频时的指标缺口 |
| 模型、数据、advocates/hyperjudges、观察和 recursion 规模 | 可行候选配置及资源估计 |
| 学生训练框架与 pair/group 接口 | 现有实现的兼容性、需修改的位置 |
| 标签聚合、near-equal、abstain 和 reliability 映射 | 候选计算及其语义、baseline 对应 |
| 单轮或多轮、checkpoint 规则与执行的消融子集 | 完整 e2e 范围、主结果与扩展结果 |

规模和工程问题由 Shi/Codex 讨论后写进运行配置。此设计保持科学边界和实验接口稳定。

---

# 14. Useful resources

资源核对日期：**2026-09-25**。具体下载、许可、软件版本与兼容性在实施时再次核对；下列文字描述的是原资源支持的范围。

## 14.1 核心方法与后训练

**[R1] EvoQuality — Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking**  
论文：<https://arxiv.org/abs/2509.25787>  
固定版本：<https://arxiv.org/html/2509.25787v5>  
用途：§3.1–3.2、Table 1、Algorithm 1；reasoning/voting、单图评分、fidelity reward 与 GRPO 的核心参照。

**[R2] EvoQuality 官方代码与 checkpoint**  
仓库：<https://github.com/bytedance/EvoQuality>  
模型：<https://huggingface.co/ByteDance/EvoQuality>  
用途：现有 pairwise reward 与训练链路。README 给出 `verl` 目录、Qwen-Pair 训练和 image/pair 推断入口。以真实仓库和固定 commit 为准，示意路径由 Codex 核对后替换。当前设计需要视频适配和 policy 端可靠度接口。

**[R12] VisualQuality-R1**  
<https://arxiv.org/abs/2505.14460>  
用途：单图质量评分、多次采样、Thurstone 比较概率与 fidelity reward 的 ranking-RL 设计基础。

**[R19] Q-Insight**  
<https://arxiv.org/abs/2503.22679>  
用途：视觉质量的 RL 后训练参照；使用的监督条件与本设计分开记录。

**[R20] Q-Align**  
<https://arxiv.org/abs/2312.17090>  
用途：视觉质量的离散文本等级和评分表示参照。

## 14.2 模型与处理器

**[R3] Qwen2.5-VL 官方模型与 Transformers 文档**  
<https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>  
<https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl>  
用途：视频输入、时间采样、像素预算与主基线底座。其他尺寸从官方集合选定具体 checkpoint。

**[R4] Qwen3-VL 官方资源**  
<https://github.com/QwenLM/Qwen3-VL>  
<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>  
<https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>  
<https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>  
<https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl>  
用途：公开大小与 Instruct/Thinking 系列、视频使用与可选底座 scaling。

**[R5] InternVL3.5 官方模型卡**  
<https://huggingface.co/OpenGVLab/InternVL3_5-8B>  
用途：异构视觉模型候选；模型卡提供视频多帧调用示例。

**[R6] Qwen 官方视觉工具与训练指引入口**  
<https://github.com/QwenLM/Qwen3-VL>  
用途：processor、`qwen-vl-utils`、视频读取和 fine-tuning 资源。使用时固定版本；视频解码兼容性单独检查。

## 14.3 数据

**[R7] Explicit-NeRF-QA 作者仓库与数据页**  
<https://github.com/YukeXing/Explicit-NeRF-QA>  
<https://huggingface.co/datasets/Chole12/Explicit-NeRF-QA>  
用途：已渲染的 synthetic 3D QA videos；作者 README 说明视频、MOS 与其他数据类型。当前模型仅使用视频，训练部分 MOS 隔离。

**[R8] Kubric / MOVi 官方资源**  
<https://github.com/google-research/kubric>  
<https://github.com/google-research/kubric/tree/main/challenges/movi>  
用途：预生成 MOVi RGB 序列与 TFDS 入口；当前使用已有数据，合成脚本属于范围之外。

**[R9] PointOdyssey 作者资源**  
<https://pointodyssey.com/>  
<https://github.com/y-zheng18/point_odyssey>  
用途：长合成视频/RGB 序列下载；只提取视频信息。作者仓库描述数据许可，实施时核对所选版本。

**[R10] 3DGS-QA**  
论文：<https://arxiv.org/abs/2511.08032>  
作者仓库：<https://github.com/diaoyn/3DGSQA>  
作者列出的数据入口：<https://huggingface.co/datasets/dyn2024/3DGS-QA/>  
用途：独立 3DGS 主观质量测试。下载入口由作者仓库提供；原生预测方法代码的发布状态与数据可用性分别核对。

**[R11] MUGSQA**  
论文：<https://arxiv.org/abs/2511.06830>  
作者仓库：<https://github.com/Solivition/MUGSQA>  
用途：rendered `.mp4`、MOS 映射与 benchmark 工具。`reference/`、`.ply` 和 reconstruction 参数保持在本设计模型权限之外；target 微调与 logistic fitting 选项按本文测试协议处理。

## 14.4 架构相关工作

**[R13] VISTA — A Test-Time Self-Improving Video Generation Agent**  
<https://arxiv.org/abs/2510.15831>  
<https://arxiv.org/html/2510.15831v1>  
用途：视频领域的 normal/adversarial/meta judge 结构参照；原任务为视频生成反馈与提示优化。

**[R14] AgentAuditor — Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge**  
<https://arxiv.org/abs/2602.09341>  
用途：推理分歧组织和局部证据审查；原工作以语言推理任务为主。

**[R15] Q-Router — Agentic Video Quality Assessment with Expert Model Routing and Artifact Localization**  
<https://arxiv.org/abs/2510.08789>  
用途：视频质量专家路由、分层计算预算和时空定位；其专家能力与信息条件单独对照。

**[R16] AQA-TTRL**  
<https://arxiv.org/html/2510.05478v2>  
用途：§3.3、Eq. 5 的标准化后 confidence weighting；原任务为音频问答，与本项目监督来源分开表述。

## 14.5 软件接口

**[R17] TRL GRPO Trainer 官方文档**  
<https://huggingface.co/docs/trl/grpo_trainer>  
用途：GRPO、VLM、rollout/reward 与训练配置。当前文档的 VLM 示例主要以图像接口描述；本任务的视频与跨 pair reward 需要实际核验适配。

**[R18] PEFT 官方文档**  
<https://huggingface.co/docs/peft/en/index>  
用途：LoRA 等参数高效训练接口；可训练模块与 rank 是独立 scaling variables。

---

# 15. 本次修订摘要

- 实验范围调整为完整的 **pre-supervise → train → test**，后训练与独立测试均为正式交付。
- 加入公开模型候选、已有 synthetic/rendered video 数据源和资源入口。
- 所有教师与学生的任务数据统一为 RGB video；参考、合成参数、3D 几何与重渲染能力从本版架构移出。
- 保留对抗搜索、证据图、视觉核验、主动观察、有限 recursion、独立 hyperjudges 和可靠度加权。
- 明确定义模型、数据、subjudge/hyperjudge、观察、文本、递归、GRPO 和测试预算等 scaling variables。
- 将 baseline、结构消融、可靠度对照、完整学生训练与独立 MOS 测试串为同一实验。
- 当前暂缓跨轮次风险控制扩展；基础数据隔离与评价语义保持清楚。

**最终检验链条：新的证据评价机制 → 更有效的自动监督 → 实际学生更新 → 独立 3D 视频质量泛化结果。**

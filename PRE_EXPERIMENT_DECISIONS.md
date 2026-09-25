# SE-VLM3DQA-1：正式实验前的决策清单

日期：2026-09-25。状态：分析与讨论稿；用户已给出 72 小时小测试预算及有条件使用 Mac 的授权，正式数值配置尚未冻结，实验尚未启动。

依据：完整阅读 `/Users/shichen/Downloads/SE-VLM3DQA-design1.md`（v1.1），核对核心作者资源，并只读检查 Lop。文档中的执行约定与候选网格是设计内容，本轮用户请求为理解与决策分析。

## 已明确的执行约束

- 本地工作目录：`/Users/shichen/GitHub/SE-VLM3DQA-1`。
- 最新用户指示覆盖之前的 Lop-only 限制：Lop 为主要执行位置；若有实际帮助，允许使用本地 Mac 算力，包括适合且经验证的模型运行。不得将旧的远程限定继续视为用户现行要求。
- 本轮未下载模型或数据、未提交作业、未运行模型、未创建或切换 Git 分支。

## 任务理解

已有 source RGB 视频 → 固定 pair manifest → 冻结教师生成偏好、可靠度与证据记录 → 同初始化的学生用单视频评分 GRPO 后训练 → 在独立 target 上先保存逐视频分数，再由脚本连接 MOS。

需要检验的因果链为：评价架构改变监督 → 监督改变学生更新 → 学生跨来源质量预测改善。

核心对照：T0 初始学生；T1 投票标签、单位权重；T2 Design 1 标签、单位权重；T3 与 T2 相同标签、审查可靠度权重。T2−T1 检查标签来源，T3−T2 检查可靠度，T3−T1 检查完整方案。不同覆盖率、训练强度及教师预算会影响这些解释。

## 最新用户资源要求

- 用户希望小测试在 72 小时内完成，同时避免规模过小而缺乏代表性。
- 本地只读核查：MacBook Pro，Apple M1 Pro，8 CPU cores，14 GPU cores，16GB unified memory；工作目录所在卷可用约 68GiB。未执行模型。
- Mac 是否改善教师吞吐或训练可行性仍需实际验证；16GB unified memory 不能等同于独占 GPU 显存。
- 规划时将环境适配、下载、排队、教师标注、训练、测试与报告计入总时间。正式开始时间需在执行记录中明确。
- 代表性优先由独立场景覆盖、真实质量差异、全局与连续时序观察、完整机制和受控对照保证；大量相关 pairs 不能代替独立内容覆盖。
- 用户转发的 GPT Pro 文本作为建议上下文；其中未明确落实的数值和额外资源授权仍不得自动视为已批准。

## Lop 当前核查事实

通过 SSH、qhost、qstat、nvidia-smi、quota 只读查询，未提交计算作业：

| 节点 | GPU | 单 GPU 显存 | 驱动 |
|---|---|---|---|
| compute-1-20 | 4 × GTX 980 | 4043 MiB | 465.19.01 |
| compute-1-21 | 4 × GTX TITAN X | 12212 MiB | 465.19.01 |
| compute-1-22 | 4 × Tesla K80 GPU devices | 11441 MiB | 465.19.01 |
| compute-1-23 | 4 × Tesla K80 GPU devices | 11441 MiB | 465.19.01 |

查询时 gpu1@compute-1-20 显示 0/40 queue slots，其他三个 GPU 队列实例显示 40/40。Queue slots 不是 GPU 数；即时空闲也不是资源预留。`qstat -u shic4` 当时无作业记录。

Home quota：约 207G / 250G；名义剩余约 43G。文件系统总体空闲 34T 不能视作该账号可用配额。

结论：当前资源约束明显。7B 视频 GRPO、32B 教师、量化后端和多 GPU 训练均未验证可行。多卡显存不自动合并。须检查 GPU 架构、驱动、PyTorch/CUDA、attention kernel、量化与训练框架的完整兼容链，并在获准的 Lop 工程 smoke test 后估计吞吐。上述硬件信息是即时快照。

## 需要用户提供的信息或决定的科学范围

| 编号 | 决策 | 建议或可选项 |
|---|---|---|
| D1 | 博士生 source、repo、prototype、checkpoint、配置及现有结果的位置 | 提供路径/链接；若没有，则明确以公开资源独立起步。已有“接近现有方法”的说法需要对应 split、输入、监督与指标协议。 |
| D2 | 首轮 source 和 target | source 优先已有合适数据，否则 Explicit-NeRF-QA；target 选择 3DGS-QA、MUGSQA 或两者，并预定主要数据集及 MUGSQA subset。 |
| D3 | 是否接受文档的 RGB-only / no-reference / 跨来源协议 | 默认沿用：MOS 不进入教师或训练；生成参数不参与配对；target 不参与训练或适配。管理层分组信息可用于隔离。 |
| D4 | Source MOS 开发权限 | 推荐严格 label-free：设置冻结后才做有标签诊断。若用于 prompt、阈值或 checkpoint 选择，需声明 source-label-assisted development。 |
| D5 | 模型优先级与资源降级顺序 | 同 checkpoint self-teacher 最利于隔离架构作用。7B 为文档基线锚点；如 Lop 不可行，是否接受较小同系列？先减少无关 scaling 与 agent 数，避免把视频观察降到失去时序证据。 |
| D6 | 预算边界 | 日历期限、GPU-hours/并发上限、可用存储及下载上限、工程适配投入上限；是否有尚未列出的 Lop 新 GPU/项目存储。 |
| D7 | 首轮所需结论与实验矩阵 | 最小主结果 T0–T3；若主张可靠度机制，应加 T4a/T4b/T5。若主张 recursion 的独立作用，需追加观察与计算相匹配的平铺对照。 |
| D8 | 随机重复与外层轮数 | 推荐一轮冻结伪标签→训练；主训练优先 3 seeds。1 seed 定位 pilot。教师标签重复与学生训练重复是不同轴；固定一套标签的多 seed 只覆盖学生随机性。 |
| D9 | 主要比较的公平性 | 同候选池、共同可用子集、同学生训练预算；教师侧至少建立 matched-compute 比较并报告实际成本。固定观察信息的对照用于归因 active observation/recursion。 |
| D10 | 主指标与 checkpoint 规则 | 预先选主要 target 的 SRCC 与主要对比（建议 T3−T1），raw PLCC 为辅助；固定训练长度/最终 checkpoint；分数据集报告，按内容组 paired bootstrap。 |

## 必须冻结，但可委托 Codex 提出具体数值的 hyperparameters

这些是完整运行配置的字段，尚未选值；不能直接把文档候选网格视为已批准运行规模。

| 模块 | 必须固定的参数与语义 |
|---|---|
| 数据 | source 数量与独立场景数、各来源比例、按内容分组的 train/dev/diag 划分、去重与跨 source/target 重叠规则。 |
| Pair graph | unique pairs、partners/video、同内容与跨内容桥接比例、度数与连通性、A/B 顺序重复率、固定随机种子。 |
| 模型 | 所有角色及学生 checkpoint/revision、Instruct/Thinking、精度/量化、角色共享方式、训练模块。 |
| 教师规模 | n_adv_pairs（每组是 A/B 两次任务）、角色重复次数、中立观察者数、auditors/claim、hyperjudges、每角色 claims 上限、baseline K。 |
| 观察 | Native video/有序多帧、全局帧数、FPS 与帧上限优先级、分辨率/pixel budget、连续片段时长与数量、crop 数与大小、时间戳、正常播放语境。 |
| 文本 | 各角色 temperature/top_p、输出 token 上限、证据上下文预算、去重与截断策略、prompt 版本。 |
| 递归 | 深度、分支数、节点数、总调用/视觉/token 上限、触发与终止规则、缓存、失败重试上限。 |
| 标签 | 多数票或证据聚合；A/B/near_equal/abstain 编码；最低有效投票数；冲突、平票、顺序不一致与解析失败策略。 |
| 可靠度 | 特征定义、组合公式、缺失项处理、映射 w=f(r)、指数/阈值/截断/下限、是否硬筛选、共同覆盖规则、固定全局归一化。 |
| Reward | 评分尺度与格式、pair-centric/multi-partner、score→preference 概率映射、方差估计与 epsilon、fidelity reward、格式项、零方差及非法输出处理。 |
| GRPO | G、pair/video/token 三种 batch 口径、gradient accumulation、LR、optimizer、schedule/warmup、steps/epochs/token 上限、clip、KL、gradient clipping。 |
| 更新容量 | LoRA/full fine-tuning、rank/alpha/dropout/target modules、vision/projector/text 更新权限、精度。 |
| 测试 | 完整 MOS 对应刺激、帧与像素预算、固定时间采样、解码方式、重复次数与聚合、失败规则、预测冻结后连接 MOS。 |
| 统计 | teacher/train/data seeds、主要假设、bootstrap 单位和次数、CI、dataset/group 指标、teacher 诊断的 MOS 近似相等阈值及敏感性。 |

## 方法定义中须先解决的六个问题

1. **Near-equal 与 abstain**：前者是有证据支持接近，后者是证据不足。候选编码为 A/B/near_equal = 1/0/0.5，abstain 掩码；0.5 是训练目标约定，不是实测人类胜率。基线也必须说明能否产生 tie/abstain，否则支持空间不同。
2. **单条 rollout 的 reward**：如果 G 个输出只得到相同的 pair 级 reward，组内归一化后没有区分信号。必须定义每条评分相对另一视频 rollout 分布/多个 partner 的 reward，再明确 group 边界。
3. **可靠度权重不能被归一化消掉**：放在最终 advantage 标准化后的 policy 项；采用冻结训练集上的全局归一化可用于匹配平均更新强度，不能逐组除以自身公共 weight。KL 独立处理。T5 打乱权重对应关系并保持分布，用于检验样本匹配作用。
4. **Coverage confound**：各方法弃权不同，主对比采用预声明的共同可用子集；各自可用全集作为整体产出视图。不能默默把更少、更容易的样本当作更好的标签。
5. **Compute / observation confound**：agent 数、调用数与 token 数不等价于相同 GPU 成本。需要视觉 token、输入/输出长度、实际 GPU 时间。相同新增视觉材料的平铺 replay 用于检查递归组织本身的作用。
6. **目标分数的含义**：从相对偏好训练出的 scalar 不自动具有跨数据集绝对标定能力；主结果分 target 计算 SRCC/raw PLCC。target MOS 拟合 logistic PLCC 必须另表。Reference-assisted 人类观看协议与 no-reference 模型输入的差别需披露。

## 建议的决策顺序

先确定 D1–D6（数据、监督、模型取舍和预算）→ 提出可行候选与工程验证计划 → 锁定矩阵、预算匹配、主要指标及公式 → 在 Lop 做工程 smoke test → 根据兼容性/显存/吞吐冻结正式数值 → 启动正式实验。工程 smoke 不应利用 target MOS 选择配置。

## 核对过的官方来源

- EvoQuality v5：https://arxiv.org/html/2509.25787v5
- EvoQuality 代码：https://github.com/bytedance/EvoQuality
- Qwen2.5-VL 模型卡：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Explicit-NeRF-QA：https://github.com/YukeXing/Explicit-NeRF-QA
- 3DGS-QA：https://github.com/diaoyn/3DGSQA
- MUGSQA：https://github.com/Solivition/MUGSQA

核查只确认页面所述资源和方法接口；完整数据包、真实 MOS 映射、既有 prototype 和运行兼容性尚未验证。

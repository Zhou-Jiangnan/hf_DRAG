# DRAG 联邦隐私实验中的攻击方式（简述）

本项目当前聚焦两类攻击，目标是评估“训练参数是否泄露隐私”。

## 1) 成员推断攻击（Membership Inference Attack, MIA）

- 思路：攻击者根据模型输出分数，判断某样本是否参与过训练。
- 在本仓库中：
  - 以 `relevant_score` 作为攻击分数；
  - 通过 member/non-member 两组分数计算 AUC；
  - 通过阈值扫描得到 `Attack Advantage = max(TPR - FPR)`。
- 价值：直接衡量“是否能区分训练样本与非训练样本”。

## 2) 更新反演攻击（规划中）

- 思路：攻击者观察参数更新（或梯度），尝试恢复训练样本的关键词、主题或语义信息。
- 当前状态：在实验计划中定义为下一步重点（先做简化版主题/关键词恢复）。
- 价值：衡量参数更新本身是否携带可还原的隐私信息。

## 3) 伪造节点/投毒攻击（易实现，强烈建议）

这类攻击在当前联邦训练框架里**可行且易落地**，因为现在已经有“client=peer、本地训练、服务端聚合”的流程。

### 3.1 模型更新投毒（Model Poisoning）

- 攻击方式：恶意客户端上传异常参数增量（例如放大梯度、反向梯度、随机噪声）。
- 最小实现：
  - 在聚合前，对指定 `malicious_client_ids` 的 `delta` 做变换；
  - 示例：`delta = -alpha * delta`（sign-flip）或 `delta = beta * delta`（amplify）。
- 评测指标：
  - 主任务降幅：EM/F1 下降多少；
  - 收敛稳定性：轮间方差、训练是否发散；
  - 防御有效性：加 clipping 后是否缓解。

### 3.2 后门投毒（Backdoor Poisoning）

- 攻击方式：恶意节点在本地样本里注入触发词（trigger），将触发词样本的目标答案定向到攻击者指定输出。
- 最小实现：
  - 在客户端本地数据预处理阶段添加 trigger；
  - 只污染一小部分比例（如 1%-5%）即可观察效果。
- 评测指标：
  - ASR（Attack Success Rate，后门成功率）；
  - 干净样本性能（Clean Accuracy/F1）；
  - 隐蔽性（触发词长度/频率对性能影响）。

### 3.3 Sybil/伪造节点攻击

- 攻击方式：攻击者控制多个伪造客户端，在采样时提高被选中概率，从而放大投毒影响。
- 最小实现：
  - 构造多个共享同一恶意更新策略的 client id；
  - 提高恶意客户端占比后比较效果。
- 评测指标：
  - 不同恶意比例（10%/20%/30%）下的任务退化曲线；
  - 与鲁棒聚合策略（trimmed mean/median）对比。

### 3.4 Free-rider（搭便车）攻击

- 攻击方式：客户端几乎不训练或上传伪更新，但希望获得全局模型收益。
- 最小实现：
  - 恶意客户端上传零增量或复用旧增量；
  - 统计其本地性能收益与真实贡献偏差。
- 评测指标：
  - 贡献-收益不一致程度；
  - 对全局模型性能/收敛速度影响。

## 4) 附加评测方法（低成本）

- 鲁棒性评测：
  - 恶意客户端比例扫描（0, 10%, 20%, 30%）；
  - 非 IID 强度扫描（topic 偏置程度）。
- 防御评测：
  - `FedAvg` vs `FedAvg+Clipping` vs `Median/Trimmed-Mean`（可先做简化版）；
  - 是否启用 DP 噪声对投毒效果的抑制。
- 代价评测：
  - 通信轮数、额外训练时长；
  - 隐私收益（MIA AUC 下降）与鲁棒收益（ASR 下降）的综合权衡。

## 当前实现位置

- MIA 指标实现：`modules/privacy_attacks.py`
- MIA 运行入口与日志输出：`simulator.py`
- 实验设计说明：`docs/fed_privacy_experiment_timeline.md`、`docs/federated_privacy_ccfc_plan.md`

## 5) CCFC 最小工作量推荐（直接可做）

如果目标是“只做 CCFC 够用的工作量”，建议用 **1 个主攻击 + 1 个鲁棒攻击**：

### A. 必做：MIA（成员推断）

- 原因：实现成本低、复现实验快、和 DP 防御天然匹配。
- 你当前项目已具备：
  - 攻击实现与指标（AUC/Advantage）；
  - 与 `metrics.csv` 的日志打通。
- 论文叙事：证明 DP 对隐私泄露（MIA）的抑制，以及带来的性能代价。

### B. 选做：Model Poisoning（sign-flip）

- 原因：比后门更容易实现，改动小，结果直观。
- 最小改法（结合当前代码）：
  1. 在 `modules/federated_trainer.py` 聚合前，对恶意客户端 `delta` 做 `delta=-alpha*delta`；
  2. 配置里加 `poison_ratio`、`poison_alpha`；
  3. 在 `simulator.py` 记录 poison 开关与比例，输出到 `metrics.csv`。
- 指标：主任务 F1/EM、MIA-AUC、训练稳定性（是否发散）。

### 为什么不建议一开始就做太多攻击？

- 后门攻击需要额外设计 trigger 与 ASR 评测，开发和调参时间更长；
- 更新反演要做重建目标和相似度评价，实验更重；
- CCFC 体量优先保证“结论清晰 + 可复现”，不求攻击面铺太广。

### 最小实验矩阵（建议）

1. Baseline: Centralized-PPO
2. FedAvg-PPO（无 DP）
3. FedAvg-PPO + DP（σ=0.6）
4. FedAvg-PPO + DP（σ=1.0）
5. FedAvg-PPO + sign-flip poisoning（10% 恶意）
6. FedAvg-PPO + sign-flip poisoning + DP（σ=0.6）

这样 6 组实验即可同时回答：
- 联邦是否有效；
- DP 是否降低 MIA；
- 投毒是否威胁系统、DP/裁剪能否缓解。

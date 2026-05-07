# 10分钟组会汇报稿（DRAG 联邦学习 + 隐私保护）

> 目标：把当前工作讲清楚“做了什么、结果怎么评估、接下来怎么做”。

---

## 0. 标题页（30秒）

**题目**：面向 DRAG 路由策略的联邦学习与隐私保护训练

**一句话摘要**：在不改 LLM 本体的前提下，对 PPO 路由器进行联邦训练，加入 DP 防护，并用 MIA 衡量隐私泄露，同时规划投毒鲁棒对比。

---

## 1. 背景与问题定义（1分钟）

- DRAG 当前是分布式检索+路由，但训练流程默认非联邦。
- 目标：实现“数据不出端”的联邦路由训练。
- 核心问题：
  1. 联邦训练会不会明显损失任务性能？
  2. DP 是否能降低隐私泄露（MIA）？
  3. 在攻击场景下是否仍稳定？

---

## 2. 当前系统改造（2分钟）

### 2.1 联邦训练主干

- 新增 `FederatedPPORunner`：
  - client 采样
  - 本地 PPO 更新
  - FedAvg 聚合
  - clip + Gaussian noise（DP）

### 2.2 隐私评估

- 新增 MIA 模块：
  - 输入 member / non-member 分数
  - 输出 `mia_auc`, `mia_attack_advantage`
- 指标接入 `metrics.csv`

### 2.3 工程入口

- `simulator.py` 按 `--rag.enable_federated_privacy` 切换训练分支
- 已有 baseline/ablation 自动脚本可跑

---

## 3. 目前可跑的实验与对比（2分钟）

### 3.1 已有可跑对比

1. Centralized-PPO
2. FedAvg-PPO
3. FedAvg-PPO + DP
4. FedAvg-PPO + DP + MIA 评估

### 3.2 建议补齐（最小增强）

5. FedAvg-PPO + Robust Aggregation（median/trimmed-mean）
6. FedAvg-PPO + sign-flip poisoning
7. FedAvg-PPO + sign-flip poisoning + 防御（DP 或 RobustAgg）

> 这 7 组基本可以形成“性能-隐私-鲁棒”闭环论证。

---

## 4. 评测口径与公平性（1.5分钟）

- 统一协议：同数据、同 seed、同预算（rounds × local_epochs）
- 指标：
  - 任务：F1/EM、hops/messages
  - 隐私：MIA-AUC、Attack Advantage
  - 鲁棒：攻击后性能降幅（可选 ASR）
- 统计要求：
  - 至少 3 seeds（0/1/2）
  - 报告 mean ± std
  - 给一张 Pareto 图（`mia_auc` vs `f1`）

---

## 5. 参考方法与可信度增强（1分钟）

可借鉴（非RAG也可迁移）：

- DDFed（NeurIPS 2024）
- RFLPA（NeurIPS 2024）
- RL-based aggregation defense（OpenReview 2024）
- FedRoLA（KDD 2024）
- ClusterGuard（2024）

汇报策略：
- 不强求全栈复现外部系统；
- 采用“机制等价对比”并明确限制。

---

## 6. 结论与下一步（2分钟）

### 6.1 当前结论

- 联邦 + DP + MIA 的最小系统已打通；
- 已具备做隐私-性能 trade-off 的实验基础；
- 目前短板在“鲁棒聚合与投毒对照”尚未完全补齐。

### 6.2 两周计划

- Week1：跑完统一协议主实验 + 三组消融
- Week2：补 sign-flip + robust aggregation + 图表/写作

### 6.3 预期产出

- 一张主结果表（性能+隐私+鲁棒）
- 两张关键图（Pareto / 攻防对照）
- 一版可投稿初稿（CCFC/CSCloud 方向）

---

## 附：汇报者提示词（可直接念）

- “我们的目标不是单点 SOTA，而是建立可复现的隐私-性能-鲁棒权衡框架。”
- “在统一协议下做机制等价对比，比强行复现不兼容系统更可靠。”
- “下一步工作聚焦最小增强：robust aggregation + poisoning 对照。”


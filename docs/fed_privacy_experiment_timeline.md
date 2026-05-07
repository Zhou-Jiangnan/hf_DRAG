# DRAG 联邦隐私版本：可跑通性、基线与实验计划

## 1) 现在这版代码能不能跑通？

结论：**能跑通“联邦 PPO + DP + MIA”主流程**，但建议先做 2 个 smoke 阶段：

### 阶段 A：纯功能 smoke（最快）

- `enable_federated_privacy=true`
- `fed_rounds=1`
- `fed_local_epochs=1`
- `fed_client_fraction=0.2`
- `test_mode=true`, `test_num_samples=20`
- `fed_privacy_attack_eval=true`
- `fed_mia_max_samples=20`

目的：验证训练、聚合、推理、MIA 统计和 `metrics.csv` 字段都能走通。

### 阶段 B：小规模稳定性 smoke

- `fed_rounds=10`
- `fed_client_fraction=0.5`
- `fed_dp_mechanism=dp_fedavg`
- `fed_dp_noise_multiplier=0.6`

目的：确认多轮训练下无异常、MIA 指标有趋势变化。

---

## 2) 效果对比基线（建议固定 6 组）

为了突出“隐私保护训练参数”的贡献，建议只在 **PPO 路由**上比较：

1. **C1: Centralized-PPO**（原始单机 PPO，无联邦、无 DP）
2. **F1: FedAvg-PPO**（联邦，无 DP）
3. **F2: FedAvg-PPO + Clip only**
4. **F3: FedAvg-PPO + DP (σ=0.3)**
5. **F4: FedAvg-PPO + DP (σ=0.6)**
6. **F5: FedAvg-PPO + DP (σ=1.0)**

> 若有精力可再补：`FedProx + DP`（作为扩展，不放主结果也可）。

---

## 3) 实验设计（主实验 + 消融）

## 实验 E1：主结果（隐私-效能权衡）

- 比较：C1/F1/F2/F3/F4/F5
- 任务指标：`exact_match`, `f1`, `avg_num_hops`, `avg_num_messages`
- 隐私指标：`mia_auc`, `mia_attack_advantage`
- 目标：画 Pareto（隐私 vs 任务性能）

## 实验 E2：DP 强度消融

- 固定 FedAvg，改变 `fed_dp_noise_multiplier ∈ {0.0, 0.3, 0.6, 1.0, 1.4}`
- 观察 AUC 是否向 0.5 下降、同时任务性能下降幅度

## 实验 E3：客户端参与率消融

- 固定 DP 配置，改变 `fed_client_fraction ∈ {0.2, 0.5, 1.0}`
- 观察通信/稳定性/性能变化

## 实验 E4：本地轮数消融

- `fed_local_epochs ∈ {1, 2, 4}`
- 观察本地过拟合是否提高 MIA 风险

## 实验 E5：数据异构（non-IID）

- IID 与 topic 偏置 non-IID 各做一套
- 核心看：non-IID 下隐私与性能是否更差

## 实验 E6：跨数据集泛化（可选）

- 选 2 个数据集（如 MMLU + medical）重复 E1
- 目标：证明方法不依赖单一数据集

## 实验 E7：恶意节点投毒（可选但推荐）

- 攻击：model poisoning（sign-flip / amplify）或后门 trigger 注入
- 防御：clipping / DP / robust aggregation（median 或 trimmed mean）
- 指标：任务性能、ASR、MIA-AUC（观察防御是否两头兼顾）

---

## 4) 最少做几个实验够 CCFC

如果时间紧，建议最小包：

- 必做：E1 + E2 + E5（3 个）
- 有余力：加 E3（第 4 个）

这样已经能完整回答：
- 联邦是否有效（F1 vs C1）
- DP 是否保护隐私（AUC/Advantage）
- 代价多大（性能/开销）
- 异构场景是否成立（IID vs non-IID）

---

## 5) 时间线（3 周版本）

## Week 1：可复现实验管线

- Day 1-2：完成 smoke（A/B）
- Day 3：跑 C1/F1/F2
- Day 4-5：跑 F3/F4/F5
- 交付：主结果原始 `metrics.csv` + 清洗脚本

## Week 2：消融与异构

- Day 1-2：E2（DP 强度）
- Day 3：E3（客户端参与率）
- Day 4-5：E5（IID vs non-IID）
- 交付：关键图表初版（2-3 张）

## Week 3：收敛与论文材料

- Day 1-2：补实验（失败重跑）
- Day 3：统计显著性与误差条
- Day 4：写实验章节
- Day 5：整理最终表格与附录命令

---

## 6) 建议的结果表结构

- 表 1（主结果）：方法 × {EM/F1/Hops/Messages/MIA-AUC/Adv}
- 图 1：MIA-AUC vs F1
- 图 2：噪声强度 vs (F1, AUC)
- 图 3：IID vs non-IID 的方法对比柱状图

---

## 7) 复现实验的关键参数（建议固定）

- `rag.random_seed`: 至少 3 个种子（0/1/2）
- `rag.query_ttl`: 固定
- `rag.num_peers`: 固定
- `rag.ppo_train_episodes` 或 `fed_rounds * fed_local_epochs`: 保持同量级

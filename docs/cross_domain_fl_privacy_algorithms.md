# 可借鉴的“非RAG领域”联邦学习隐私保护算法（面向当前 DRAG 路由FL）

## 1) 结论

你的对象虽然是“联邦路由算法”，但可直接借鉴大量**跨领域** FL 隐私/鲁棒方法。建议优先借鉴以下 4 条路线：

1. **DP 优化路线**（client-level DP）
2. **鲁棒聚合路线**（抗投毒）
3. **SecAgg + 鲁棒联合路线**（密文场景可防投毒）
4. **学习型聚合路线**（RL-based aggregation）

---

## 2) 近三年可参考方法（不限RAG）

### A. DDFed（NeurIPS 2024）

- 核心：同时处理隐私保护与投毒鲁棒（dual defense）。
- 适配价值：可直接映射到你当前“DP + 攻击评估”框架，作为 stronger baseline。
- 参考：
  - Paper PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/824f98c4e9bb301b0b96fa2ae071360b-Paper-Conference.pdf

### B. RFLPA（NeurIPS 2024）

- 核心：在 Secure Aggregation 约束下仍做抗投毒。
- 适配价值：你当前有 `fed_secure_agg` 开关，RFLPA思路可用于后续增强。
- 参考：
  - Paper PDF: https://papers.neurips.cc/paper_files/paper/2024/file/bcbdc25dc4f0be5ae8ac07232df6e33a-Paper-Conference.pdf

### C. RL-based Aggregation Defense（OpenReview 2024）

- 核心：用强化学习动态学聚合规则，应对复杂投毒。
- 适配价值：你本身就是 PPO 路由，和 RL 思路同构，改造成本低于传统项目。
- 参考：
  - OpenReview: https://openreview.net/forum?id=di03SQhH88

### D. FedRoLA（KDD 2024）

- 核心：分层/相似性聚合，提高对模型投毒鲁棒性。
- 适配价值：可作为你 `FedAvg -> RobustAgg` 的候选实现。
- 参考：
  - KDD 2024 entry (institution page): https://researchwith.stevens.edu/en/publications/fedrola-robust-federated-learning-against-model-poisoning-via-lay/

### E. ClusterGuard（2024）

- 核心：安全聚合与鲁棒聚合联合，解决“只加密不鲁棒”问题。
- 适配价值：为 `secure_agg=true` 场景提供路线。
- 参考：
  - ePrint: https://eprint.iacr.org/2024/2082

---

## 3) 如何落地到当前项目（最小改造）

### Step 1：先补 Robust Aggregation（优先）

在 `modules/federated_trainer.py` 的聚合位置增加：

- coordinate-wise median
- trimmed mean（去头去尾比例可配）

并增加配置：
- `fed_agg_method`: fedavg / median / trimmed_mean
- `fed_trim_ratio`

### Step 2：补最小投毒攻击

- 在本地更新汇总前加入恶意客户端变换：`delta = -alpha * delta`
- 配置：`poison_enabled`, `poison_ratio`, `poison_alpha`

### Step 3：把 SecAgg 场景单独成组

- 组A：`secure_agg=false`
- 组B：`secure_agg=true`
- 对比同一攻击强度下指标变化（F1 / MIA-AUC / 攻击成功率）

### Step 4：补“学习型聚合”轻量版（可选）

- 用一个小策略网络决定每轮客户端权重（非必须端到端复现论文）
- 先在模拟环境验证是否优于 FedAvg/median

---

## 4) 推荐对比矩阵（够发表且工作量可控）

1. FedAvg
2. FedAvg + DP
3. Median + DP
4. TrimmedMean + DP
5. FedAvg + Poison
6. Median + Poison
7. Median + Poison + DP

> 这组矩阵是“跨领域算法迁移到你项目”的最小有力证据链。

---

## 5) 为什么这样做有说服力

- 不是只和“自己方法”比，而是和**跨领域经典防御机制**比。
- 即使外部方法系统不兼容，也可做“机制等价对比”。
- 你已有 MIA 与 DP 基础，新增聚合策略后，可信度会明显提升。


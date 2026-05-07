# DRAG 联邦隐私改造：CCFC 体量筛选总结 + 组会汇报大纲

## 1. 先给结论（筛选后的最小可发表方案）

> 目标：在可控工作量内，讲清楚“隐私-性能-鲁棒性”三者关系。

推荐只做两类攻击：

1. **MIA（成员推断，必做）**
2. **Model Poisoning（sign-flip，选做但强烈推荐）**

对应防御只保留两种：

- DP（当前已支持：clip + Gaussian）
- 可选鲁棒聚合（median/trimmed-mean，先做其中一种）

这样最少 6 组实验就能形成完整论证链：

- C1 Centralized-PPO
- F1 FedAvg-PPO
- F2 FedAvg-PPO + DP(σ=0.6)
- F3 FedAvg-PPO + DP(σ=1.0)
- A1 FedAvg-PPO + sign-flip poisoning(10%)
- A2 FedAvg-PPO + sign-flip poisoning(10%) + DP(σ=0.6)

---

## 2. 如何结合当前项目（按改动优先级）

### P0（已有）

- 联邦训练主干：`modules/federated_trainer.py`
- MIA：`modules/privacy_attacks.py`
- 指标入库：`simulator.py -> metrics.csv`

### P1（建议新增，工作量可控）

- 在聚合前增加投毒注入钩子：
  - 指定恶意客户端比例 `poison_ratio`
  - 对恶意客户端执行 `delta = -alpha * delta`
- 新增日志字段：`poison_ratio`, `poison_alpha`, `is_poison_enabled`

### P2（有余力）

- 加一个鲁棒聚合分支（median 或 trimmed-mean）
- 对比 `FedAvg vs FedAvg+RobustAgg vs FedAvg+DP`

---

## 3. 近3年可结合的开源/可复现论文与方法（用于增强可信度）

下面按“你这个项目最容易接入”的优先顺序给出：

1. **LeadFL (ICML 2023)**：客户端侧防御投毒，适合与现有服务端防御组合。  
   Link: https://proceedings.mlr.press/v202/zhu23j.html

2. **Turning Privacy-preserving Mechanisms against FL (CCS 2023)**：说明“仅靠隐私机制仍可能被攻击绕过”，可作为威胁动机。  
   Link: https://arxiv.org/abs/2305.05355

3. **DataDefense (arXiv 2023, v2 2024)**：基于少量外部防御样本的投毒防御思路，可借鉴“客户端重要性加权”。  
   Link: https://arxiv.org/abs/2305.02022

4. **OpenFedLLM (2024)**：开源联邦训练框架（含多种 FL 算法），可作为工程可复现背书与扩展方向。  
   Link: https://arxiv.org/abs/2402.06954

5. **Binary FL with Client-level DP (GLOBECOM 2023)**：关注 client-level DP 和通信开销，可用于 DP 机制讨论与对照。  
   Link: https://arxiv.org/abs/2308.03320

6. **DDFed (NeurIPS 2024 Poster)**：隐私保护与抗投毒联合防御范式，可作为你论文“下一步工作”或高阶对比。  
   Link: https://openreview.net/forum?id=EVw8Jh5Et9

7. **DPFL Systematic Review (2024/2025 revision)**：作为 related work 与 taxonomy 依据，增强综述可信度。  
   Link: https://arxiv.org/abs/2405.08299

> 说明：以上条目里，1/2/3/4/5/7 都可直接从论文页获取 PDF；4 明确提到提供 codebase，适合直接复现。

---

## 4. 评测指标（建议最小集合）

任务性能：
- `f1`, `exact_match`, `avg_num_hops`, `avg_num_messages`

隐私：
- `mia_auc`, `mia_attack_advantage`

鲁棒性：
- Poison 场景下主任务降幅
- 若做后门则加 `ASR`（可选，不是 CCFC 必需）

效率（可选）：
- 每轮训练耗时
- 通信量估计（参数量 × 轮数 × 客户端数）

### 建议补充的消融（提升可信度，工作量仍可控）

1. **DP 强度消融**：`fed_dp_noise_multiplier ∈ {0.0, 0.3, 0.6, 1.0, 1.4}`  
2. **客户端参与率消融**：`fed_client_fraction ∈ {0.2, 0.5, 1.0}`  
3. **本地训练轮数消融**：`fed_local_epochs ∈ {1, 2, 4}`  

建议标准：
- 每个设置至少 3 个随机种子（0/1/2）；
- 报告均值 ± 标准差；
- 至少给一张 Pareto 图（`mia_auc` vs `f1`）。

---

## 5. 组会汇报清单大纲（可直接照着讲）

### A. 问题定义（2页）
- DRAG 为什么要做 FL + 隐私
- 现有风险：MIA、投毒

### B. 系统改造（3页）
- 当前代码架构图（client=peer）
- 联邦训练流程（本地训练→聚合）
- 隐私与攻击模块位置

### C. 实验设计（3页）
- 6 组最小实验矩阵
- 数据划分（IID/non-IID）
- 指标定义与记录方式

### D. 结果与分析（4页）
- 主结果表：性能 + MIA
- DP 噪声对隐私/性能曲线
- 投毒场景下防御效果

### E. 可信度与对比文献（2页）
- 近3年方法映射到本项目
- 你和已有工作的差异点

### F. 结论与下一步（1页）
- 当前结论三条
- 下一步（鲁棒聚合 / backdoor / 反演）

---

## 6. 执行时间线（3周）

- **Week1**：跑通 6 组主实验（含 MIA）
- **Week2**：补 non-IID + poison 组
- **Week3**：整理图表、写作、补充 ablation

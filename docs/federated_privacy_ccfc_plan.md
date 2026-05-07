# DRAG 联邦学习 + 隐私保护改造方案（CCFC 工作量）

> 目标：在**尽量小改动**前提下，为 DRAG 增加联邦训练与隐私保护实验能力，并给出可发表级别（CCFC 体量）的对比与攻击评估框架。

## 1. 先定边界：你要“联邦化”的对象

DRAG 项目里最合适联邦化的是路由策略（PPO/GRPO Router），而不是 LLM 本体：

- 当前路由器训练入口在 `simulator.py`，并调用 `DRAGNetwork.ppo_train/grpo_train`。
- 路由器是一个轻量策略网络，参数体积远小于大模型，适合在多节点模拟联邦训练。
- 每个 peer 本来就有局部知识库，天然符合“数据不出端”。

**建议**：做“联邦路由训练 + 本地知识不共享”。

---

## 2. 最小代码改造路径（按实现顺序）

### Step A：配置层（已具备占位）

在 `rag` 配置下增加联邦隐私参数（已在本仓库占位）：

- `enable_federated_privacy`
- `fed_rounds`
- `fed_local_epochs`
- `fed_client_fraction`
- `fed_dp_mechanism`
- `fed_dp_clip_norm`
- `fed_dp_noise_multiplier`
- `fed_secure_agg`
- `fed_privacy_attack_eval`

这些参数可以先只用于实验开关，不必一次性全实现。

### Step B：新增联邦协调器模块（核心新增）

新增 `modules/federated_trainer.py`，包含：

1. `ClientUpdate` 数据结构：`client_id`, `delta`, `num_samples`, `metrics`。
2. `FederatedCoordinator`：
   - `sample_clients(fraction)`
   - `broadcast(global_weights)`
   - `local_train(client_id, local_data, local_epochs)`
   - `aggregate(updates, method="fedavg")`
3. DP 插件接口：
   - `clip_update(delta, clip_norm)`
   - `add_noise(delta, sigma)`

本项目建议先做 `FedAvg + (Clip + Gaussian Noise)`，后续再扩展。

### Step C：对接现有训练循环

在 `simulator.py` 中替换单机 `ppo_train/grpo_train` 分支：

- 当 `enable_federated_privacy=false`：保持现有流程不动。
- 当 `enable_federated_privacy=true`：
  1. 将 peers 划分为客户端（可 1 peer = 1 client）。
  2. 每轮下发全局路由器参数。
  3. 客户端在本地调用现有的路由训练逻辑（建议封装成 `train_one_client_epoch`）。
  4. 上传参数差分（或梯度）到聚合端。
  5. 聚合端执行 Clip/Noise（可在客户端或服务端实现，先固定一种）。
  6. 产出新全局参数，进入下一轮。

### Step D：攻击评估模块

新增 `modules/privacy_attacks.py`，先做 2 类高价值攻击：

1. **Membership Inference (MIA)**：判断样本是否参与训练。
2. **Gradient/Update Inversion（简化版）**：尝试从更新恢复关键词/主题。

实现上优先“可重复、可量化”而非最强攻击。

---

## 3. 对比方法怎么选（建议 6 组，够 CCFC）

聚焦“隐私-性能权衡”，避免方法太散。

1. **Centralized-DRAG**（非联邦，无 DP）
2. **FedAvg-DRAG**（联邦，无 DP）
3. **FedAvg + Clip only**
4. **FedAvg + DP (Gaussian)**
5. **FedAvg + DP + Secure Aggregation（可模拟）**
6. **FedProx + DP**（可选；如果时间紧可不做）

### 检索路由算法维度（建议固定，避免爆炸）

- 主实验：仅 PPO（或仅 GRPO，二选一）。
- 附录/补充：TARW/RW/FL 只做推理对比，不做联邦训练。

这样能控制工作量，同时突出“隐私保护训练参数”的贡献。

---

## 4. 攻击方式设计（重点：隐私）

### 攻击面 1：成员推断（首选主结果）

- 攻击者输入：模型输出置信度、loss proxy、路由命中模式等。
- 训练 shadow attack model，判断样本是否在某客户端训练集。
- 指标：AUC、TPR@FPR=1%、Attack Advantage。

**期望现象**：
- 无 DP 时 AUC 明显高于随机。
- 增大噪声后 AUC 接近 0.5，但主任务性能下降。

### 攻击面 2：梯度/更新反演（次主结果）

- 攻击者观测客户端更新（或聚合前更新，白盒设定）。
- 优化一个候选输入使其梯度接近观测更新。
- 用文本嵌入相似度评估“恢复程度”，避免强求还原完整句子。

指标建议：
- Top-k topic hit rate
- 关键词恢复 F1
- 嵌入余弦相似度

### 攻击面 3（可选）：属性推断

- 推断样本 topic/domain（医疗、新闻等）。
- 指标：macro-F1。

---

## 5. 实验指标与画图（论文友好）

主任务效能：

- QA 准确率 / F1
- 平均 hop、消息数（已有）

隐私指标：

- ε（若用 RDP accountant）
- MIA AUC / Advantage
- 反演恢复分数

系统指标：

- 每轮通信量
- 训练时长

图表建议：

1. 隐私-效能 Pareto 曲线（x: AUC 或 ε，y: QA）
2. 噪声强度 vs 性能/攻击成功率
3. 客户端异质性（non-IID）下的稳健性折线图

---

## 6. 数据划分与 non-IID 设计

DRAG 的 topic 非常适合做联邦异构：

- **IID**：每客户端随机采样各 topic。
- **non-IID-1（标签偏置）**：每客户端只保留少数 topic。
- **non-IID-2（数量偏置）**：客户端样本量长尾分布。

至少报告 IID + non-IID-1，两种即可满足 CCFC 工作量。

---

## 7. 工程实现优先级（两周冲刺版）

### P0（必须）

- 联邦训练主循环（FedAvg）
- DP-SGD/DP-FedAvg 的 clip+noise
- MIA 攻击评估脚本

### P1（建议）

- Secure Aggregation 模拟开关
- non-IID 划分脚本
- 自动化 ablation 脚本

### P2（有余力再做）

- FedProx/SCAFFOLD
- 反演攻击增强版

---

## 8. 你可以直接照着改的文件清单

- `modules/options.py`：新增联邦隐私配置项（已占位）。
- `config/rag.yaml`：新增默认联邦隐私参数（已占位）。
- `simulator.py`：加入 `enable_federated_privacy` 分支。
- `modules/federated_trainer.py`：新增联邦协调器与聚合。
- `modules/privacy_attacks.py`：新增 MIA/反演评估。
- `modules/exp_logger.py`：记录隐私指标（AUC、epsilon、attack_advantage）。

---

## 9. 推荐论文叙事（CCFC 够用）

一句话贡献模板：

1. 提出 DRAG 下的联邦路由训练框架，数据不出端；
2. 在参数更新中加入 DP 机制与（可选）安全聚合；
3. 构建成员推断与更新反演攻击评估，量化隐私-效能权衡。

只要把对比和攻击评估做扎实，这个体量对 CCFC 是够的。

---

## 10. 常见问题（针对当前 DRAG 代码）

### Q1：改成联邦学习后，需要改 DRAG 的网络结构吗？

不需要强制改图结构（Barabási-Albert 拓扑可保留），优先改“训练编排层”。  
也就是：查询路由的图结构不变，但训练从单机 PPO 改为“多客户端本地更新 + 服务器聚合”。

### Q2：当前 DRAG 本地运行，怎么模拟分布式节点和知识？

在本项目里可直接采用“**client=peer**”模拟：  

- 每个 `peer` 作为一个联邦客户端；
- `peer_topics` 对应客户端可见知识域；
- 客户端本地训练时，只使用该 peer topic 下的样本；
- 服务器端只聚合路由器参数更新，不接触原始样本。

这在单机环境就能模拟“数据不出端”的分布式训练过程。

### Q3：PPO 路由如何适配新的联邦 DRAG 架构？

适配要点：

1. 全局初始化一份 PPO Router 权重；
2. 每轮下发给采样客户端；
3. 客户端在本地调用现有 `ppo_train`（限制 `query_peer_ids=[client_id]`）；
4. 上传本地参数差分 `delta`；
5. 服务端执行 FedAvg +（可选）DP clipping/noise；
6. 得到新全局权重进入下一轮。

本仓库已按该思路新增 `modules/federated_trainer.py` 和 `simulator.py` 分支入口。

### 当前实现进度（隐私评估）

- 已新增 `modules/privacy_attacks.py`，实现阈值型 MIA 评估；
- 已在 `simulator.py` 中接入 `mia_auc`、`mia_attack_advantage` 等字段到 `metrics.csv`；
- 通过 `rag.fed_mia_holdout_ratio` 与 `rag.fed_mia_max_samples` 控制 MIA 样本划分与评估规模。

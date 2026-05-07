# Federated RAG 对比方法与落地路线（面向当前 DRAG 项目）

## 1) 先回答问题：有没有可参考的开源方法？

有，而且近两年开始增多。至少可以确认：

- **FedRAG (Vector Institute, 开源代码)**：支持 centralized/federated 的 RAG 微调框架，可直接作为工程对照。  
  Repo: https://github.com/VectorInstitute/fed-rag

- **FedE4RAG (2025)**：隐私保护联邦嵌入学习（Localized RAG），可作为“隐私检索侧”方法参考。  
  Paper: https://arxiv.org/abs/2504.19101

- **FedMosaic (2026)**：参数高效适配器路线（adapter-based federated RAG），可作为“低通信量”扩展方向。  
  Paper: https://arxiv.org/abs/2602.05235

> 结论：不是“完全没有”，但整体仍是新方向，你现在做 DRAG-FL+DP+攻击评估仍有发表空间。

---

## 2) 针对本项目，建议优先对比的方法（按实现难度排序）

### M1. FedAvg-PPO（你当前已具备）

- 用作联邦基线。
- 与 centralized PPO 对比，体现“联邦代价”。

### M2. FedAvg-PPO + DP（你当前已具备）

- 通过 clip + noise 展示隐私-性能权衡。

### M3. FedAvg-PPO + Robust Aggregation（建议新增）

- 在服务端聚合端引入 median 或 trimmed-mean。
- 目标：提高抗投毒鲁棒性。

### M4. FedAvg-PPO + Poisoning（建议新增）

- 最小实现用 sign-flip model poisoning。
- 与 M3/M2 组合，形成攻击-防御闭环。

### M5. Adapter-based Federated Route Tuning（可选）

- 借鉴 FedMosaic 思路，只同步小参数（adapter/LoRA）以降通信成本。

---

## 3) 可信度增强：建议加入的公开评测维度

- 主任务：EM/F1、avg_num_hops、avg_num_messages
- 隐私：MIA-AUC、Attack Advantage
- 鲁棒：Poison 场景下性能降幅（可选 ASR）
- 效率：每轮训练时间、通信估算

---

## 4) 实现方式与操作步骤（直接执行版）

### Step A：固定主实验矩阵（6 组）

1. Centralized-PPO
2. FedAvg-PPO
3. FedAvg-PPO + DP(σ=0.6)
4. FedAvg-PPO + DP(σ=1.0)
5. FedAvg-PPO + sign-flip poisoning(10%)
6. FedAvg-PPO + sign-flip poisoning(10%) + DP(σ=0.6)

### Step B：跑消融（你已具备脚本）

- 噪声强度、client fraction、local epochs 三组消融。
- 每组 3 seeds（0/1/2），记录均值±标准差。

### Step C：补最小投毒实现（1-2 天）

- 在 `modules/federated_trainer.py` 聚合前加入恶意客户端更新变换：
  - `delta = -alpha * delta`
- 配置新增：`poison_ratio`, `poison_alpha`, `poison_enabled`
- 在 `simulator.py` 将投毒参数写入 `metrics.csv`

### Step D：产出组会/论文必备图表

- 图1：主结果表（任务+隐私）
- 图2：MIA-AUC vs F1（Pareto）
- 图3：攻击前后对比（Poison vs Defense）

---

## 5) 推荐引用（近3年，和你课题强相关）

1. FedRAG codebase (Vector Institute): https://github.com/VectorInstitute/fed-rag
2. Privacy-Preserving Federated Embedding Learning for Localized RAG (FedE4RAG), 2025: https://arxiv.org/abs/2504.19101
3. FedMosaic: Federated RAG via Parametric Adapters, 2026: https://arxiv.org/abs/2602.05235
4. Federated RAG Systematic Mapping Study (Findings EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.388.pdf


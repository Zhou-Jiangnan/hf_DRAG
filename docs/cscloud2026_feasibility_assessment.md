# CSCloud 2026 投稿可行性严格评估（基于当前 DRAG-FL 隐私方案）

## 1) 结论先行

- **主题匹配：高**（你的课题与 CSCloud 的 cloud security / privacy / AI-enabled security 主题直接重合）。
- **工作量是否够投：够投，但离“稳中”还差 2-3 个关键增强项**。
- **当前中稿概率（主观区间）**：
  - 现状直接投：**35%–55%**
  - 补齐建议后再投：**55%–75%**

> 说明：CSCloud 官方页面未公开固定 acceptance rate，本区间是基于该会主题宽度、历年程序规模、以及你当前方案完整度做的工程化估计，不是官方数据。

---

## 2) 与 CSCloud 2026 主题要求的对齐度

根据 CSCloud 2026 CFP，会议重点包括：

- Security and Privacy on Clouds with AI
- Cloud users’ privacy information protection
- Emerging attack methods in cloud/fog/edge computing
- Reinforcement learning-based security mechanism

你的项目对应关系：

- DRAG 路由（PPO/联邦化）= RL-based mechanism
- DP + MIA = privacy protection + attack evaluation
- Poisoning 扩展 = emerging attack method

结论：**方向上完全在题内**。

---

## 3) 当前方案的优势与短板

### 优势（已具备）

1. 有完整工程链路：FL 训练、DP 机制、MIA 指标、日志与脚本。
2. 有可复现实验入口：baseline + ablation 脚本。
3. 问题定义清晰：隐私-性能权衡 + 鲁棒性。

### 短板（影响中稿概率）

1. **缺少更强对比基线**（例如 robust aggregation）。
2. **统计显著性不足风险**（如果只有单 seed）。
3. **攻击面略单薄**（只有 MIA 时，安全故事线不够“厚”）。

---

## 4) 为了 CSCLOUD2026 更稳，最小增强清单（优先级）

### P0（必须补，1周内可完成）

- 3 seeds（0/1/2）复现实验，报告 mean±std。
- 完成 3 组关键消融：
  - DP 噪声强度
  - client fraction
  - local epochs
- 输出至少 1 张 Pareto 图（MIA-AUC vs F1）。

### P1（强烈建议，1周）

- 增加一种轻量投毒（sign-flip）+ 一种防御（clipping/robust agg 二选一）。
- 在主表中给出“无攻击 / 有攻击”并排结果。

### P2（可选，锦上添花）

- non-IID 强度实验（至少 2 档）。
- 通信成本估算（轮数 × 客户端 × 参数量）。

---

## 5) 评审视角下的“中稿风险点”与规避

### 风险点 A：创新性被认为偏工程整合

- 规避：强调“DRAG 场景下的隐私-鲁棒统一评估框架”，并给出攻击-防御对照图。

### 风险点 B：实验不充分

- 规避：最少 6 组主实验 + 3 组消融 + 3 seeds。

### 风险点 C：缺少可信对标

- 规避：引入近3年相关工作（LeadFL / OpenFedLLM / DDFed 等）做对照与定位。

---

## 6) 推荐投稿策略

1. **先做稳健版短文目标**：先保证可复现、图表完整、结论闭环。
2. **标题聚焦**：突出“Federated RL Routing + Privacy Leakage Evaluation”。
3. **摘要不贪多**：一条主贡献（隐私-性能权衡）+ 一条安全扩展（poisoning）。
4. **准备 Plan-B**：若主会不稳，可同步准备 workshop/special track 版本。

---

## 7) 你现在该做什么（执行顺序）

- 第1步：跑完 ablation 脚本并汇总 3 seeds。
- 第2步：补 sign-flip poisoning 最小实现。
- 第3步：出 3 张核心图（主结果、Pareto、攻击防御对照）。
- 第4步：按“问题-方法-实验-威胁模型-结论”写稿。


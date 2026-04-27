# Rung 5 MTL + MoE 深度理解 Audit

> 2026-04-24 code-council + stats-professor 深审合集
> 简体中文 + English jargon
> 配套 plan 文件 `~/.claude-eva/plans/i-want-you-to-imperative-badger.md`

---

## 目录

1. Kendall Uncertainty-Weighted MTL Loss —— 公式 + 为什么单任务退化成 MSE
2. Negative Transfer —— 为什么 uncertainty weighting 修不了
3. Regime-Gated MoE 架构与 loss
4. MoE 为什么没 work —— Shazeer 2017 警告
5. HMM Underidentified 问题
6. 关键 Hyperparameter 清单（所有 Rung 5 变体）
7. Fix 优先级排序（交叉链接 plan Tier C）

---

## 1. Kendall Uncertainty-Weighted MTL Loss

### 1.1 公式

定义在 `main.py:154-246` 的 `UncertaintyMTLLoss` 类。对每个 active task $k$：

$$
\mathcal{L}_k = \frac{1}{2} e^{-s_k} \cdot \frac{\text{MSE}_k}{\text{Var}(y_k^{\text{train}})} + \frac{1}{2} s_k
$$

- $s_k = \log \sigma_k^2$ 是可学 parameter（`nn.Parameter`），初始化 0 (`main.py:183-187`)
- $\exp(-s_k)$ 是 **precision**（逆方差），loss 里 task k 的权重
- $\text{Var}(y_k)$ 每个 fold 用 **训练子集**（剥离 val tail 后）计算一次，在后续训练过程中是常量
- $0.5 \cdot s_k$ 是正则项，阻止 $s_k \to -\infty$（否则 precision 会爆炸）
- 多任务合并：**sum** 不是 mean（`main.py:221, 228, 234`）

**引用**：Kendall, Gal, Cipolla (CVPR 2018), "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"

### 1.2 NaN 处理

- `ret3m` 在 training window 最后 3 个月是 NaN（future return 未观察）→ `valid = ~torch.isnan(y_ret3m)` mask (`main.py:224-228`)
- `vol` 类似处理 (`main.py:230-235`)
- `ret` 不 mask，假设从来不缺

### 1.3 关键退化：`n_tasks=1` 时等于 scaled MSE

当 `active_tasks = {"ret"}`（也就是 Rung 4，之前标为 `5a`），loss 公式展开：

$$
\mathcal{L} = 0.5 \cdot e^{-s_{\text{ret}}} \cdot \frac{\text{MSE}_{\text{ret}}}{\text{Var}(y_{\text{ret}})} + 0.5 \cdot s_{\text{ret}}
$$

对 $s$ 求偏导设零：$\frac{d\mathcal{L}}{ds} = -0.5 \cdot e^{-s^*} \cdot L_{\text{ret}} + 0.5 = 0$

解得 $e^{-s^*} = 1 / L_{\text{ret}}$，代回得：

$$
\mathcal{L}^* = 0.5 + 0.5 \cdot \log(L_{\text{ret}})
$$

这是 $L_{\text{ret}}$ 的**单调变换**。**共享层梯度方向跟 pure MSE 完全一样**，log-sigma 的学习除了加一个可训练 scalar 外没有任何改变 model 参数最优解的贡献。

**实际意义**：Rung 4 = normalized MSE + 一个 dead 参数 `s_ret`。Kendall uncertainty 机制**是为多任务平衡设计的**，单任务时形同虚设。这点在 report 里应该明确说明，不要假装 uncertainty 对单任务有什么意义。

### 1.4 Variance normalization 的 regime-sensitivity 问题

`var_ret = float(y_ret_tr.var())` per fold (`main.py:203`)。目的：让 MSE 变 scale-free，不同 fold 可比。

**问题**：
- 高波动 fold（如 2020 COVID 期训练数据）→ `var_ret` 大 → loss 被 shrink → 梯度小 → 更新慢
- 低波动 fold → `var_ret` 小 → loss 被放大 → 梯度猛 → 可能 overstep

这跟 volatility targeting 的建议方向**正好相反**（vol targeting 要求在波动高时加大信号强度）。

**缓解**：可考虑 rolling / EWMA variance estimate，或用长期平均 variance 替代 per-fold。

---

## 2. Negative Transfer —— 为什么 Uncertainty Weighting 修不了

### 2.1 机制层面

Uncertainty weighting 只调整 loss **magnitude**，不能翻转 **gradient direction**。

考虑 shared encoder 参数 $\theta$：
- `ret` task 梯度：$g_{\text{ret}} = \nabla_\theta \mathcal{L}_{\text{ret}}$
- `ret3m` task 梯度：$g_{\text{3m}} = \nabla_\theta \mathcal{L}_{\text{3m}}$
- 合并：$\nabla_\theta \mathcal{L}_{\text{total}} = w_{\text{ret}} g_{\text{ret}} + w_{\text{3m}} g_{\text{3m}}$（$w$ 是 uncertainty 权重）

如果 $g_{\text{ret}}$ 和 $g_{\text{3m}}$ **方向冲突**（余弦相似度为负），合并梯度只是两者的**衰减 partial cancellation**。无论 $w$ 如何调整，都不能让冲突消失 —— 只能让某一侧贡献变小。

Shared representation 学到的是两个任务的**妥协空间**，对两个都 mediocre。这就是 negative transfer。

**引用**：Yu, Kumar, Gupta, Levine, Hausman, Finn (NeurIPS 2020), "Gradient Surgery for Multi-Task Learning" 引入 PCGrad：对每个任务梯度，投影到其他任务梯度的 normal plane（消除方向冲突），保留 magnitude。

### 2.2 5b 实证 —— Negative Transfer 确认

| 变体 | active tasks | IC mean | IC t-stat |
|---|---|---|---|
| Rung 4 (单任务 ret) | {ret} | +0.0048 | 0.254 |
| **5b** | {ret, ret3m} | **-0.0036** | -0.211 |
| 5c | {ret, vol} | +0.0061 | 0.324 |
| 5d | {ret, ret3m, vol} | +0.0173 | 0.892 |

**5b 比 4 更差，IC 翻负号**，这是典型 negative transfer。3 个月 forward return 跟 1 个月 forward return 在 2022 regime shift 期间方向冲突（长期 continuation vs 短期 reversal），uncertainty weighting 无法修复。

5d 比 5b/5c 都好，说明**加第三个任务（vol）能缓解 ret vs ret3m 的冲突** —— vol 预测强迫 encoder 学到跟 return 正交的 risk representation，某种程度上稀释了冲突。但这是副作用，不是设计初衷。

### 2.3 Fix 方案（plan 里 Tier C4）

**推荐**：skip 实现 PCGrad，而是**在 report 里把 5b 结果作为 negative transfer 实证**来讨论。PCGrad 要 second-order gradient，40+ 行代码，3 天 deadline 风险大。

---

## 3. Regime-Gated MoE 架构

### 3.1 plain MoE (`regmtl.py`)

**架构**：
```
Shared Encoder: Linear(14 → 64) → ReLU → Dropout(0.10) → Linear(64 → 32) → ReLU → Dropout(0.10)
Gate Network:   Linear(3 → 16) → ReLU → Linear(16 → K)  [K=3 experts]
Expert Heads:   每个 active task K 个 Linear(32 → 1)（nn.ModuleList）
```

**Gate 输入**：3-dim HMM regime posteriors（`regime_p0, regime_p1, regime_p2`），从 HMM 每 fold 重算。

**Mixing**：
$$
\hat{y}_{\text{mixed}}^{(k)} = \sum_{j=1}^{K} g_j \cdot \hat{y}_{\text{expert}_j}^{(k)}, \quad g = \text{softmax}(\text{gate}(r))
$$

**Softmax temperature = 1.0 硬编码**（`regmtl.py:126`）—— 没有 temperature 退火。

### 3.2 Loss 只作用在 mixed output

`criterion(preds, yr, y3m, yv)` 把同一个 `UncertaintyMTLLoss` 作用在 **`y_mixed`**（`regmtl.py:253`）。

**重要**：
- **没有 per-expert loss**
- **没有 gate entropy regularizer**
- **没有 load-balancing penalty**（Shazeer 2017 建议）

### 3.3 Enhanced MoE (`regmtl_enhanced.py`) 差异

| 组件 | plain MoE | Enhanced MoE |
|---|---|---|
| 股票 feature 数 | 14 (V2) | 11 (7 LASSO + 4 interaction) |
| Encoder hidden1 | 64 | 32 |
| Encoder hidden2 | 32 | 16 |
| Gate 输入维度 | 3 (仅 regime) | 6 (3 regime + 3 macro) |
| Gate hidden | 16 | 12 |
| Macro 缩放 | N/A | StandardScaler per-fold |

**Look-ahead 污染**：`ENHANCED_MOE_FEATURE_COLS`（7 个 feature）是跑 LASSO 在**所有 58 fold** 上算 selection frequency 选的 → test window 的信息泄漏到 feature selection 步骤 → 同 `2c_OLS_LASSO3/5` 一样的 look-ahead bug。

Fix plan 在 Tier C5：pre-2020 folds 重跑 LASSO 选 feature。

---

## 4. MoE 为什么没 work —— Shazeer 2017 警告

### 4.1 Shazeer et al. 2017 (ICLR) 核心观点

引用：Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (ICLR 2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

**soft-gate MoE 的 failure mode**：如果没有显式约束，
1. **Expert collapse**：所有 expert 收敛到类似函数（homogeneous experts），K=3 退化到 K=1
2. **Gate collapse**：gate 全选同一个 expert，其他闲置
3. **Load imbalance**：某 expert 处理大多数样本，训练不稳定

**Shazeer 提出的 defence**：
- **Importance regularizer**：鼓励 gate entropy，防止 always pick same expert
- **Load regularizer**：惩罚 $\text{Var}(\text{gate.mean}(0))$（每 expert 在 batch 里的平均使用率 variance）

### 4.2 MA751 实测

Gate weights 实证 (`results_5d_enhanced_moe.parquet` 分析):

| 指标 | 值 |
|---|---|
| Expert 0 mean gate weight | 0.33 |
| Expert 1 mean gate weight | 0.33 |
| Expert 2 mean gate weight | 0.34 |
| Gate entropy std | 0.08-0.12 |

**几乎完全均匀分布**。意味着 experts 没 specialize，或者 gate 没学到区分不同 regime 该用哪个 expert。

**K=3 的 MoE 在事实上等于 K=1 的 MTL**，额外的 2 个 expert 只是增加了无用参数。

### 4.3 Fix 方案（plan Tier C1）

加 gate entropy + load balancing 正则项（~10 行）：

```python
gate_w = preds["gate_weights"]  # [B, K]
# Gate entropy: 鼓励 per-sample 不要塌到单 expert
entropy_per_sample = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=1)
# Load balance: batch-level 使用率 variance 惩罚
load_var = torch.var(gate_w.mean(dim=0))

loss = mtl_loss - lambda_ent * entropy_per_sample.mean() + lambda_lb * load_var
# lambda_ent=0.01, lambda_lb=0.01 初试
```

---

## 5. HMM Underidentified 问题

### 5.1 参数量估计

HMM 设置 (`regime.py`):
- K = 3 states
- observation = 5-dim vector (`mkt_ret_1m, mkt_rv_1m, VIXCLS, T10Y2Y, BAMLH0A0HYM2`)
- `covariance_type='full'` (每个 state 一个 5×5 对称协方差)

**自由参数**：
- Initial state distribution：K-1 = **2**
- Transition matrix：K·(K-1) = **6**
- Means：K × 5 = **15**
- Covariances：K × 5·(5+1)/2 = K × 15 = **45**
- **合计**：2 + 6 + 15 + 45 = **68 个自由参数**

### 5.2 早期 fold 的识别不足

- Training window 最小 60 months（`config.py:186`）
- 最早 fold：60 observation / 68 params → **underidentified**
- 中间 fold：80-90 obs → 边界 identified
- 最后 fold：~118 obs → 1.7x 参数量，仍然 tight

**后果**：早期 fold 的 regime posteriors 是 noisy，gate input 不稳定 → MoE 学到的 gate mapping 在早期和晚期 fold 之间有 structural break。

### 5.3 Fix 方案

- **短期**：降到 `covariance_type='diag'` 把 cov params 从 45 降到 15，总参数量从 68 降到 **38**
- **中期**：降 K 到 2（bull/bear 二元）
- **长期**：把 HMM input 降到 3 维（VIX + T10Y2Y + market return）

（plan 里没明确列入 Tier，考虑 Tier C 后期补充）

---

## 6. 关键 Hyperparameter 清单（所有 Rung 5 变体）

### 6.1 Architecture

| 参数 | MTL (5b/c/d) | plain MoE | Enhanced MoE | 来源 |
|---|---|---|---|---|
| n_factors | 14 (V2+miss=19) | 14 | 11 | `config.py` |
| hidden1 | 64 | 64 | **32** | `main.py:127`, `regmtl.py`, `regmtl_enhanced.py:83` |
| hidden2 | 32 | 32 | **16** | `main.py:129`, `regmtl_enhanced.py:84` |
| dropout | 0.10 | 0.10 | 0.10 | 多处 |
| activation | ReLU | ReLU | ReLU | |
| K (experts) | N/A | **3** | **3** | `regmtl.py:55` |
| Gate hidden | N/A | 16 | 12 | `regmtl.py:80`, `regmtl_enhanced.py:85` |
| Gate input dim | N/A | 3 (regime) | 6 (3 regime + 3 macro) | `regmtl_enhanced.py:184-196` |

### 6.2 Optimizer

| 参数 | 值 | 文件:行 |
|---|---|---|
| Optimizer | Adam | `main.py:294` |
| lr | 1e-3 | `main.py:265` |
| weight_decay | 1e-5 | `main.py:266` |
| batch_size | 512 | `main.py:262` |
| grad_clip | 1.0 (仅 model.parameters()) | `main.py:319` |

**Bug**：`clip_grad_norm_` 只剪 `model.parameters()`，不剪 `criterion.parameters()`（log-sigma）。log-sigma 可能受到不受控的大梯度更新。Fix ~1 行。

### 6.3 Scheduler / Training loop

| 参数 | 值 | 文件:行 |
|---|---|---|
| Scheduler | ReduceLROnPlateau | `main.py:300` |
| Scheduler patience | 8 | `main.py:301` |
| Scheduler factor | 0.5 | `main.py:301` |
| min_lr | 1e-5 | `main.py:301` |
| max_epochs | 150 | `main.py:262` |
| early-stop patience | 20 | `main.py:267` |
| val_frac | 0.10 (chronological tail) | `main.py:268,271-276` |

**val_frac 是时序 tail** —— 早停 decision 基于 train window 最近 10%，跟 test month 时间上最接近 → **早停间接 peeks 测试期 regime**。

### 6.4 Loss specific

| 参数 | 值 | 文件:行 |
|---|---|---|
| log_sigma init | 0.0 (per task) | `main.py:183-187` |
| var_k | Var(y_k_train_split) | `main.py:190-207` |
| Loss combine | sum over tasks | `main.py:221,228,234` |
| Softmax temp (MoE) | 1.0 (硬编码) | `regmtl.py:126` |
| Gate entropy reg | **无** | - |
| Load balance reg | **无** | - |
| Per-expert loss | **无** | - |

### 6.5 HMM

| 参数 | 值 | 文件:行 |
|---|---|---|
| K (regimes) | 3 | `config.py:85` |
| covariance_type | 'full' | `regime.py:149` |
| n_iter (EM) | 300 | `regime.py:108` |
| random_state | 42 | `regime.py:108` ✅ |
| min HMM train | 24 months | `regime.py:137-140` |

### 6.6 Reproducibility

| 文件 | torch.manual_seed | np.random.seed |
|---|---|---|
| `main.py` | **无** | **无** |
| `regmtl.py` | **无** | **无** |
| `regmtl_enhanced.py` | **无** | **无** |
| `rung5_planned.py` | **无** | **无** |
| `rung5_combined.py` | **无** | **无** |
| `regime.py` | N/A | HMM 有 random_state=42 |

**所有 Rung 5 代码都没 seed**，结果不可复现。Fix 在 plan Tier B3 —— 每 fold 加 `torch.manual_seed(fold_idx)` + `np.random.seed(fold_idx)`。

---

## 7. Fix 优先级排序（交叉链接 plan 的 Tier C）

对应 `~/.claude-eva/plans/i-want-you-to-imperative-badger.md` 的 Tier B / C 编号。

| Priority | 名称 | Plan Tier | 状态 |
|---|---|---|---|
| **高** | `torch.manual_seed(fold_idx)` 所有 Rung 5 文件 | B3 | Day 2 |
| **高** | MSE → Huber (`smooth_l1_loss`) in MTL losses | B2 | Day 2 |
| **高** | Target winsorize 1%/99% | B4 | Day 1 ✅（在 V2 harness 已做） |
| 中 | MoE gate entropy + load balance 正则 | C1 | Day 2 |
| 中 | Rung 5 planned: rank → `fwd_sector_return` | C2 | Day 2 |
| 中 | Pairwise ranking loss in MLP/MTL | C3 | Day 3 if time |
| 中 | Clean feature selection (pre-2020 LASSO) for Enhanced MoE | C5 | Day 2 |
| 低 | PCGrad gradient surgery for 5b/5c/5d | C4 | SKIP / document |
| 低 | ListNet / ApproxNDCG | D1 | SKIP |
| 低 | torchsort differentiable Spearman | D2 | SKIP |
| 低 | Gate temperature annealing | D3 | gate on C1 |
| 低 | HMM `covariance_type='diag'` + K=2 | - | Day 3 if time |

---

## 参考文献

- Kendall, A., Gal, Y., Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR.
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C. (2020). *Gradient Surgery for Multi-Task Learning*. NeurIPS (PCGrad).
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Caruana, R. (1997). *Multitask Learning*. Machine Learning 28, 41-75.
- Ruder, S. (2017). *An Overview of Multi-Task Learning in Deep Neural Networks*. arXiv:1706.05098.
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., Hinton, G. E. (1991). *Adaptive Mixtures of Local Experts*. Neural Computation 3, 79-87.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS.
- Diebold, F. X., Mariano, R. S. (1995). *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics.
- Fama, E. F., MacBeth, J. D. (1973). *Risk, Return, and Equilibrium: Empirical Tests*. Journal of Political Economy 81(3).

---

*Document maintained alongside the plan at `~/.claude-eva/plans/i-want-you-to-imperative-badger.md`. Last updated 2026-04-24.*

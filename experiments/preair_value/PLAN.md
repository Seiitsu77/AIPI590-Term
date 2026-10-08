# 预注册：开播前 AniList 信号能否在 metadata 之外预测新番受众

**状态：结果出来之前写定并提交。** 写这份方案时，任何开播前特征和 label 之间的关系都还没有计算过。唯一看过的结果是：
- label 审查：只比较 label 和 AniList 当前人气，没有碰任何特征；
- 打乱 label 后的空跑：数字没有意义。

## 问题

一部新番在开播当天还没有任何评分，Anime Compass 的 ALS 看不见它。我们要检验：**开播前 AniList 上的"想看"信号，能不能在只用 metadata 的基线之上，把同一季的新番排得更准。**

这对应 Anime Compass 新番栏的"值不值得推"这一半。"推给谁"不在这次检验的范围内。

## 队列与切分

- **队列**：AniList 上 2021 冬 – 2025 秋共 20 季的 TV 番，要求首播日期完整、trends 已下载，共 926 部，全部是日本作品。不排除任何一部，包括开播前没有任何记录的 13 部。
- **训练集**：2021–2023 年的 12 季，532 部。
- **测试集**：2024–2025 年的 8 季，394 部。
- 按时间切分，不随机切分。

## Label

**主 label：`y_score80 = log1p(score_80plus)`**，即 AniList 上打 80/100 分及以上的人数，快照时间 2026-10-07。

选它是因为 `label_audit.py` 的审查结果（只比较 label 和 AniList 当前人气，不涉及特征）：

| | MAL `members` | AniList `score_80plus` |
|---|---|---|
| 覆盖率 | 91%，最冷门五分之一只有 72% | 100% |
| 缺失的番 | 人气中位数 5,322，有数据的是 43,728。缺失集中在冷门番 | 无 |
| 快照日期 | 不明；最近一季明显还没累积完 | 2026-10-07 |
| 衡量什么 | 加入列表的人数。和 AniList 人气的季内 Spearman 是 0.91–0.996 | 打高分的人数，和 Anime Compass 的正反馈定义（≥8 分）一致 |

**稳健性 label：`y_mal_members = log1p(MAL members)`**，只在有这个值的 844 部上用。它的作用是回应"用 AniList 预测 AniList"的质疑：特征来自 AniList，这个 label 来自 MAL。

## 特征

所有特征都只用首播日 00:00（日本时间）之前就知道的信息。

**M：metadata 基线（27 个）**
- `src_*`：原作类型 one-hot，包括 MANGA、LIGHT_NOVEL、ORIGINAL、VIDEO_GAME、MULTIMEDIA_PROJECT（训练集里各至少 10 部），其余合并为 OTHER。
- `genre_*`：AniList 的 18 个题材，多选 0/1。题材由 AniList 编辑设定，不是用户投票的 tags。
- `log_duration`：单集时长的对数。
- `is_sequel`：是否有一个 PREQUEL 或 PARENT 关系的动画，并且它的开始日期**确定早于**本作首播。
- `log_prequel_pop`：这些前作在本作首播**前一天**的 AniList 人气，取最大值，再取 log1p。数据来自 `download_prequel_popularity.py`，取前作在首播前的最后一条 trend 记录，离首播最远 12 天。不是续作则为 0。`prequel_pop_missing` 标记"是续作但前作没有记录"的情况。
- `studio_prior`：主要工作室之前作品的平均 log1p（开播后第 120 天人气）。只算队列内、并且第 120 天早于本作首播的作品。`log_studio_n_prior` 是这些作品的数量。没有历史的番用训练集的平均值填补，并用 `studio_no_history` 标记。

**A：AniList 开播前信号（7 个）**。只用 `days_from_premiere <= -1` 的 trend 记录。
- `log_pop_m1`、`log_pop_m30`、`log_pop_m90`：首播前 1、30、90 天的片单人数，取 log1p。缺的天在前后两条开播前记录之间线性插值；最后一条记录之后的天沿用最后的值；第一条记录之前记为 0。
- `growth_30`：`log_pop_m1 − log_pop_m30`，即开播前一个月的增长。
- `log_trending_m7`：首播前 7 天 `trending` 的平均值，取 log1p。
- `not_tracked`：开播前完全没有记录。
- `log_days_tracked`：AniList 第一次记录到这部番，距首播多少天，取 log1p。

## 模型

- **主模型：ridge 回归。** 先标准化（只用训练集拟合），然后 `RidgeCV`，alpha 从 10⁻² 到 10³ 共 11 个值里选。
- **描述性对照：`HistGradientBoostingRegressor`**，超参数固定，不调参：`max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0`。
- 每个模型只在训练集上拟合一次。

## 指标与推断

- **指标**：在每个测试季内部算预测值和 label 的 Spearman 相关，再对 8 个测试季取平均。新番栏只需要把同一季的番排对，所以季内指标也消除了不同年份 label 累积时间不同的影响。
- **推断**：配对 bootstrap，重复 2,000 次，随机种子 20261008。每次在每个测试季内有放回地重抽测试番，同一次抽样同时用来算 M 和 M+A，得到差值。取第 2.5 和第 97.5 百分位作为 95% 区间。

## 判定规则

| 检验 | 内容 | 通过条件 |
|---|---|---|
| **H1（主检验）** | ridge，`y_score80`，全部 394 部测试番，M+A 对比 M | 差值 > 0，且 95% 区间下界 > 0 |
| **H2（次要检验）** | 同上，只看测试集里的非续作（`is_sequel == 0`）。模型仍用全部训练集拟合 | 同上 |

H2 来自第一轮 README 的假设：另类数据最有用的地方，是 metadata 看不出来的原创作品和新作。

**只有 H1 和 H2 给出通过或不通过的结论。** 以下分析都只是描述性的，区间是否包含 0 都不算证据：
- boosted trees 版本的 H1、H2；
- `y_mal_members` 版本的 H1、H2（训练集和测试集都只用有这个值的番）；
- 2024、2025 年分别单独作为测试集；
- 只用 A 的模型（可以和第一轮 Wikipedia 单独使用时的 0.565 对比）。

## 已知局限（写在结果之前）

- **特征和主 label 来自同一个平台。** 开播前在 AniList 上加了片单的人，有一部分就是后来打 80 分以上的人。这不是 look-ahead（特征确实都是开播前的），但会让 A 显得比跨平台时更强。MAL 稳健性分析就是为此设的。
- **bootstrap 只考虑了测试集的抽样误差，没有考虑模型训练的随机性。** 空跑时，boosted trees 版本的 H2 在随机 label 上得到了不含 0 的区间（+0.142，区间 [+0.002, +0.276]）。这也是 boosted trees 只作为描述性分析的原因。
- **题材和单集时长是今天的版本**，可能在播出后改过。原作类型、工作室、前作关系在开播前就确定了。
- **集数没有放进特征**，因为最终集数常常是播出后才确定的。第一轮的基线用了集数，所以这次的基线在这一项上偏弱，对 A 有利。
- **工作室历史只从 2021 年算起**，所以 2021 年的训练番大多没有历史，填了平均值。
- **前作人气用的是 AniList 的数据**，和 A 同一个平台。它算作基线里"系列前作人气"的部分，所以基线更强，检验更保守。

## 执行顺序

1. `python experiments/preair_value/label_audit.py`：label 审查（已完成）
2. `python scripts/ani_list/download_prequel_popularity.py`，然后 `python scripts/ani_list/build_tables.py`（已完成）
3. `python experiments/preair_value/features.py`（已完成）
4. `python experiments/preair_value/evaluate.py --dry-run`（已完成）
5. **提交这份方案和全部代码**
6. `python experiments/preair_value/evaluate.py`：正式运行，只跑一次。结果写入 `results/results.json`，并在 `RESULTS.md` 里如实报告，不管结果是什么。

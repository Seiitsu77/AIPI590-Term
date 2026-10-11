# 来源说明

这个目录最初是在作者本机的另一个 git 仓库里完成的，2026-10-11 移植到团队仓库。本文件和 `.gitattributes` 的这次改动都发生在正式测试之后，不涉及方案、模型或结果。

## commit 哈希对照

移植时内容逐字节相同（每一步的 git tree 哈希都核对过一致），但父提交不同，所以 commit 哈希变了。`RESULTS.md`、`AI_USAGE.md`、`MODEL_SELECTION.json`、`results/*.json` 和 `src/pre_run_check.py` 里引用的都是**本机仓库**的哈希。

| 内容 | 本机仓库 | 团队仓库 | 作者时间（本机提交时间） |
|---|---|---|---|
| 预注册、manifest、全部代码（冻结点） | `37defa8` | `c0d2691` | 2026-10-10 18:06:48 −04:00 |
| `.gitattributes` | `836016b` | `b28beb6` | 2026-10-10 18:07:00 −04:00 |
| `MODEL_SELECTION.json` | `45e2905` | `70bc4f7` | 2026-10-10 18:36:36 −04:00 |
| 正式测试结果、报告、AI 使用记录 | `8b7d9b2` | `1c784c0` | 2026-10-10 18:47:56 −04:00 |
| 报告修订（规则核对、AUC） | `822105a` | `bfba0ba` | 2026-10-10 19:01:14 −04:00 |

## 时间戳能证明什么

表里的时间是本机提交时记录的作者时间。推送到公开仓库发生在 2026-10-11，晚于正式测试（2026-10-10 22:42 UTC）。所以 GitHub 上的记录**不能独立证明**方案冻结早于测试；能说明先后顺序的只有本机仓库的提交历史，以及结果文件里记录的运行时间和代码哈希。

## 不在仓库里的东西

| 内容 | 原因 | 怎么核对 |
|---|---|---|
| 原始评分 `rating_complete.csv` | 818 MB；实验约定不上传 | SHA-256 在 `manifest/split_manifest.json` 的 `inputs.ratings` |
| AniList 开播前窗口（`candidate_trends.parquet`、`candidate_anilist_meta.csv`） | AniList API 条款禁止把接口数据当作数据存储，仓库是公开的 | SHA-256 在 manifest 的 `alt_data_inputs`；可用 `experiments/alt_data_coldstart/src/collect_candidate_windows.py` 重新抓取，但重新抓到的数据不保证逐字节相同 |
| 逐用户 ID 名单 `manifest/users_*.csv` | 实验约定不上传 | SHA-256 在 manifest 的 `files`；`python src/make_split_manifest.py --verify` 按规则重建并比对 |
| `artifacts/`（实验 ALS、特征、用户向量、锁文件原件） | 模型产物，体积大 | SHA-256 在 `MODEL_SELECTION.json` 的 `inputs`；锁文件的副本在 `results/final_test.lock` |

在仓库里、并且被 manifest 固定了摘要的输入：`anime_coldstart/data/` 下的 Wikipedia 浏览量矩阵和词条映射，以及 `experiments/alt_data_coldstart/outputs/personalized_coldstart/candidate_items.csv`。

## 在别的机器上运行

`src/common.py` 里有三个作者本机的绝对路径：评分文件、Anime Compass 的片库，以及通过相对路径引用的 `anime_coldstart/data/raw/anime.csv`（MAL 片库快照，来自 Kaggle 的 Anime Recommendation Database 2020，不在仓库里）。换机器需要改这几个常量并自备这三个文件，脚本会校验它们的 SHA-256。

**正式测试不能重跑。** `final_test.py` 看到锁文件就拒绝运行；换一台没有锁文件的机器重跑，也不再是预注册意义上的那一次测试。

## 换行符

`manifest/split_manifest.json` 和 `manifest/wikipedia_article_audit.json` 是在 Windows 上写出的，它们的摘要是按 CRLF 换行的字节固定的，而 git 里存的是 LF。`.gitattributes` 让这两个文件在任何平台上都检出为 CRLF，这样固定的摘要在别的机器上也能对上。文件内容没有变。

`MODEL_SELECTION.json` 没有被任何检查固定摘要。汇报里提到过它的 SHA-256（`fea86cde…`），那是本机 CRLF 版本的值。

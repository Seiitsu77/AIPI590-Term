# 数据说明

原始数据和 CSV 都已经提交进仓库（仓库是公开的，见文末"使用条款"）。除了 `ani_list_sample_2016_2020/`，其他数据都可以用下面的命令从头重建。

| 目录 | 来源 | 内容 |
|---|---|---|
| `ani_list/` | [AniList GraphQL API](https://docs.anilist.co) | 2021 冬 – 2025 秋每季的番剧列表、metadata，以及 TV 番开播前 365 天到开播后 120 天的**每日**人气 |
| `ani_list_sample_2016_2020/` | AniList（Qingyi 抓取） | 2016–2020 年首播的 10 部 TV 番，每部是**完整**的 trends 历史，用来看 trends 能回溯到多早。`trends.csv` 的列和 `ani_list/trends.csv` 相同。详见该目录的 README；生成脚本在 Qingyi 本地，还没有放进仓库 |
| `anime_offline_database/` | [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database) | 约 4.1 万部番的跨站 ID 映射（MAL、AniList、Kitsu、AniDB、ANN 等）和基础 metadata |
| `jikan/` | [Jikan](https://jikan.moe)（非官方 MAL API） | MAL 的 metadata 和评分分布。2026-10-07 时 API 宕机，还没下载 |

## 重建

```bash
python scripts/anime_offline_database/download.py      # 约 62 MB，几秒
python scripts/anime_offline_database/build_tables.py
python scripts/ani_list/download.py                     # 约 1,000 个请求，限速每分钟 30 次，约 75 分钟，可断点续传
python scripts/ani_list/build_tables.py
python scripts/ani_list/coverage_check.py               # 按人气五分位检查 trends 覆盖率
```

## 文件

`ani_list/`

| 文件 | 一行是 | 主键 |
|---|---|---|
| `anime.csv` | 一部番（所有 format；只有 TV 抓了 trends） | `anilist_id`，`mal_id` 对应 MAL |
| `trends.csv` | 一部番的一天 | `anilist_id` + `date` |
| `relations.csv` | 一条关联（SOURCE、PREQUEL、SEQUEL……），动画和漫画混在一起 | `anilist_id` |
| `external_links.csv` | 一个外部链接（官网、Twitter、流媒体……） | `anilist_id` |
| `coverage.csv` | 覆盖率检查结果 | |

`anime_offline_database/`：`anime.csv`（一行一部番，每个站点的 ID 一列）和 `id_map.csv`（长表，用来 join）。`entry_id` 换一个版本就会变，join 时要用站点 ID。

## 哪些列可以当"开播前"特征

这个项目的核心是：特征只能用开播当时已经知道的信息。

| 能不能用 | 列 | 原因 |
|---|---|---|
| ✅ | `trends.csv` 里 `days_from_premiere <= -1` 的行 | 当天记录，事后不会改写 |
| ✅ | `format`、`source`、`main_studios`、`staff`、`season`、`start_date` | 开播前就确定了 |
| ⚠️ | `relations.csv` | 包含开播**之后**才出现的续作。只能用 `target_start_date` 早于本作首播日期的关联（前作、原作） |
| ⚠️ | `tags`、`genres` | 用户投票产生，是今天的版本，可能混进了播出后的投票 |
| ⚠️ | `episodes`、`trailer_id` | 可能在播出后更新过 |
| ❌ 只能当 label | `popularity`、`favourites`、`average_score`、`mean_score`、`status_*`、`score_*`，以及 offline-db 的 `score_mean` | 都是今天的快照 |

label 的候选：
- `score_80plus`：AniList 上打 80 分及以上的人数。今天的快照，最早几季的番累积时间更长。
- `popularity_1y`：开播满一年那天的片单人数。所有番的时间跨度相同。
- 等 Jikan 恢复后可以加 MAL 的 `votes_8plus`。它和 Anime Compass 训练用的正反馈定义一致。

## 已知的数据问题

- **日期是日本时间。** AniList 在日本时间零点给 trends 打时间戳。day 0 是首播当天，严格的开播前数据要用 `<= -1`。
- **`popularity` 从 2018-03-20 才开始有**（Qingyi 的样本发现，已核实）。更早的行只有 `trending`，`in_progress` 从 2017-01-13 开始有。所以用 AniList 开播前人气的队列，首播时间最早只能到 2019 年左右。2021–2025 队列的数据最早是 2019-12-08，一行都不缺 popularity。
- **偶尔有重复的天。** 两行完全相同，建表时已经去重。
- **会跳过某些天。** 冷门番更常见：全量 926 部里，最冷门五分之一的中位数是缺 49 天，最热门五分之一是 1 天。缺口前后 popularity 通常变了，所以要用前后两天**插值**；直接用前一天的值补只能得到下界。缺多少天记在 `anime.csv:trend_missing_days`。
- **少数番开播前没有任何记录。** 全量 926 部里有 13 部（1%），试抓看到的都是儿童向或小众番。这本身就带信息，建议保留这些番，加一个标记列，不要丢掉。
- **MAL ID 两个来源已经交叉核对过。** 926 部里 920 部一致、0 部冲突；2 部只有 AniList 有 MAL ID，4 部两边都没有。

## 使用条款

- **AniList**：非商业使用免费。条款禁止大规模收集数据，也禁止把 API 当作数据存储；对课程作业这类纯教育用途比较宽松。我们只抓了项目需要的队列。注意仓库是公开的，而这些数据也提交在里面。
- **anime-offline-database**：ODbL 1.0 + DbCL 1.0。使用时要署名"anime-offline-database by manami-project"；衍生出的数据库要用 ODbL 共享。
- **Jikan / MAL**：Jikan 是从 MAL 页面爬数据的非官方 API，要遵守它的限速（每秒 3 次、每分钟 60 次）。

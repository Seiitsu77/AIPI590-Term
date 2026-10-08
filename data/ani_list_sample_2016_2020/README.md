# AniList 样本：2016–2020 首播的 10 部动画

来源是 [AniList GraphQL API](https://docs.anilist.co)，2026-10-08（UTC）抓取。这是一次可行性试抓，用来看 AniList 有哪些字段、历史人气（MediaTrend）能回溯到多早。它和 `ani_list/` 的队列不重叠：那边是 2021–2025，这里是 2016–2020。

## 样本怎么选的

从 MAL 2020 快照里 2016–2020 首播的 TV 动画中，沿 MAL 成员数排名等距取 10 部，所以从最冷门到最热门都有。它不是随机样本，10 部也撑不起任何比例结论。和 MAL 的匹配只用 `idMal`，没有用标题。

## 文件

| 文件 | 一行是 | 主键 | 行数 |
|---|---|---|---:|
| `anilist.csv` | 一部番 | `id`（AniList）、`mal_id` | 10 |
| `anilist_media_fields.csv` | AniList `Media` 类型的一个字段（来自 schema introspection） | `field` | 55 |
| `trends.csv` | 一部番的一天 | `anilist_id` + `date` | 22,647 |

`trends.csv` 的列和 `ani_list/trends.csv` 完全相同，可以直接拼在一起。`anilist.csv` 的列和 `ani_list/anime.csv` **不同**（比如这里是 `id`，那边是 `anilist_id`），是另一个脚本展平的。

`raw/` 是 API 的原始响应：

- `anilist_media.json`、`anilist_media_fields.json`：10 部番的 metadata 和字段清单。
- `trends/<anilist_id>/full_pNNNN.json`：完整历史，从最早一天开始，每页 50 行。
- `trends/<anilist_id>/window_pNNNN.json`：开播前 90 天，用 `date_greater` / `date_lesser` 查的，用来验证按日期查询的结果和完整历史一致（10 部全部一致）。
- `trends/<anilist_id>/latest_p0001.json`、`recheck_p0001.json`：最新一行，相隔约 25 分钟抓了两次。
- `request_log.jsonl`：每次 HTTP 请求的状态码和限速 header。505 次请求全部成功，没有 429。

## 和 `ani_list/` 不一样的地方

- **trends 覆盖整个生命周期。** 从 AniList 上的第一条记录一直到 2026-10-08，`days_from_premiere` 从 −391 到 3841，不是开播前 365 天到开播后 120 天。
- **2018-03-20 之前没有 `popularity`。** 更早的行只有 `trending`；`in_progress` 从 2017-01-13 开始有。所以 2016 年首播的 5 部在开播前只有 `trending`，2018 年及以后首播的 4 部开播前三项都有。
- **有 1 部开播前没有任何记录。** `anilist_id` 121199 的条目是首播三年后才建的。其余 9 部共有 847 行开播前数据。
- **`pageInfo.total` 不是真实行数。** 它恒为 5000，翻页只能看 `hasNextPage`。

## 已知的数据问题

- **日期是日本时间。** `trends.csv` 的 `date` 已经按日本时间取；day 0 是首播当天，开播前数据要用 `days_from_premiere <= -1`。
- **`anilist.csv` 里的 `trends_first_day` 和 `trends_first_rows` 早了一天。** 这两列是按 UTC 取的日期，以 `trends.csv` 为准。
- **重复的天。** 原始响应里有 25 行和另一行完全相同，`trends.csv` 已去重，`raw/` 里保留原样。没有同一天两行数值不同的情况。
- **`anilist.csv` 的 `staff_top`、`characters_top`、`trends_first_rows` 被查询截断了**，分别只取了 5、3、3 条，不代表实际数量。
- **除 trends 以外的列都是今天的快照。** `averageScore`、`popularity`、`favourites`、`tags`、`rankings` 等不能当开播前特征。

## 重建

生成脚本目前不在这个仓库里，在 Qingyi 本地的 term project 目录：`anime_coldstart/src/peek_anilist_jikan.py`（metadata）和 `experiments/alt_data_coldstart/src/collect_trends.py`（trends，限速每分钟 20 次，带缓存和断点续传）。

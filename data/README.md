# 数据说明

原始数据和 CSV 都已经提交进仓库（仓库是公开的，见文末"使用条款"）。除了 `ani_list_sample_2016_2020/`，其他数据都可以用下面的命令从头重建。

| 目录 | 来源 | 内容 |
|---|---|---|
| `ani_list/` | [AniList GraphQL API](https://docs.anilist.co) | 2021 冬 – 2025 秋每季的番剧列表、metadata，以及 **TV、ONA、剧场版、短篇**开播前 365 天到开播后 120 天的**每日**人气（特别篇和 OVA 只有列表） |
| `ani_list_live/` | AniList，2026-10-08 抓取 | **2026 秋**（正在播）和 **2027 冬**（即将播）两季，文件和 `ani_list/` 一样。用于演示，和预注册实验的历史数据分开放。2026 秋季 55 部、2027 冬季 50 部 TV 番全部有 trends，而且都有开播前数据；ONA、剧场版、短篇另有 46 部。2027 冬季有 45 部只知道"1 月"开播，暂时以 **2027-01-01 作为临时首播日**（`anime.csv` 里 `premiere_provisional = 1`）：它们的 `days_from_premiere` 从 1 月 1 日算起，所以"day −1"是 12 月 31 日，不是真正开播的前一天。定档后用 `download.py --seasons 2027-winter --out data/ani_list_live --provisional-start --refresh-listings` 重跑，脚本会按真实日期重新抓这些番 |
| `ani_list_sample_2016_2020/` | AniList（Qingyi 抓取） | 2016–2020 年首播的 10 部 TV 番，每部是**完整**的 trends 历史，用来看 trends 能回溯到多早。`trends.csv` 的列和 `ani_list/trends.csv` 相同。详见该目录的 README；生成脚本在 Qingyi 本地，还没有放进仓库 |
| `wikipedia/` | [Wikidata](https://www.wikidata.org) + [Wikimedia Pageviews API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/) | 2,059 部新番（TV、ONA、剧场版、短篇；历史队列 + 当季和下一季）在英文、日文、中文 Wikipedia 上的**每日浏览量**，从首播前 365 天到首播后 120 天 |
| `anime_offline_database/` | [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database) | 约 4.1 万部番的跨站 ID 映射（MAL、AniList、Kitsu、AniDB、ANN 等）和基础 metadata |
| `anime_compass/` | [Anime Compass](https://github.com/Seiitsu77/anime-compass) 的公开模型（Hugging Face `Seiitsu/anime-compass-data`） | 训练好的 ALS 模型（每部番一个 128 维向量，用 MAL ID 索引）和它的片库 |

原计划还要用 [Jikan](https://jikan.moe)（非官方 MAL API），但它从 2026-10-07 起一直连不上，已经放弃。它原本负责的番剧列表、metadata、标准答案和 MAL ID 对照，现在都由 AniList 和 anime-offline-database 提供。

## 重建

```bash
python scripts/anime_offline_database/download.py      # 约 62 MB，几秒
python scripts/anime_offline_database/build_tables.py
python scripts/anime_compass/download.py                # 约 14 MB，按朋友 manifest 里锁定的 SHA-256 校验
python scripts/ani_list/download.py                     # 新番 TV：约 1,000 个请求，限速每分钟 30 次，约 75 分钟，可断点续传
python scripts/ani_list/download.py --formats ONA MOVIE TV_SHORT --provisional-start   # 非 TV 新番：约 1,060 个请求，约 40 分钟
python scripts/ani_list/download_prequel_popularity.py  # 新番前作在开播前一天的人气，约 40 个请求
python scripts/ani_list/download_catalog.py             # 老番（需要上面的 ALS 模型）：305 个请求，约 10 分钟
python scripts/ani_list/build_tables.py                 # 生成 anime/trends/relations/external_links/prequel_popularity/catalog.csv
python scripts/ani_list/coverage_check.py               # 按人气五分位检查 trends 覆盖率
python scripts/ani_list/catalog_check.py                # 检查新老番词汇对齐
python scripts/wikipedia/download.py                    # 约 13,600 条浏览量序列，限速每秒 2 个，约 2 小时，可断点续传（同一时间只能跑一个）
python scripts/wikipedia/build_tables.py
python scripts/wikipedia/coverage_check.py              # 和第一轮实验对比覆盖率
```

## 文件

`ani_list/`

| 文件 | 一行是 | 主键 |
|---|---|---|
| `anime.csv` | 一部番（所有 format；TV、ONA、MOVIE、TV_SHORT 抓了 trends） | `anilist_id`，`mal_id` 对应 MAL |
| `trends.csv` | 一部番的一天 | `anilist_id` + `date` |
| `relations.csv` | 一条关联（SOURCE、PREQUEL、SEQUEL……），动画和漫画混在一起 | `anilist_id` |
| `external_links.csv` | 一个外部链接（官网、Twitter、流媒体……） | `anilist_id` |
| `coverage.csv` | 覆盖率检查结果 | |
| `catalog.csv` | 一部**老番**（Anime Compass ALS 里有学到向量的番），列和 `anime.csv` 相同，trends 相关的列为空 | `anilist_id`，`mal_id` 对应 ALS 的 `anime_ids` |

`catalog.csv` 用来让新番和老番使用同一套词汇（`scripts/ani_list/download_catalog.py` 生成，`catalog_check.py` 检查）：
- **覆盖率：** 15,247 部学到向量的老番里，13,131 部（86.1%）在 AniList 上找到了。缺的 2,116 部大多不是正片：特别篇 567、OVA 532、ONA 374、音乐 237、剧场版 216，TV 只有 190。它们通常是 MAL 单独列条目、AniList 并进正片的情况，例如《Fate/stay night UBW 序章》。
- **词汇对齐**（926 部新番的词汇有多少在老番里也出现）：

  | 字段 | 之前（offline-db / Anime Compass 片库） | 现在（AniList） |
  |---|---|---|
  | 主要工作室 | 206 个里 3%（按出现次数 5%） | 73%（按出现次数 88%） |
  | tags | 0% | 100% |
  | 题材 | 100% | 100% |
  | 原作类型 | 没有 | 100% |
  | staff | 没有 | 60%（按出现次数 72%） |

  对不上的 56 个工作室，大多是 2021 年以后才出第一部作品的新工作室（如 Studio Bind、Gekkou），老番里本来就没有。这是真实的冷启动，不是数据问题。
- **ID：** AniList 的 idMal 和 offline-db 有 12,973 部一致，4 部冲突（两边编号差 1 的录入错误）。另有 13 个 MAL ID 各对应 2 个 AniList 条目（AniList 把一部作品拆成了两个条目），用的时候要去重。

`anime_offline_database/`：`anime.csv`（一行一部番，每个站点的 ID 一列）和 `id_map.csv`（长表，用来 join）。`entry_id` 换一个版本就会变，join 时要用站点 ID。

`wikipedia/`

| 文件 | 一行是 | 主键 |
|---|---|---|
| `mapping.csv` | 一部番映射到的一个词条，以及映射原因 `role` 和重定向数 | `anilist_id` + `wiki` + `article` |
| `titles.csv` | 一部番：Wikidata 条目、三种语言的词条、开播前 30 天和 90 天的浏览量 | `anilist_id` |
| `title_daily.csv` | 一部番的一天：三种语言的浏览量，分成 `*_anime`（番自己 + 系列词条）和 `*_source`（原作 + 媒体系列词条），以及合计 `total` | `anilist_id` + `date` |

- **映射方法：** MAL ID →（Wikidata P4086）番的条目，再取它的**系列**（P179）、**原作**（P144）和**媒体系列**（P8345）条目，以及系列条目自己的原作和媒体系列。四层的词条都算进来。必须这样做，因为番自己的词条常常是开播后才从原作词条拆出来的：例如"Frieren (TV series)"在开播前浏览量是 0，开播前的关注都在原作词条"Frieren"上。如果像第一轮那样只取番自己和系列的词条，最热门五分之一里会有 110/176 部的开播前浏览量为 0。
- **重定向也计入。** 浏览量按读者访问的标题记录，所以词条改名前的浏览量记在旧标题（现在的重定向）下面。例如《赛马娘》的英文词条以前叫"Uma Musume Pretty Derby"，开播前那段时间的 21.6 万次浏览都在这个旧标题下。历史队列 826 部有词条的番里，有 210 部的开播前浏览量因为计入重定向而增加了 10% 以上，开播前总浏览量增加了 7%。
- **每个页面只下载一次，时间窗口取所有用途的并集。** 一个页面可能同时是某部番的词条和另一个词条的重定向。之前这两种用途各自生成下载任务、写同一个文件，窄窗口会覆盖宽窗口（例如日文《Cardfight!! Vanguard Divinez》只剩 2020–2022 年的数据，用到它的 2024–2025 年的番就成了 0）。已修复并重抓，现在 19,955 个"番 × 页面"组合都完整覆盖了各自需要的时间段。
- **每个页面对每部番只算一次。** Wikidata 的链接有时会指向一个现在已经是重定向的页面（例如日文"鬼滅の刃 遊郭編"），它会被同时当成词条和别的词条的重定向。建表时已经去重，否则会有 19 部番被重复计数。
- **覆盖率**（`coverage_check.py`，历史队列 926 部，按 AniList 当前人气在每季内分五档）：

  | | 第一轮（2016–2020） | 这次（2021–2025） |
  |---|---|---|
  | 映射到词条 | 67% | 89% |
  | 最冷门五分之一有开播前浏览量（day −30..−1） | 7% | 68% |
  | 最热门五分之一中，开播前浏览量为 0 的番 | 36/215 | 0/176 |

- **当季和下一季：** 2026 秋季 55 部里有 67% 映射到了词条。2027 冬季 47 部（有 MAL ID 的）里 36 部有词条，到目前为止都有浏览量；但它们用的是临时首播日 1 月 1 日，"开播前 30 天"落在 12 月，现在还没有数据。
- **404 的意思：** 返回 404 的 1,240 条序列（绝大多数是重定向）表示这个标题在那段时间没有被访问过，不是错误。抽查时已排除标题编码的问题。
- **开播前数据是当时记录的：** Pageviews 按天记录、不回填。但"取哪些词条"用的是今天的 Wikidata 结构；某个词条开播前还不存在的话，它开播前的浏览量就是 0，这是真实情况。

**非 TV 新番**（ONA、剧场版、短篇）

trends、前作人气和 Wikipedia 的抓法都和 TV 一样，但覆盖率低不少，ONA 尤其低：

| 队列 | 格式 | 部数 | AniList 有开播前数据 | 有 MAL ID | 有 Wikipedia 词条 | Wikipedia 有开播前浏览量 |
|---|---|---:|---:|---:|---:|---:|
| 2021–2025 | TV | 926 | 99% | 100% | 89% | 89% |
| 2021–2025 | ONA | 531 | 45% | 91% | 24% | 22% |
| 2021–2025 | 剧场版 | 309 | 85% | 97% | 47% | 44% |
| 2021–2025 | 短篇 | 219 | 66% | 97% | 32% | 29% |
| 当季 + 下一季 | ONA / 剧场版 / 短篇 | 15 / 16 / 15 | 93% / 100% / 80% | | | |

ONA 里有大量宣传短片、网络短篇和合作企划，很多在 AniList 上是播出后才建条目的，所以另类数据对它们的覆盖本来就弱。剧场版的"首播"是影院上映日，受众和评分的累积方式和 TV 不同，分析时要单独看。

**预注册实验只用 TV。** `experiments/preair_value/` 的代码已经明确筛选 `format == "TV"`，加入非 TV 的 trends 之后，特征表和原来逐字节一致。

`anime_compass/`

- `als_production_item_factors.npz`：数组 `anime_ids`（MAL ID）、`item_factors`（18,064 × 128）和 `metadata_json`。模型用 Kaggle 的 MAL 评分训练，只用了 8 分及以上的评分，共 30.9M 条正反馈、30.8 万用户。
- `anime_catalog_serving.json`：同样 18,064 部番的片库，字段有 `id`、`title`、`type`、`episodes`、`start_year`、`genres`、`members`、`synopsis`。
- **模型里没有用户向量。** 用户向量是从他喜欢的番临时算出来的：`x_u = (YᵀY + α·Y_uᵀY_u + λI)⁻¹ (α·Σ y_i)`，其中 α = 5、λ = 0.05，记在 metadata 里（见 anime-compass 的 `backend/anime_agent/als_serving.py`）。所以任何新用户（比如 AniList 用户）都能放进同一个空间。
- **2021 年以后的番虽然在片库里，但向量是 0。** 片库后来补录了新番，可评分只到 2021 年 1 月左右：2022 年及以后的番 100% 是零向量，2021 年的番 91% 是零向量。也就是说，对这个模型来说它们是真正的冷启动。
- **队列里有 7 部番不是冷启动。** 它们在 2020-12-27 到 2021-01-10 之间首播，已经有学到的向量（MAL 39586、39783、40786、41694、42203、42897、42923）。评估"推给谁"时要剔除或单独标记。
- **926 部队列番里，有 133 部（14%）的前作在模型里有学到的向量**，可以借前作的位置来放置新番。其余的只能靠 metadata 去找相似的老番。

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
- 如果想要 MAL 社区的对照，可以用 MAL 官方 API v2（需要自己注册 Client ID）。它有人数和均分，但没有评分分布，所以算不出"8 分及以上的人数"。

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
- **Wikidata / Wikipedia 浏览量**：Wikidata 是 CC0，浏览量数据可以自由使用。Wikimedia 的 REST API 会对请求太快的客户端返回 429，所以下载脚本限速为每秒 2 个，并且在 User-Agent 里写了仓库地址作为联系方式。
- **Anime Compass 模型**：代码是 MIT 许可，训练数据来自 CC0 的 [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)。
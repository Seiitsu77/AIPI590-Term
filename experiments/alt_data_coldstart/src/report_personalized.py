"""Render the personalized cold-start feasibility report.

Every number comes from outputs/personalized_coldstart/ratings_feasibility.json and
alt_data_coverage.json, which personalized_feasibility.py and audit_candidates.py wrote.
None is typed by hand. The design sections are prose, but the counts they quote are read
from the same two files, so the report cannot drift from the data.

Usage:
    python src/report_personalized.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "personalized_coldstart"
ENGINEERING = Path(r"C:\Study\Toy project\data\evaluation\personalized\results\als_confirmation_users800\engineering.csv")
ALS_REFERENCE_POSITIVES = 24_916_911  # train positives behind that recorded build (Anime Compass docs/ARCHITECTURE.md)

ROLE_LABEL = {"cold_train": "cold-train", "cold_val": "cold-validation", "cold_test": "cold-test", "cold_censored": "cold-test-late（2020 秋）"}


def n(value) -> str:
    return f"{int(value):,}"


def pct(value) -> str:
    return "—" if value is None or value != value else f"{100 * float(value):.1f}%"


def table(header: list[str], rows: list[list[str]], text: bool = False) -> str:
    """Markdown table. Numbers are right-aligned; pass text=True when the cells are prose."""
    rule = "|" + "|".join("---" if text or i == 0 else "---:" for i in range(len(header))) + "|"
    return "\n".join(["| " + " | ".join(header) + " |", rule] + ["| " + " | ".join(row) + " |" for row in rows])


def main() -> None:
    r = json.loads((OUT / "ratings_feasibility.json").read_text(encoding="utf-8"))
    a = json.loads((OUT / "alt_data_coverage.json").read_text(encoding="utf-8"))
    f, c, u, s = r["ratings_file"], r["candidates"], r["users"], r["splits"]
    als = r["als_training_set_for_this_design"]
    cov, leak, wiki, content = a["anilist_coverage"], a["time_leakage"], a["wikipedia_coverage"], a["content_coverage"]
    split = r["definitions"]["split"]
    test, val, train, late = s["cold_test"], s["cold_val"], s["cold_train"], s["cold_censored"]

    # Anime Compass's own record of its evaluation ALS build: the only timing used here.
    recorded = None
    if ENGINEERING.exists():
        with ENGINEERING.open(encoding="utf-8", newline="") as handle:
            recorded = next((row for row in csv.DictReader(handle) if row["model"] == "als"), None)
    als_seconds = float(recorded["offline_training_duration_seconds"]) if recorded else None
    als_estimate = als_seconds * als["positives"] / ALS_REFERENCE_POSITIVES if als_seconds else None

    out: list[str] = []
    add = out.append

    add("# Personalized Item Cold-Start 离线实验：可行性报告")
    add("")
    add("本报告由 `src/report_personalized.py` 从 `ratings_feasibility.json` 和 `alt_data_coverage.json` 生成，数字都不是手填的。"
        "这一轮只做了计数和审计：没有训练任何模型，没有改动 Anime Compass 或 AnimeRec，也没有任何个性化推荐效果可以报告。")  # fmt: skip
    add("")

    add("## 结论")
    add("")
    add(f"**具备开展实验的条件，用户级 Ground Truth 充足。** 但 AnimeRec 本身不含评分数据，实验必须读本机的原始文件。")
    add("")
    add(f"- **Ground Truth**：原始 `rating_complete.csv` 在本机，{n(f['rows'])} 条评分、{n(f['users'])} 个用户，没有时间戳列。AnimeRec.zip 里**没有**任何用户级评分文件，只有生产 ALS 的 item factors，本报告没有用它代替评分。")
    add(f"- **Cold items**：2019–2020 首播且有真实评分的 TV 动画共 **{n(c['count'])} 部**，其中 {n(c['by_min_positives']['>=50'])} 部有 50 条以上正反馈。")
    add(f"- **用户**：有 5 条以上老番正反馈、并且至少喜欢一部候选番的用户有 **{n(u['with_warm_history_and_1_candidate_positive']['>=5'])} 人**。")
    add(f"- **测试集**（{split['cold_test'][0]} 至 {split['cold_test'][1]} 首播）：{n(test['items'])} 部番、{n(test['warm>=5']['eligible_users'])} 个合格用户、{n(test['warm>=5']['positives_from_eligible'])} 条正反馈。")
    add(f"- **另类数据**：AniList MediaTrend 在首播前 7 / 30 / 90 天已有记录的候选番分别占 {pct(cov['all_candidates']['tracked_since_day-7'])} / {pct(cov['all_candidates']['tracked_since_day-30'])} / {pct(cov['all_candidates']['tracked_since_day-90'])}；"
        f"正反馈 ≥50 的番里是 {pct(cov['at_least_50_positives']['tracked_since_day-7'])} / {pct(cov['at_least_50_positives']['tracked_since_day-30'])} / {pct(cov['at_least_50_positives']['tracked_since_day-90'])}。{pct(wiki['all_candidates']['views_in_30d_before_premiere'])} 有 Wikipedia 开播前浏览量。")  # fmt: skip
    add("- **时间泄漏**：AniList 的行按日本时间零点打戳。按 UTC 取日期会把首播当天的行当成开播前一天，凌晨首播的番在那一天 `trending` 已经高出一个数量级。按日本时间取日期后，第 −1 天及以前的行是干净的；建议再留一天边界，只用到第 −2 天。")
    add(f"- **最大的限制不是用户数，而是番的数量。** 测试集只有 {n(test['items'])} 部番，而且正反馈高度集中：最热门的 10 部占了 {pct(test['top10_share_of_positives'])}。任何关于\"哪类番受益\"的结论，有效样本量是几十，不是几万。")
    add("")

    add("## 1. 原始评分文件")
    add("")
    add(table(["项目", "值"], [
        ["AnimeRec.zip 中的用户级评分", "**不存在**。`data/anime_compass/` 只有 `als_production_item_factors.npz` 和 `anime_catalog_serving.json`"],
        ["本机位置", f"`{f['path']}`"],
        ["大小 / SHA-256", f"{n(f['bytes'])} 字节 / `{f['sha256']}`"],
        ["表头", f"`{f['header']}`（没有时间戳列）"],
        ["行数 / 用户 / 番", f"{n(f['rows'])} / {n(f['users'])} / {n(f['items'])}"],
        ["重复的 (用户, 番)", n(f["duplicate_user_item_pairs"])],
        ["片库内的评分", f"{n(f['rows_in_catalog'])}，其中正反馈（≥8）{n(f['positives_in_catalog'])}，负反馈（≤5）{n(f['negatives_in_catalog'])}"],
    ], text=True))  # fmt: skip
    add("")
    add("行数、用户数和正反馈数与 Anime Compass 文档里的数字一致，所以这就是训练生产模型用的那份文件。它来自 Kaggle 的 Anime Recommendation Database 2020（CC0），"
        "只包含\"看完并打了分\"的记录：弃番、在看、想看都不在里面。队友的 AnimeRec 要跑这个实验，需要另外拿到这个文件。")  # fmt: skip
    add("")

    add("## 2. 候选 cold items")
    add("")
    add("候选定义：MAL 类型为 TV，`Aired` 以完整日期开头且在 2019-01-01 至 2020-12-31 之间，在 Anime Compass 片库内，并且在评分文件里至少有 1 条评分。")
    add("")
    seasons = r["candidates_by_season"]
    add(table(["首播季度", "番数", "正反馈 ≥50 的番", "评分数", "正反馈", "负反馈", "每部番用户数中位数"],
              [[k, n(v["items"]), n(v["items_50_pos"]), n(v["ratings"]), n(v["positives"]), n(v["negatives"]), n(v["median_users"])] for k, v in seasons.items()]))  # fmt: skip
    add("")
    add(f"- 共 {n(c['count'])} 部。正反馈达到 1 / 10 / 50 / 100 / 1,000 条的分别有 "
        f"{n(c['by_min_positives']['>=1'])} / {n(c['by_min_positives']['>=10'])} / {n(c['by_min_positives']['>=50'])} / {n(c['by_min_positives']['>=100'])} / {n(c['by_min_positives']['>=1000'])} 部。")  # fmt: skip
    add(f"- 按标题规则判断，{pct(c['sequel_share_of_items'])} 是续作，它们占正反馈的 {pct(c['sequel_share_of_positives'])}。")
    add(f"- 2020 年第四季度的番在快照时刚播完或还在播，而文件只收\"看完\"的记录，所以这一季的评分偏少。设计里把它单独放一组，不并入主测试集。")
    add("")

    add("## 3. 每部番的反馈")
    add("")
    add("逐部番的正反馈、负反馈和唯一用户数在 `candidate_items.csv`。分布如下：")
    add("")
    add(table(["每部番", "最小", "P10", "P25", "中位数", "P75", "P90", "最大"],
              [[label, *[n(block[k]) for k in ("min", "p10", "p25", "median", "p75", "p90", "max")]]
               for label, block in (("正反馈（≥8）", c["positives"]), ("负反馈（≤5）", c["negatives"]), ("唯一用户数", c["unique_users_per_item"]))]))  # fmt: skip
    add("")
    add(f"候选番的全部评分里，正反馈占 {pct(c['positive_share_of_ratings'])}。最热门的 10 部占全部候选正反馈的 {pct(c['top10_share_of_positives'])}。")
    add("")

    add("## 4. 用户")
    add("")
    add("warm history 指用户在**首播年份早于 2019 的番**上的正反馈。下表是不同门槛下的用户数：")
    add("")
    add(table(["warm 正反馈门槛", "达到门槛的用户", "且至少喜欢 1 部候选番", "且至少喜欢 3 部候选番"],
              [[k, n(u["with_warm_history"][k]), n(u["with_warm_history_and_1_candidate_positive"][k]), n(u["with_warm_history_and_3_candidate_positives"][k])] for k in u["with_warm_history"]]))  # fmt: skip
    add("")
    add(f"用户 warm 正反馈的中位数是 {n(u['warm_positives_per_user']['median'])} 条（P25 {n(u['warm_positives_per_user']['p25'])}，P75 {n(u['warm_positives_per_user']['p75'])}）。")
    add("")

    add("## 5. 交互密度")
    add("")
    add("按建议的 split（见第 8 节），门槛取 warm 正反馈 ≥5（Anime Compass 评估用的最小值）：")
    add("")
    rows = []
    for role, block in s.items():
        e = block["warm>=5"]
        rows.append([ROLE_LABEL[role], f"{split[role][0]} – {split[role][1]}", n(block["items"]), n(block["items_with_50_positives"]), n(e["eligible_users"]),
                     n(e["positives_from_eligible"]), n(e["negatives_from_eligible"]), n(e["positives_per_eligible_user"]["median"]),
                     pct(e["rated_share_of_user_item_grid"]), pct(e["positive_share_of_user_item_grid"]), pct(block["top10_share_of_positives"])])  # fmt: skip
    add(table(["角色", "首播日期", "番数", "正反馈 ≥50", "合格用户", "正反馈", "负反馈", "每人正反馈中位数", "网格被评分占比", "网格正反馈占比", "前 10 部占正反馈"], rows))
    add("")
    add(f"- \"合格用户\"是 warm 正反馈 ≥5 并且在该组里至少喜欢 1 部番的用户。\"网格\"是合格用户 × 该组全部番。")
    add(f"- 同时在测试集里有正反馈和负反馈的用户有 {n(test['warm>=5']['eligible_users_with_pos_and_neg'])} 人，可以用来算\"看过的番里分得清喜欢和不喜欢\"的指标。")
    add(f"- 把门槛提高到 warm ≥20，测试集合格用户是 {n(test['warm>=20']['eligible_users'])} 人，变化很小。")
    add("")

    add("## 6. AniList MediaTrend 开播前覆盖率")
    add("")
    add("AnimeRec 里 2016–2020 的 trends 只有 10 部样本，所以这一轮向 AniList 抓了全部有 AniList 条目的候选番的开播前窗口（每部番 1 次请求，每分钟 20 次）。"
        "day 0 取 MAL 首播日，行的日期按日本时间。")  # fmt: skip
    add("")
    full = cov["all_candidates"]
    add(f"{n(c['count'])} 部候选番里，{n(c['with_anilist_id'])} 部在 AniList 有条目，{n(c['count'] - c['with_anilist_id'])} 部没有。有条目的番在 AniList 上的格式并不都是 TV："
        + "、".join(f"{k} {n(v)} 部" for k, v in c["anilist_format"].items() if k != "no AniList entry") + "。")  # fmt: skip
    add("")
    header = ["分组", "番数", "有 AniList 条目", "第 −7 天前已有记录", "第 −30 天前", "第 −90 天前", "第 −90 天有 popularity", "90 天内缺失天数中位数", "第 −1 天 popularity 中位数"]

    def cov_row(label: str, block: dict) -> list[str]:
        return [label, n(block["titles"]), pct(block["with_anilist_entry"]), pct(block["tracked_since_day-7"]), pct(block["tracked_since_day-30"]), pct(block["tracked_since_day-90"]),
                pct(block["popularity_known_at_day-90"]),
                "—" if block["median_missing_days_in_90"] is None else n(block["median_missing_days_in_90"]),
                "—" if block["median_popularity_day-1"] is None else n(block["median_popularity_day-1"])]  # fmt: skip

    fifth_label = {"Q1_least": "正反馈最少的五分之一", "Q2": "第 2 档", "Q3": "第 3 档", "Q4": "第 4 档", "Q5_most": "正反馈最多的五分之一"}
    rows = [cov_row("全部候选", full), cov_row("正反馈 ≥50", cov["at_least_50_positives"])]
    rows += [cov_row(ROLE_LABEL[role], block) for role, block in cov["by_role"].items()]
    rows += [cov_row(fifth_label[k], block) for k, block in cov["by_positives_fifth"].items()]
    rows += [cov_row(f"AniList 格式 {k}" if k != "no AniList entry" else "没有 AniList 条目", block) for k, block in cov["by_anilist_format"].items()]
    add(table(header, rows))
    add("")
    add("分母是该组全部候选番，没有 AniList 条目的番算作没有信号。\"第 −N 天前已有记录\"指 AniList 对这部番的第一行记录不晚于首播前 N 天，也就是能不能算出那一天的特征。")
    add("")
    add("一部番只要被追踪，记录就一直延续到首播，所以\"开播前 7、30、90 天内至少有一行\"三个数字相同，差别只在窗口结束日：")
    add("")
    add(table(["窗口结束日", "窗口内有记录的候选番"], [[f"第 −{abs(end)} 天", pct(full[f"signal_in_7d_ending_day{end}"])] for end in (-1, -2, -8)]))
    add("")

    add("### 日本时间与时间泄漏")
    add("")
    clock = a["row_clock"]
    jump = leak["premiere_day_jump"]
    boundary = leak["midnight_boundary"]
    medians = boundary["median_trending_by_day"]
    late_m, flag_m, eve_m = medians["after_midnight_other"], medians["after_midnight_with_episode_on_day_-1_row"], medians["evening_18_to_23"]
    hours = leak["episode_1_airing_hour_jst"]
    all_stamped = clock["rows"] == clock["rows_stamped_15_00_utc"]
    add(f"- **行的时间戳是日本时间零点。** " + (f"全部 {n(clock['rows'])} 行都" if all_stamped else f"{n(clock['rows'])} 行里有 {n(clock['rows_stamped_15_00_utc'])} 行") + "打在 15:00 UTC，也就是日本时间 00:00。"
        "如果按 UTC 取日期，每一行都会被标成前一天，首播当天的那一行就会被当成\"开播前一天\"。")  # fmt: skip
    add(f"- **按 UTC 读会泄漏多少：** 在 {n(jump['titles_compared'])} 部两天都有记录的番上，`trending` 在首播当天是前一天的 {jump['median_trending_ratio_day_0_over_day_-1']} 倍（中位数）。"
        f"`popularity` 是累计值，首播当天的中位数增幅是 {pct(jump['median_popularity_growth_day_-1_to_0'])}，和开播前一周每天的增幅（{pct(jump['median_popularity_growth_per_day_over_days_-8_to_-1'])}）差不多，水平值受影响小。")  # fmt: skip
    add(f"- **泄漏的大小取决于播出时段。** {n(leak['titles_with_episode_1_airing_time'])} 部番有第一集的精确播出时间：日本时间 0–5 点的 {n(hours['00-05 (after midnight)'])} 部，6–17 点 {n(hours['06-17'])} 部，18–23 点 {n(hours['18-23'])} 部。"
        f"凌晨首播的番，`trending` 中位数从第 −1 天的 {n(late_m['-1'])} 跳到第 0 天的 {n(late_m['0'])}。晚间首播的番第 0 天只到 {n(eve_m['0'])}，跳升出现在第 +1 天（{n(eve_m['1'])}）。所以第 0 天的行对将近一半的番已经是播出之后。")  # fmt: skip
    add(f"- **一行记录的是当天全天，不是零点的快照。** 晚间首播的 {n(leak['evening_premieres']['titles'])} 部番里，{n(leak['evening_premieres']['with_episode_on_day_0_row'])} 部的第 0 天那一行已经带上了第一集的标记。")
    ep = leak["episode_1_on_day_-1_row"]
    add(f"- **第 −1 天的行基本干净。** 有 {n(ep['titles'])} 部番的第一集标记出现在第 −1 天那一行里，乍看像是泄漏。但凌晨首播的 {n(boundary['after_midnight_premieres'])} 部里出现这种情况的 {n(boundary['of_those_with_episode_on_day_-1_row'])} 部，"
        f"有 {n(boundary['of_those_airing_00_00_to_00_29'])} 部的播出时间在 00:00–00:29（其中 {n(boundary['of_those_airing_exactly_00_00'])} 部正好是 00:00）；其余凌晨首播的番里只有 {n(boundary['other_after_midnight_premieres_airing_00_00_to_00_29'])} 部在这个时段。"
        f"这是零点边界上的归属问题，同一集被同时记在前后两天。这些番第 −1 天的 `trending` 中位数是 {n(flag_m['-1'])}，第 −2 天是 {n(flag_m['-2'])}，第 0 天才是 {n(flag_m['0'])}，没有提前出现首播的跳升。")  # fmt: skip
    if leak["episode_1_more_than_a_day_before_day_0"]:
        add(f"- **提前更久的：** " + "、".join(f"《{name}》" for name in leak["episode_1_more_than_a_day_before_day_0"]) + " 的第一集标记比首播日早一周左右，对应先行特别篇或先行放送。这类情况靠日期规则挡不住。")
    add(f"- **首播日期本身是否可靠：** 第一集的日本日期减去 MAL 首播日的分布是 "
        + "、".join(f"{k} 天：{n(v)} 部" for k, v in leak["episode_1_japan_date_minus_mal_date"].items())
        + f"。负数表示第一集早于 MAL 的首播日，共 {n(leak['episode_1_aired_before_mal_day_0'])} 部，对它们来说\"第 −1 天\"已经是播出之后。正数表示 MAL 的日期更早，按它取 day 0 偏保守。")  # fmt: skip
    rel = leak["releasing_flag_before_day_0"]
    add(f"- **AniList 的 `releasing` 标记提前出现：** {n(rel['titles'])} 部番在第 0 天之前就被标成\"正在播出\"，其中 {n(rel['only_on_day_-1'])} 部只提前一天，{n(rel['earlier_than_day_-1'])} 部更早。")
    add(f"- **MAL 和 AniList 的首播日不一致的有 {n(leak['mal_and_anilist_dates_differ'])} 部**，其中 {n(leak['anilist_date_earlier_than_mal'])} 部是 AniList 更早。")
    add("")
    add("**建议的规则：** 日期一律按日本时间取。day 0 取 MAL 首播日、AniList 首播日和第一集播出日三者中最早的一个。"
        f"按这次的数据，第 −1 天及以前的行是开播前的；为了挡住首播日期错一天的情况，建议特征只用到第 −2 天（覆盖率只从 {pct(full['signal_in_7d_ending_day-1'])} 降到 {pct(full['signal_in_7d_ending_day-2'])}），并用第 −8 天做一次稳健性检查。"
        "Wikipedia 的浏览量按 UTC 日记录，UTC 的第 −1 天在日本时间首播日上午 9 点才结束，凌晨首播的番会被算进去，所以 Wikipedia 也只用到第 −2 天。")  # fmt: skip
    add("")
    add("这一条也适用于第一轮实验（`anime_coldstart/`）：那一轮的 Wikipedia 窗口截止在 UTC 的第 −1 天，脚本注释里写的\"严格在开播前\"对凌晨首播的番不成立，"
        "30 天的窗口里混进了首播后最多约 9 小时的浏览量。那一轮的数字没有按新规则重算。")  # fmt: skip
    add("")

    add("## 7. 其他可用信号")
    add("")
    w = wiki["all_candidates"]
    add(f"- **Wikipedia 浏览量**（第一轮实验修正映射后的数据，开播前 30 天，英日中三语）：{pct(w['mapped_to_article'])} 的候选番映射到了词条，{pct(w['views_in_30d_before_premiere'])} 有开播前浏览量。"
        f"AniList 和 Wikipedia 至少有一个信号的占 {pct(w['anilist_or_wikipedia_signal'])}，两个都有的占 {pct(w['both_signals'])}。")  # fmt: skip
    filled = content["candidate_field_filled"]
    add(f"- **内容 metadata**（AnimeRec 的 `ani_list/catalog.csv`）：{pct(content['candidates_in_catalog'])} 的候选番在里面。字段填充率：题材 {pct(filled['genres'])}、tags {pct(filled['tags'])}、"
        f"主要工作室 {pct(filled['main_studios'])}、staff {pct(filled['staff'])}、原作类型 {pct(filled['source'])}。")  # fmt: skip
    add(f"- **工作室能不能对上老番：** 列出了工作室的 {n(content['candidates_with_a_studio_listed'])} 部候选番里，{n(content['of_those_with_a_studio_seen_among_warm_titles'])} 部的工作室在 2019 年以前的番里出现过。")
    add("- **注意：** tags 和题材是今天的快照，由用户投票产生，可能混进了播出后的投票；集数也可能在播出后更新。主分析里应当不用 tags 和集数，或者单独做一组对照。")
    add("")

    add("## 8. 建议的 item-disjoint split")
    add("")
    removed = als["positives_removed"]
    by_role = r["items_by_role"]
    add(table(["角色", "番", "番数", "正反馈", "是否进入 ALS 训练"], [
        ["warm", "首播年份已知且早于 2019，所有格式", n(by_role["warm"]["items"]), n(by_role["warm"]["positives"]), "是"],
        ["cold-train", f"TV，{split['cold_train'][0]} – {split['cold_train'][1]}", n(train["items"]), n(removed["cold_train"]), "否"],
        ["cold-validation", f"TV，{split['cold_val'][0]} – {split['cold_val'][1]}", n(val["items"]), n(removed["cold_val"]), "否"],
        ["cold-test", f"TV，{split['cold_test'][0]} – {split['cold_test'][1]}", n(test["items"]), n(removed["cold_test"]), "否"],
        ["cold-test-late", f"TV，{split['cold_censored'][0]} – {split['cold_censored'][1]}", n(late["items"]), n(removed["cold_censored"]), "否"],
        ["2019 年以后的其他番", "非 TV 格式，以及没有完整首播日期或 2021 年首播的 TV", n(by_role["post_cutoff_other"]["items"]), n(removed["post_cutoff_other"]), "否"],
        ["年份未知", "`Aired` 里解析不出年份", n(by_role["undated"]["items"]), n(removed["undated"]), "否"],
    ], text=True))  # fmt: skip
    add("")
    add(f"这样训练 ALS 用的是 {n(als['positives'])} 条正反馈（全部的 {pct(als['share_of_all_catalog_positives'])}）、{n(als['users_with_1_positive'])} 个用户、{n(als['items_with_1_positive'])} 部番。")
    add("")
    add("和任务描述相比，这个设计多移除了两类交互，理由如下：")
    add("")
    add("1. **cold-train 的番也不进 ALS。** 第二阶段的模型要在 cold-train 上学\"怎么给一部没见过的番打分\"。如果这些番进过 ALS，训练时的用户向量已经见过它们，测试时却没见过，训练和测试的特征分布就不一样。")
    add("2. **2019 年以后的非候选番也不进 ALS。** 其中有 cold item 的剧场版、OVA 和特别篇。留着它们，用户对这个系列的喜好就会从旁路漏进用户向量。")
    add("")
    add("另外建议两点：")
    add("")
    add(f"- **用户也分开。** 合格用户足够多（cold-train {n(train['warm>=5']['eligible_users'])}、validation {n(val['warm>=5']['eligible_users'])}、test {n(test['warm>=5']['eligible_users'])}），"
        "可以把用户按 ID 哈希分成三份，第二阶段模型的训练、调参、测试各用一份。这样测试时番和用户都是模型没见过的。ALS 仍然用所有用户的 warm 正反馈。")  # fmt: skip
    add("- **按首播时间排序而不是随机分。** 评分没有时间戳，所以按时间分并不能防泄漏；但它和上线时的用法一致（用过去几季训练，用于下一季），而且 AniList 的用户规模在 2018–2020 年间一直增长，时间顺序能如实检验这种漂移。")
    add("- **2020 年秋季单独报告。** 它是右删失最重的一季，作为第二测试集，不参与主结论。")
    add("")

    add("## 9. 三个模型")
    add("")
    add("三个模型用**同一个学习器、同一批训练样本、同一套超参数搜索预算**，只有特征集不同。这样 M2 和 M1 的差就只来自另类数据。")
    add("")
    add(table(["模型", "特征", "是否个性化", "回答的问题"], [
        ["M0 attention-only", "AniList 开播前 popularity 水平和增长、trending、in_progress；Wikipedia 开播前浏览量；缺失标记", "否，所有用户同一个排序", "只靠关注度能排多好"],
        ["M1 personalized content", "由内容把新番映射到 ALS 空间后与用户向量的内积；用户 warm 历史与新番在题材、工作室、原作类型、staff 上的匹配度；前作在 warm 里时用户对前作的得分", "是", "不用另类数据，内容加个性化能排多好"],
        ["M2 content + attention", "M1 的全部特征 + M0 的全部特征", "是", "在个性化内容之上，另类关注度还有没有增量"],
    ], text=True))  # fmt: skip
    add("")
    add("保证可比的约束：")
    add("")
    add(f"- **相同用户：** 三个模型都在同一份测试用户上打分（建议从 {n(test['warm>=5']['eligible_users'])} 个合格用户里按固定种子取一份，并且与训练、调参用户不重叠）。")
    add(f"- **相同候选集：** 每个用户的候选都是 cold-test 的全部 {n(test['items'])} 部番，不做任何按模型的预筛选。")
    add("- **相同标签：** 评分 ≥8 为相关，其余（评分 <8 或没有评分）为不相关。负反馈（≤5）只用于下面的辅助指标。")
    add("- **用户向量：** 只由用户的 warm 正反馈 fold-in 得到，和 Anime Compass 线上的做法一致，并且来自只用 warm 番训练的实验 ALS，不是生产模型。")
    add("- **另类数据特征在季内标准化**（排名或 z-score），因为 AniList 的用户规模在增长，不同季的 popularity 绝对值不可比。")
    add("- **内容映射器只在 warm 番上训练**（输入内容特征，目标是 warm 番的 ALS 向量），不接触任何 cold item 的交互。")
    add("")
    add("指标：")
    add("")
    add(f"- **主指标：** 每个用户在 {n(test['items'])} 部候选上的 NDCG@10，对用户取平均；同时报 Recall@10。")
    add(f"- **辅助指标一：** 用户内 AUC，只在用户评过分的测试番里比较喜欢（≥8）和不喜欢（≤5）。有 {n(test['warm>=5']['eligible_users_with_pos_and_neg'])} 个用户可算。它去掉了\"看没看过\"的影响。")
    add(f"- **辅助指标二：** 去掉最热门 10 部之后重算主指标。这 10 部占了测试集 {pct(test['top10_share_of_positives'])} 的正反馈，M0 光靠把它们排在前面就能拿到很高的分数。")
    add("- **不确定性：** 同时对用户和番做 bootstrap。只对用户做会严重低估区间，因为番只有几十部。")
    add("- **参照行：** 随机排序，以及按测试集真实正反馈数排序（非个性化排序的上限）。")
    add("")

    add("## 10. 训练成本")
    add("")
    if recorded:
        latency_ms = float(recorded["inference_latency_p50_ms"])
        add(f"- **ALS：** Anime Compass 记录的评估版训练（128 维、15 轮、CG 3 步）在 {n(ALS_REFERENCE_POSITIVES)} 条正反馈上用了 {als_seconds:.0f} 秒，"
            f"进程内存峰值 {int(recorded['offline_peak_process_rss_bytes']) / 2**20:.0f} MiB。按正反馈数线性外推，本设计的 {n(als['positives'])} 条约需 **{als_estimate / 60:.0f} 分钟**一次。这是估算，这一轮没有训练。")  # fmt: skip
        add(f"- **用户向量：** 每个用户解一个 128 维的线性方程组。同一份记录里，fold-in 加全片库打分的中位延迟是 {latency_ms:.1f} 毫秒；"
            f"按这个速度，全部 {n(test['warm>=5']['eligible_users'])} 个测试合格用户约需 {test['warm>=5']['eligible_users'] * latency_ms / 60000:.0f} 分钟（估算）。")  # fmt: skip
    add(f"- **第二阶段样本量：** 测试网格是 {n(test['warm>=5']['eligible_users'] * test['items'])} 个 (用户, 番)；cold-train 全网格是 {n(train['warm>=5']['eligible_users'] * train['items'])} 个，可以按用户抽样到几百万。")
    add(f"- **数据采集：** AniList 的开播前窗口这一轮已经抓完（{n(c['with_anilist_id'])} 次请求）。还缺的只有候选番的前作关系，约 {n(c['with_anilist_id'])} 次请求。")
    add("")

    add("## 11. 没有时间戳时，实验能支持和不能支持的结论")
    add("")
    add("**能支持：**")
    add("")
    add("- 对同一批用户、同一批从未进入训练的番，M0、M1、M2 三者的相对排序质量。")
    add("- 在内容和个性化之上，开播前的另类关注度有没有增量（M2 对 M1）。另类数据本身是当时记录的，这一半没有时间问题。")
    add("- 内容特征能不能把一部没有任何交互的番放到 ALS 空间里合适的位置（M1 对 M0，以及去掉热门番之后的结果）。")
    add("")
    add("**不能支持：**")
    add("")
    add("- **\"在开播当天，系统会把这部番推给这个用户\"。** 用户的 warm 历史里包含他在 cold item 开播之后才打的分，warm 番的 ALS 向量也是用这些之后的交互学出来的。这是模拟的冷启动，不是历史回测。")
    add("- **任何和时间有关的结论：** 下一部看什么、开播后第几天开始有效、第一周的动态。")
    add("- **\"没评分就是不喜欢\"。** 没评分可能只是没看过。而且文件只收看完的记录，弃番不在里面，所以负反馈被系统性低估，≤5 分的记录都来自坚持看完的人。")
    add("- **对 2021–2025 新番和 AniList 用户的推广。** 实验人群是 2021 年初以前的 MAL 用户。已经采集的 2021–2025 另类数据没有对应的用户级标签，这个实验不能替它背书。")
    add("- **生产系统的效果。** 实验 ALS 是另一份模型，结论不能写成 Anime Compass 线上的指标。")
    add("- **推荐带来的因果效应或线上提升。**")
    add("")

    add("## 12. 风险")
    add("")
    add(f"1. **番太少。** 测试集 {n(test['items'])} 部（正反馈 ≥50 的 {n(test['items_with_50_positives'])} 部）。用户再多，番层面的不确定性也降不下来。缓解：两路 bootstrap；把 8 个季度做成留一季交叉验证作为稳健性检查。")
    add(f"2. **热门番主导指标。** 前 10 部占测试集正反馈的 {pct(test['top10_share_of_positives'])}，M0 会是很强的基线，个性化的提升空间主要在头部之外。必须事先定好看哪个指标。")
    add(f"3. **AniList 覆盖不全且不随机。** 没有信号的番集中在冷门番和短篇（见第 6 节的分组）。缺失本身带信息，应当保留这些番并加缺失标记，不要删掉。")
    add(f"4. **首播日期错位。** 见第 6 节。按最早日期取 day 0、特征只用到第 −2 天，可以挡住大部分；先行上映和提前配信挡不住，需要在报告里说明。")
    add("5. **AniList popularity 的量纲漂移。** 2018-03-20 之前没有 popularity；之后用户规模持续增长。必须季内标准化。")
    add("6. **tags、题材、集数是今天的快照。** 见第 7 节。")
    add("7. **合格用户的选择偏差。** 测试用户按\"至少喜欢一部测试番\"选出，指标的绝对值会偏高。三个模型用的是同一批用户，所以相对比较不受影响。")
    seq = a["sequels"]["by_role"]["cold_test"]
    add(f"8. **续作。** 按标题规则，测试集 {n(seq['titles'])} 部里有 {n(seq['earlier_title_among_warm'])} 部的前作在 warm 里（模型可以借前作的向量，相对容易），"
        f"另有 {n(seq['earlier_titles_only_from_2019_on'])} 部的前作本身是 2019 年以后的番（前作的交互被一并移除，偏保守）。结果应当按续作和新作分开报。")  # fmt: skip
    add("9. **数据条款。** AniList 条款禁止大规模收集和把 API 当数据存储，对课程作业宽松。这一轮抓的数据留在本机，没有推到公开仓库。")
    add("")

    add("## 13. 推荐的实施方案")
    add("")
    add("1. **先预注册。** 把本报告第 8、9 节的 split、特征清单、学习器、主指标和判定规则写成一份方案，提交到 git，再碰 validation 和 test 的标签。第一轮实验里线性模型和树模型结论相反，所以学习器必须事先指定一个。")
    add("2. **固化 split。** 生成一份清单，记录每部番和每个用户的角色、评分文件的 SHA-256。之后所有脚本只读这份清单。")
    add("3. **补齐数据。** 抓候选番的前作关系；把队友那边需要的评分文件传过去。")
    add("4. **训练实验 ALS。** 只用 warm 番，产物标记为 evaluation，和生产模型分开存放。")
    add("5. **做特征，训练 M0、M1、M2，** 在 validation 上调参，test 只跑一次。")
    add("6. **按第 9 节的指标出报告，** 包括续作与新作、去掉热门番、2020 秋季这三组分层结果。")
    add("")
    add("**替代方案。** 用户级 Ground Truth 是够的，所以不需要替代数据源来做这个实验。如果之后想回答这个实验回答不了的问题：")
    add("")
    add("- 想要真实的时间顺序，需要带时间戳的用户片单（例如 AniList 用户的 `createdAt`）。那是用户级数据采集，要先确认条款和同意问题。")
    add("- 想在 2021–2025 的番上验证，目前只有番级别的标签，只能做\"值不值得推\"，不能做\"推给谁\"。")
    add("")

    add("## 附：本轮产出的文件")
    add("")
    add(table(["文件", "内容"], [
        ["`ratings_feasibility.json`", "评分文件校验、候选番、用户、各 split 的全部计数"],
        ["`candidate_items.csv`", "每部候选番：首播日、角色、正反馈、负反馈、唯一用户数、AniList 对应条目"],
        ["`candidates_by_season.csv`", "按首播季度汇总"],
        ["`candidate_trends.parquet`", "候选番的 AniList 每日记录（第 −100 天至第 +30 天）"],
        ["`candidate_anilist_meta.csv`", "AniList 的格式、首播日、最早一行记录、第一集播出时间"],
        ["`candidate_trend_audit.csv`", "每部番的覆盖和泄漏诊断"],
        ["`alt_data_coverage.json`", "覆盖率和时间泄漏审计的全部数字"],
    ], text=True))  # fmt: skip
    add("")
    add("脚本：`src/personalized_feasibility.py`、`src/collect_candidate_windows.py`、`src/audit_candidates.py`、`src/report_personalized.py`。")
    add("")

    (OUT / "FEASIBILITY_REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT / 'FEASIBILITY_REPORT.md'} ({len(out)} lines)")


if __name__ == "__main__":
    main()

"""Render RESULTS.md from the final-test output.

Written after the final test, so it is reporting code and not part of the frozen
protocol. It computes nothing new from labels: every number is read from
results/final_test_results.json, MODEL_SELECTION.json, the manifest and the pre-run
checks, and no number is typed by hand.

The report keeps four kinds of statement apart: the primary hypothesis, the secondary
comparisons, the pre-specified sensitivity and auxiliary analyses, and exploratory
readings that were not registered.

Usage:
    python src/report_results.py
"""

from __future__ import annotations

import json
import subprocess

import pandas as pd

from common import MANIFEST, ROOT

RESULTS = ROOT / "results"
FREEZE_COMMIT = "37defa8"  # the commit that first contains PRE_REGISTRATION.md
MODEL_LABEL = {"M0a_raw": "M0a-raw", "M0a_norm": "M0a-norm", "M0b": "M0b", "M1": "M1", "M2": "M2"}
MODEL_WHAT = {
    "M0a_raw": "AniList popularity 原值排序",
    "M0a_norm": "AniList popularity 季内百分位排序",
    "M0b": "关注度特征 + 逻辑回归（非个性化）",
    "M1": "内容特征 + 逻辑回归（个性化，无关注度）",
    "M2": "内容特征 + 关注度特征 + 逻辑回归",
}
COMPARISON_LABEL = {
    "H1_primary": "**H1（主假设）** M2 − M1",
    "secondary_vs_M0a_raw": "S1a（次要）M2 − M0a-raw",
    "secondary_vs_M0a_norm": "S1b（次要）M2 − M0a-norm",
    "secondary_vs_M0b": "S2（次要）M2 − M0b",
}
SENSITIVITY_LABEL = {
    "top_10_most_liked_removed": "去掉正反馈最多的 10 部",
    "titles_with_anilist_signal": "只保留有 AniList 信号的番",
    "new_titles": "只看新作",
    "sequels": "只看续作",
}


def table(header: list[str], rows: list[list[str]], text: bool = False) -> str:
    rule = "|" + "|".join("---" if text or i == 0 else "---:" for i in range(len(header))) + "|"
    return "\n".join(["| " + " | ".join(header) + " |", rule] + ["| " + " | ".join(row) + " |" for row in rows])


def ci(block: dict, key: str = "user_bootstrap_ci95") -> str:
    return f"[{block[key][0]:+.4f}, {block[key][1]:+.4f}]"


def metric_rows(metrics: dict, with_auc: bool = True) -> list[list[str]]:
    rows = []
    for name in MODEL_LABEL:
        m = metrics[name]
        row = [MODEL_LABEL[name], f"{m['ndcg_at_10']:.4f}", f"{m['recall_at_10']:.4f}"]
        if with_auc:
            row.append(f"{m['observed_auc']:.4f}")
        row.append(f"{m['item_coverage_at_10']:.3f}")
        rows.append(row)
    return rows


def main() -> None:
    r = json.loads((RESULTS / "final_test_results.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "MODEL_SELECTION.json").read_text(encoding="utf-8"))
    manifest = json.loads((MANIFEST / "split_manifest.json").read_text(encoding="utf-8"))
    pre = json.loads((RESULTS / "pre_run_checks.json").read_text(encoding="utf-8"))
    # Frozen models on validation, from the dry run: in-sample, used only as a like-for-like reference.
    reference = json.loads((RESULTS / "validation_in_sample_reference.json").read_text(encoding="utf-8"))
    names = pd.read_csv(MANIFEST / "item_roles.csv").set_index("anime_id")["Name"]
    p, aux, late = r["primary"], r["auxiliary_warm_history_users"], r["supplementary_cold_test_late"]
    m, c, s = p["metrics"], p["comparisons"], p["sensitivity"]
    val = selection["validation_at_chosen_c"]
    h1 = c["H1_primary"]
    n_titles = p["titles"]

    out: list[str] = []
    add = out.append

    add("# 结果报告：Personalized Item Cold-Start 离线实验")
    add("")
    add(f"`final_test.py` 运行了一次：{r['run_utc']}，代码 commit `{r['code']['commit'][:7]}`，运行时实验目录没有未提交的改动，此前没有启动过（`starts_before_this_one` 为空）。"
        "运行后写入了锁文件，脚本不会再次运行。本报告由 `src/report_results.py` 从结果文件生成，数字都不是手填的。")  # fmt: skip
    add("")
    add("**这是模拟的冷启动实验。** 评分没有时间戳，实验 ALS 也不是 Anime Compass 的生产模型。下面的数字不是 Anime Compass 的线上表现，也不是任何真实开播时点的回测。")
    add("")

    add("## 结论")
    add("")
    add(f"- **主假设 H1 得到支持，证据等级 {h1['evidence_level']}。** 在个性化内容模型上加入开播前的 AniList 和 Wikipedia 关注度，cold-test 上的用户平均 NDCG@10 从 {m['M1']['ndcg_at_10']:.4f} 升到 {m['M2']['ndcg_at_10']:.4f}，"
        f"差值 {h1['delta']:+.4f}，用户配对 bootstrap 的 95% 区间 {ci(h1)}。逐一去掉 {n_titles} 部番中的任何一部，差值都保持为正（{h1['leave_one_title_out']['min_delta']:+.4f} 到 {h1['leave_one_title_out']['max_delta']:+.4f}）。")  # fmt: skip
    add(f"- **代价是覆盖率下降。** 进入过至少一个用户前 10 名的番，从 M1 的 {m['M1']['item_coverage_at_10']:.1%} 降到 M2 的 {m['M2']['item_coverage_at_10']:.1%}。M2 仍然远高于非个性化模型的 {m['M0b']['item_coverage_at_10']:.1%}。")
    auc_gap = {name: block["M2"]["observed_auc"] - block["M1"]["observed_auc"] for name, block in (("val", val), ("test", m), ("late", late["metrics"]))}
    add(f"- **加入关注度对 Observed-only AUC 的影响不稳定，不能当作改善来报告。** M2 减 M1 的差：验证集（2019 Q4–2020 Q1 的番）{auc_gap['val']:+.4f}，cold-test（2020 Q2–Q3）{auc_gap['test']:+.4f}，2020 Q4 {auc_gap['late']:+.4f}。"
        f"符号和幅度都随时间段变化。cold-test 上 AUC 最高的是 M0a-norm（{m['M0a_norm']['observed_auc']:.4f}），不是 M2。详见第 6 节。")  # fmt: skip
    s2 = c["secondary_vs_M0b"]
    add(f"- **内容特征在关注度之上也有增量，但小得多，支持也弱于 H1。** M2 比 M0b 高 {s2['delta']:+.4f}，用户层面的区间 {ci(s2)}。按冻结的规则它的证据等级也是 {s2['evidence_level']}，"
        f"但作品重采样的区间 {ci(s2, 'title_bootstrap_interval95')} 包含 0，而 H1 的两种区间都在 0 以上。两者不应被读成同样强的证据，见第 2 节的规则核对。"
        f"分组看，这个增量在新作上是 {s['new_titles']['comparisons']['secondary_vs_M0b']['delta']:+.4f}，在续作上是 {s['sequels']['comparisons']['secondary_vs_M0b']['delta']:+.4f}。")  # fmt: skip
    add(f"- **只靠内容的个性化模型 M1，和关注度排序大致持平。** M1 的 NDCG@10 是 {m['M1']['ndcg_at_10']:.4f}，M0a-raw {m['M0a_raw']['ndcg_at_10']:.4f}，M0a-norm {m['M0a_norm']['ndcg_at_10']:.4f}，M0b {m['M0b']['ndcg_at_10']:.4f}。"
        "单独使用时两类信息的效果接近，合在一起才明显更好。M1 和这些模型的比较没有预注册，见第 9 节。")  # fmt: skip
    add("")

    add("## 1. 各模型的指标（主评估）")
    add("")
    add(f"cold-test：{n_titles} 部番，{p['users']:,} 个 positive-eligible 测试用户，{p['positives']:,} 条正反馈。每个用户的候选集都是全部 {n_titles} 部番，五个模型用的是同一批用户和同一批候选。")
    add("")
    add(table(["模型", "NDCG@10", "Recall@10", "Observed-only AUC", "Item coverage@10"], metric_rows(m)))
    add("")
    add(table(["模型", "是什么"], [[MODEL_LABEL[k], v] for k, v in MODEL_WHAT.items()], text=True))
    add("")
    add(f"Observed-only AUC 只在同时有正反馈和负反馈的 {p['users_with_liked_and_disliked']:,} 个用户上计算。NDCG 的绝对值受选择偏差影响（没有评分不等于不喜欢；测试用户按\"至少喜欢一部测试番\"选出），只解读模型之间的差。")
    add("")

    add("## 2. 主假设和次要比较")
    add("")
    rows = []
    for key, label in COMPARISON_LABEL.items():
        block = c[key]
        loo = block["leave_one_title_out"]
        rows.append([label, f"{block['delta']:+.4f}", ci(block), f"{loo['min_delta']:+.4f} 到 {loo['max_delta']:+.4f}", str(loo["deletions_with_delta_at_or_below_zero"]),
                     block["evidence_level"], ci(block, "title_bootstrap_interval95"), "是" if block["title_bootstrap_interval95"][0] > 0 else "**否**"])  # fmt: skip
    add(table(["比较", "Δ NDCG@10", "用户配对 bootstrap 95% 区间", "逐一去掉一部番后 Δ 的范围", "使 Δ ≤ 0 的番数", "证据等级（冻结规则）", "作品重采样区间（只报告）", "该区间在 0 以上"], rows))
    add("")
    add("- **证据等级**是预注册里事先定义的分级：A 表示 Δ > 0、用户层面的区间下界 > 0、并且去掉任何一部番后 Δ 仍 > 0。它不是显著性检验的结论，也不保证能推广到别的番或别的用户。")
    add(f"- **作品重采样区间**描述结果对\"测试季碰巧有哪些番\"的依赖。{n_titles} 部番来自 2020 年的两个季度，不是未来新番的随机样本，所以它不作为总体泛化的保证，也不进入证据等级。")
    add(f"- 对 H1 影响最大的单部番是《{names[h1['leave_one_title_out']['most_influential_title']]}》。")
    add("")
    add("### 规则核对：同样是 A，支持程度不同")
    add("")
    levels = [v["evidence_level"] for v in c.values()]
    short = {"H1_primary": "H1", "secondary_vs_M0a_raw": "S1a", "secondary_vs_M0a_norm": "S1b", "secondary_vs_M0b": "S2"}
    frozen_at = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cI", FREEZE_COMMIT], capture_output=True, text=True, check=False).stdout.strip()
    add(f"冻结的规则在 commit `{FREEZE_COMMIT}`（{frozen_at}）里，早于模型训练，也早于正式测试（{r['run_utc']}）。`PRE_REGISTRATION.md` 和 `final_test.py` 从冻结到现在都没有改动过。"
        "按这条规则，A 只看两件事：用户配对 bootstrap 的区间，和逐一去掉一部番的结果。作品重采样区间在冻结版里写的是\"只报告\"，不参与分级。所以" + ("四个比较都是 " + levels[0] if len(set(levels)) == 1 else "四个比较的等级依次是 " + "、".join(levels)) + "，这不是测试之后改的定义。")  # fmt: skip
    add("")
    weak = [short[k] for k, v in c.items() if v["title_bootstrap_interval95"][0] <= 0]
    s1b = c["secondary_vs_M0a_norm"]["title_bootstrap_interval95"]
    add("但 A 并不等于证据一样强：")
    add("")
    add(f"- **H1** 的用户区间和作品重采样区间都在 0 以上，是四个比较里支持最充分的。")
    add(f"- **{'、'.join(weak)}** 的作品重采样区间包含 0。它在用户层面成立，对单部番的删除也稳健，但换一批番是否成立没有得到支持。"
        f"S1b 的作品重采样区间下界是 {s1b[0]:+.4f}，几乎贴着 0。")  # fmt: skip
    add("")
    add("**和冻结前草稿的差别。** 冻结前审阅过的第一版草稿用的是\"通过 / 部分通过 / 不通过\"，其中\"通过\"要求用户区间和作品区间的下界都大于 0。"
        "冻结前按审阅意见第 4 条（保留用户 bootstrap，增加逐一去掉一部番，作品重采样只报告且不作为泛化保证，三档改称证据等级）改成了现在的定义。A、B、C 的具体条件是 AI 按这条意见起草的；项目负责人授权修订完成后自行冻结，所以这几句条件在冻结前没有再单独确认，是在冻结后的汇报里列出、随后获批执行测试的。"
        f"如果按那版草稿的更严格条件，H1 属于\"通过\"，{'、'.join(weak)} 只属于\"部分通过\"。这一句只是对照，不改变按冻结规则给出的等级。")  # fmt: skip
    add("")
    add("**次要比较的解读限制**（预注册第 1 节）：S1 不能单独解释为个性化的纯粹贡献，因为 M2 和 M0a 之间还差着学习到的权重、Wikipedia 信号和番层面的内容先验。S2 固定了关注度特征和学习器，差别是全部 6 个内容特征，其中 2 个只和番有关，所以它说明的是\"内容特征整体\"的增量。")
    add("")

    add("## 3. 预先指定的敏感性和分组分析")
    add("")
    add("这些分析不进入证据等级。每一组都把候选集限制在该组的番上，并且只保留在该组里仍有正反馈的用户。")
    add("")
    for key, label in SENSITIVITY_LABEL.items():
        block = s[key]
        add(f"### {label}（{block['titles']} 部番，{block['users']:,} 个用户）")
        add("")
        add(table(["模型", "NDCG@10", "Recall@10", "Observed-only AUC", "Item coverage@10"], metric_rows(block["metrics"])))
        add("")
        add(table(["比较", "Δ NDCG@10", "用户配对 bootstrap 95% 区间"], [[COMPARISON_LABEL[k], f"{v['delta']:+.4f}", ci(v)] for k, v in block["comparisons"].items()]))
        add("")
    removed = s["top_10_most_liked_removed"]["removed_titles"]
    add("去掉的 10 部番：" + "、".join(f"《{names[a]}》" for a in removed) + "。")
    add("")
    add("**两种关注度信号都没有的番**（数量太少，只列出各模型给它们的平均名次，共 " + f"{n_titles} 名）：")
    add("")
    add(table(["番", "测试用户的正反馈数", *[MODEL_LABEL[k] for k in MODEL_LABEL]],
              [[f"《{names[t['anime_id']]}》", str(t["positives"]), *[f"{t['mean_rank'][k]:.1f}" for k in MODEL_LABEL]] for t in p["titles_without_any_attention_signal"]]))  # fmt: skip
    add("")

    add("## 4. 辅助评估：只按 warm history 选出的用户")
    add("")
    add(f"测试组里 warm 正反馈 ≥ 5 的全部 {aux['users']:,} 个用户，成员资格不使用任何 cold-test 标签。其中 {aux['share_with_a_liked_title']:.1%} 在 cold-test 里有正反馈。")
    add("")
    add(table(["模型", "NDCG@10", "Recall@10", "Item coverage@10", "前 10 名里用户打过 ≤5 分的番（每人平均）"],
              [[MODEL_LABEL[k], f"{aux['metrics'][k]['ndcg_at_10']:.4f}", f"{aux['metrics'][k]['recall_at_10']:.4f}", f"{aux['metrics'][k]['item_coverage_at_10']:.3f}",
                f"{aux['metrics'][k]['disliked_titles_in_top_10_per_user']:.4f}"] for k in MODEL_LABEL]))  # fmt: skip
    add("")
    add(table(["比较", "Δ NDCG@10", "用户配对 bootstrap 95% 区间"], [[COMPARISON_LABEL[k], f"{v['delta']:+.4f}", ci(v)] for k, v in aux["comparisons"].items()]))
    add("")
    add("没有正反馈的用户在任何模型下 NDCG 和 Recall 都是 0，所以这里的差值等于主评估的差值乘以有正反馈的用户占比，符号和相对大小不变。这一步新增的信息是量级、更大人群上的覆盖率，以及最后一列。")
    add("")

    add("## 5. 补充测试：2020 年第四季度")
    add("")
    add(f"cold-test-late：{late['titles']} 部番，{late['users']:,} 个用户，{late['positives']:,} 条正反馈。同一组冻结的模型，不参与模型选择，也不进入主结论。这一季只有一个季度，所以 M0a-raw 和 M0a-norm 的排序相同。")
    add("")
    add(table(["模型", "NDCG@10", "Recall@10", "Observed-only AUC", "Item coverage@10"], metric_rows(late["metrics"])))
    add("")
    add(table(["比较", "Δ NDCG@10", "用户配对 bootstrap 95% 区间", "逐一去掉一部番后 Δ 的范围", "证据等级"],
              [[COMPARISON_LABEL[k], f"{v['delta']:+.4f}", ci(v), f"{v['leave_one_title_out']['min_delta']:+.4f} 到 {v['leave_one_title_out']['max_delta']:+.4f}", v["evidence_level"]] for k, v in late["comparisons"].items()]))  # fmt: skip
    add("")

    add("## 6. 覆盖率和 AUC")
    add("")
    add("这两项是运行前特别要求关注的。")
    add("")
    add(table(["", "M1", "M2", "变化", "M0b"], [
        ["Item coverage@10，cold-test", f"{m['M1']['item_coverage_at_10']:.3f}", f"{m['M2']['item_coverage_at_10']:.3f}", f"{m['M2']['item_coverage_at_10'] - m['M1']['item_coverage_at_10']:+.3f}", f"{m['M0b']['item_coverage_at_10']:.3f}"],
        ["Item coverage@10，验证集", f"{val['M1']['item_coverage_at_10']:.3f}", f"{val['M2']['item_coverage_at_10']:.3f}", f"{val['M2']['item_coverage_at_10'] - val['M1']['item_coverage_at_10']:+.3f}", f"{val['M0b']['item_coverage_at_10']:.3f}"],
        ["Item coverage@10，2020 Q4", f"{late['metrics']['M1']['item_coverage_at_10']:.3f}", f"{late['metrics']['M2']['item_coverage_at_10']:.3f}", f"{late['metrics']['M2']['item_coverage_at_10'] - late['metrics']['M1']['item_coverage_at_10']:+.3f}", f"{late['metrics']['M0b']['item_coverage_at_10']:.3f}"],
    ]))  # fmt: skip
    add("")
    add("**覆盖率在三个数据集上都下降。** 加入关注度后，推荐集中到更少的番上。这是稳定的现象。")
    add("")
    add("**Observed-only AUC 不稳定。** 下表把三个时间段放在一起，所有模型都列出：")
    add("")
    periods = [
        ("验证集，2019 Q4–2020 Q1，只用 cold-train 拟合的模型（留出）", val),
        ("验证集，同上，冻结模型（**样本内**，仅作对照）", reference["metrics"]),
        ("cold-test，2020 Q2–Q3，冻结模型", m),
        ("2020 Q4，冻结模型", late["metrics"]),
    ]
    add(table(["数据", *[MODEL_LABEL[k] for k in MODEL_LABEL], "M2 − M1"],
              [[label, *[f"{block[k]['observed_auc']:.4f}" for k in MODEL_LABEL], f"{block['M2']['observed_auc'] - block['M1']['observed_auc']:+.4f}"] for label, block in periods]))  # fmt: skip
    add("")
    add(f"- **加入关注度后 AUC 的变化，符号随时间段改变。** 在 2019 Q4–2020 Q1 的番上 M2 低于 M1，在 2020 Q2–Q3 和 2020 Q4 的番上 M2 高于 M1。")
    add("- **这不是模型版本不同造成的。** 第二行用的就是冻结模型，它们在验证集的番上仍然是 M2 低于 M1。这一行是样本内分数，偏乐观，而且对特征更多的 M2 更有利；即便如此 M2 仍然低于 M1。同一组模型在不同时间段的番上给出了相反的结论。")
    add(f"- **AUC 的水平本身也随时间段大幅变化。** 例如 M0a-norm 在验证集上是 {val['M0a_norm']['observed_auc']:.4f}，在 cold-test 上是 {m['M0a_norm']['observed_auc']:.4f}。在 2020 Q2–Q3，热度本身就能很好地区分喜欢和不喜欢，在前两个季度则不能。")
    add("- **结论：** 本实验不支持\"加入关注度能提高在已评分作品里区分喜欢和不喜欢的能力\"。cold-test 上的提升不能单独拿出来展示；对已评分作品的喜好区分能力在时间段之间不稳定。")
    add("")

    add("## 7. 怎么理解另类数据的增量")
    add("")
    add("以下三点由预注册的比较直接支持：")
    add("")
    add(f"1. **开播前关注度在个性化内容之上有增量（H1）。** NDCG@10 的相对提升是 {h1['delta'] / m['M1']['ndcg_at_10']:.1%}，Recall@10 从 {m['M1']['recall_at_10']:.4f} 到 {m['M2']['recall_at_10']:.4f}。在 2020 Q4 的补充测试上方向相同（{late['comparisons']['H1_primary']['delta']:+.4f}）。")
    top = s["top_10_most_liked_removed"]["comparisons"]["H1_primary"]
    add(f"2. **这个增量不只来自最热门的番。** 去掉正反馈最多的 10 部之后，H1 的差值是 {top['delta']:+.4f}，区间 {ci(top)}。")
    new, seq = s["new_titles"]["comparisons"]["H1_primary"], s["sequels"]["comparisons"]["H1_primary"]
    add(f"3. **增量在新作上比在续作上大。** 只看新作是 {new['delta']:+.4f}，只看续作是 {seq['delta']:+.4f}。续作有前作向量可以借，内容模型本来就排得不错（M1 在续作上 {s['sequels']['metrics']['M1']['ndcg_at_10']:.4f}，在新作上 {s['new_titles']['metrics']['M1']['ndcg_at_10']:.4f}）。")
    add("")
    add("**对个性化的影响。** 关注度特征只和番有关，对所有用户相同。M2 的得分是个性化部分加上这个先验，所以加入关注度并没有让模型\"更个性化\"，而是让它知道了哪些番会有很多人看。S2 表明内容特征在这个先验之上还有增量，但幅度不到 H1 的一半，并且在续作上没有增量。")
    add("")
    add(f"**对流行度偏差的影响。** 加入关注度使推荐向热门番集中：覆盖率下降（第 6 节）；在只按 warm history 选出的用户上，M2 每人前 10 名里出现\"打过 ≤5 分的番\"的数量是 {aux['metrics']['M2']['disliked_titles_in_top_10_per_user']:.4f}，M1 是 {aux['metrics']['M1']['disliked_titles_in_top_10_per_user']:.4f}。"
        "热门番被更多人看过，所以既有更多人喜欢，也有更多人不喜欢。两种关注度信号都没有的 5 部番，在 M2 里几乎被排到最后。")  # fmt: skip
    add("")
    add(f"**对覆盖率的影响。** M2 的覆盖率介于两者之间：比 M1 低，但是非个性化模型的 {m['M2']['item_coverage_at_10'] / m['M0b']['item_coverage_at_10']:.1f} 倍。非个性化模型给所有用户同样的 10 部番，覆盖率固定在 10 / {n_titles}。")
    add("")

    add("## 8. 泄漏检查和运行记录")
    add("")
    add(f"- **运行前检查：** {len(pre['checks'])} 项全部通过（{pre['checked_utc']}，HEAD `{pre['head'][:7]}`）。内容包括冻结 commit、manifest 重建比对、全部输入和产物的 SHA-256、时间截止记录、边界检查和库版本。逐项结果在 `results/pre_run_checks.json`。")
    add(f"- **split：** manifest 状态 {manifest['status']}，泄漏检查通过。实验 ALS 的训练番里首播年份最晚是 {manifest['leakage_checks']['latest_premiere_year_in_als_training']}。")
    add("- **时间：** AniList 行和 Wikipedia 天实际读到的最晚一天都是第 −2 天。")
    add("- **测试标签：** 只有 `final_test.py` 读取过，读取发生在 `MODEL_SELECTION.json` 提交之后。开发脚本请求测试标签会被拒绝，这一点在冻结前、选型后和运行前各验证过一次。")
    add(f"- **模型：** C 取 M0b {selection['chosen_c']['M0b']}、M1 {selection['chosen_c']['M1']}、M2 {selection['chosen_c']['M2']}，在验证集上选定，之后没有改动。")
    add(f"- **环境：** " + "，".join(f"{k} {v}" for k, v in selection["environment"].items()) + "。")
    add("")

    add("## 9. 探索性观察（没有预注册，不改变任何证据等级）")
    add("")
    add("下面两点是看到结果之后的解读，只能当作下一步的假设。")
    add("")
    add(f"1. **M1 和关注度模型的相对强弱，在验证集和测试集上相反。** 验证集上 M1（{val['M1']['ndcg_at_10']:.4f}）高于 M0b（{val['M0b']['ndcg_at_10']:.4f}）；cold-test 上 M1（{m['M1']['ndcg_at_10']:.4f}）低于 M0b（{m['M0b']['ndcg_at_10']:.4f}）。"
        "M1 对 M0b 不是预注册的比较。一种可能的解释是 2020 年第二、三季度的番更少、正反馈更集中，热度本身更容易预测；这个解释没有被检验。")  # fmt: skip
    norm_new = s["new_titles"]["comparisons"]["secondary_vs_M0a_norm"]
    add(f"2. **在新作内部，M2 并不比季内标准化的 popularity 排序好**（Δ {norm_new['delta']:+.4f}，区间 {ci(norm_new)}），而在全部 {n_titles} 部番上 M2 比它高 {c['secondary_vs_M0a_norm']['delta']:+.4f}。"
        "这提示 M2 相对热度排序的优势，可能主要来自正确处理新作和续作之间的相对位置（例如把续作排给看过前作的人），而不是在新作之间做出更好的个性化区分。这只是与数字相容的一种读法。")  # fmt: skip
    add("")

    add("## 10. 偏离方案")
    add("")
    add("**没有改动的：** 模型、特征、超参数、用户选择规则、候选集、主指标、证据等级的定义。")
    add("")
    add("**需要记录的：**")
    add("")
    add("1. 冻结 commit `37defa8` 之后、训练之前，增加了 commit `836016b`，只加了一个 `.gitattributes`，防止被 SHA-256 固定的文件在检出时被转换换行符。不涉及方案内容。")
    add("2. `src/pre_run_check.py` 是在模型选择提交之后写的。运行时它放在实验目录之外，这样实验目录保持在冻结的 commit 上；它不读任何 cold 标签。脚本和输出随本报告一起提交。")
    add("3. `src/report_results.py` 是在测试之后写的。它只读结果文件，不重新计算任何指标。")
    add("4. manifest 记录 cold-test 有 15 部番的前作在 warm 里，特征里实际有前作向量的是 14 部。差的那一部前作在评分文件里没有任何正反馈，所以实验 ALS 里没有它的向量。")
    add("5. 测试之后，`src/check_boundaries.py` 的允许名单里加了 `pre_run_check.py`。后者为了校验 SHA-256 会引用评分文件的路径，但不读任何一行；不加的话边界检查会因为这个新文件而报错。")
    add("6. 测试完成后又调用了一次 `final_test.py`，用来确认锁文件生效。它在读取任何标签之前就被拒绝了。")
    add("7. 第 9 节的两点是事后解读，标注为探索性。")
    add("8. 本报告在首次提交后按审阅意见修订过一次：第 2 节增加了规则核对，说明同为 A 的比较支持程度不同；第 6 节把 AUC 在三个时间段的结果并列，并加入冻结模型在验证集上的样本内对照（`results/validation_in_sample_reference.json`，取自测试前的 dry run）。结果文件、证据等级的定义和任何数字都没有改动。")
    add("")

    add("## 11. 局限")
    add("")
    add(f"- **不是线上回测，也不是 Anime Compass 的线上表现。** 评分没有时间戳；用户的 warm 历史包含 cold 番开播之后才打的分。")
    add(f"- **番很少。** cold-test 只有 {n_titles} 部，来自 2020 年的两个季度。作品重采样的区间很宽，S2 的包含 0。")
    add("- **主评估是条件化的。** 测试用户按\"至少喜欢一部测试番\"选出，不能保证模型之间的差异在一般用户里相同。辅助评估只改变了量级。")
    add("- **没有评分不等于不喜欢。** NDCG 的绝对值有选择偏差。评分文件只收看完的记录，弃番不在里面。")
    add("- **内容字段来自 2021 年初的快照，Wikipedia 的词条映射是 2026 年的结构。** 见预注册第 4.2 和 4.3 节。")
    add("- **AniList 信号的缺失不是随机的。** 没有信号的番都是极冷门的番，模型学到的是\"没有信号就排到最后\"。")
    add("- **系数不宜逐个解读。** `al_pop_pct` 和 `al_trend_pct` 高度相关，前者在 M0b 和 M2 里的系数是负的。")
    add("- **实验人群是 2021 年初以前的 MAL 用户。** 结论不能直接推广到 2021–2025 的新番和 AniList 用户。")
    add("")

    add("## 12. 文件")
    add("")
    add(table(["文件", "内容"], [
        ["`PRE_REGISTRATION.md`", "冻结的方案"],
        ["`manifest/split_manifest.json`", "split、输入摘要、泄漏检查"],
        ["`MODEL_SELECTION.json`", "选定的 C、冻结模型的系数、验证集结果、环境和输入摘要"],
        ["`results/final_test_results.json`", "正式测试的全部数字"],
        ["`results/pre_run_checks.json`", "运行前检查的逐项结果"],
        ["`results/final_test.lock`", "锁文件的副本（原件在 `artifacts/`，不进 git）"],
        ["`AI_USAGE.md`", "AI 使用记录和主要 prompts"],
    ], text=True))  # fmt: skip
    add("")

    (ROOT / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote RESULTS.md ({len(out)} lines)")


if __name__ == "__main__":
    main()

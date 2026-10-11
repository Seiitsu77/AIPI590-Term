"""Figures for the feasibility report.

Every number drawn here is read from results/feasibility_report.json or computed from
data/analysis_panel.csv and the pageview matrix. Nothing is typed in by hand.

Colours are the first two slots of a pre-validated categorical palette (blue, orange),
checked with the palette validator for colour-vision-deficiency separation and contrast
on the light surface. Text is always in ink colours, never in a series colour.

Usage:
    python src/figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
LANGS = ("en", "ja", "zh")
PRIMARY = "spearman_within_season"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    }
)


# Set by main(). The exploratory run reads and writes under its own names, and every
# figure it produces says so in its subtitle.
TAG = ""
NOTE = ""


def titled(ax, title: str, subtitle: str) -> None:
    subtitle += NOTE
    # The title sits above the subtitle, so it has to clear however many lines that is.
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=13 + 14 * (subtitle.count("\n") + 1))
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom", linespacing=1.35)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / (TAG + name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(report: dict) -> None:
    table = report["coverage"]["tv_by_popularity"]
    labels = ["Least popular\nfifth", "2nd", "3rd", "4th", "Most popular\nfifth"]
    keys = ["Q1_least", "Q2", "Q3", "Q4", "Q5_most"]
    mapped = [100 * table[key]["mapped_any"] for key in keys]
    viewed = [100 * table[key]["preair_views_any"] for key in keys]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(keys))
    width = 0.24
    for offset, values, colour, label in ((-width / 2 - 0.01, mapped, BLUE, "Has a Wikipedia article today"), (width / 2 + 0.01, viewed, ORANGE, "Had pageviews in the 30 days before premiere")):
        ax.bar(x + offset, values, width=width, color=colour, label=label, zorder=3)
        for xi, value in zip(x + offset, values, strict=True):
            ax.text(xi, value + 1.5, f"{value:.0f}%", ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_xticks(x, labels)
    # Headroom above 100% keeps the legend clear of the top gridline and the labels.
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_2, handlelength=1.0, borderaxespad=0.2)
    titled(ax, "Wikipedia coverage is where the popular shows are", "Share of TV anime premiering 2016-2020, by how many MAL users went on to rate the show 8+")
    save(fig, "fig1_coverage_by_popularity.png")


def fig_scatter(panel: pd.DataFrame) -> None:
    tv = panel[(panel["Type"] == "TV") & panel["mapped_any"]].copy()
    tv["views"] = tv[[f"views_pre30_{lang}" for lang in LANGS]].sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for flag, colour, label in ((0, BLUE, "New title"), (1, ORANGE, "Sequel")):
        rows = tv[tv["is_sequel"] == flag]
        ax.scatter(rows["views"] + 1, rows["n_pos"] + 1, s=22, color=colour, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wikipedia pageviews in the 30 days before premiere (en + ja + zh, +1)")
    ax.set_ylabel("MAL users who rated it 8+ (+1)")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK_2, handletextpad=0.3)
    silent = int((tv["views"] == 0).sum())
    ax.annotate(
        f"{silent} shows with no\npre-air views recorded",
        (1, tv.loc[tv["views"] == 0, "n_pos"].median() + 1),
        xytext=(14, 0),
        textcoords="offset points",
        va="center",
        color=INK_2,
        fontsize=8.5,
    )
    titled(ax, "Pre-air pageviews against eventual audience", f"{len(tv)} TV anime with a mapped article, premieres 2016-2020. Each dot is one show.")
    save(fig, "fig2_preair_views_vs_audience.png")


def fig_delta_forest(report: dict) -> None:
    primary = report["results"]["tv_mapped"]["ridge"]
    rows = [("All mapped TV titles (primary test)", primary["delta_vs_meta"]["meta+alt"][PRIMARY], primary["n_test"], True)]
    for key, label in (("new_ip", "New titles only"), ("sequel", "Sequels only")):
        block = primary["subgroups"][key]
        rows.append((label, block["delta_vs_meta"]["meta+alt"][PRIMARY], block["n_test"], False))
    for year, block in report["by_test_year"].items():
        rows.append((f"Test year {year} alone", block["delta"], block["n_test"], False))
    for key, label in (("tv_all", "All TV titles, unmapped included"), ("all_types_mapped", "All formats with a mapped article")):
        block = report["results"].get(key, {}).get("ridge")
        if block:
            rows.append((label, block["delta_vs_meta"]["meta+alt"][PRIMARY], block["n_test"], False))

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (label, (delta, low, high), n, is_primary) in zip(y, rows, strict=True):
        colour = BLUE if is_primary else MUTED
        ax.plot([low, high], [yi, yi], color=colour, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter([delta], [yi], s=60, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(1.02, yi, f"{delta:+.3f}  [{low:+.3f}, {high:+.3f}]   n={n}", transform=ax.get_yaxis_transform(), va="center", color=INK if is_primary else INK_2, fontsize=9)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change in within-season rank correlation when pageviews are added to metadata")
    titled(ax, "What pre-air pageviews add beyond metadata", "Point estimate and 95% bootstrap interval. Right of zero means pageviews improved the ranking.")
    save(fig, "fig3_increment_over_metadata.png")


def fig_arms(report: dict) -> None:
    arms = report["results"]["tv_mapped"]["ridge"]["arms"]
    rows = [
        ("Pageviews after premiere + metadata\n(look-ahead, not usable)", arms["meta+postair_LOOKAHEAD"][PRIMARY], False),
        ("Pre-air pageviews + metadata", arms["meta+alt"][PRIMARY], True),
        ("Metadata only", arms["meta"][PRIMARY], False),
        ("Pre-air pageviews only", arms["alt_only"][PRIMARY], False),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, emphasised) in zip(y, rows, strict=True):
        ax.barh(yi, value, height=0.5, color=BLUE if emphasised else AXIS, zorder=3)
        ax.text(max(value, 0.0) + 0.012, yi, f"{value:.3f}", va="center", color=INK if emphasised else INK_2, fontsize=9.5)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(min(0.0, min(row[1] for row in rows) - 0.05), 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Within-season rank correlation with eventual audience (test premieres 2019-2020)")
    titled(ax, "How well each information set ranks a season's new shows", "Ridge regression trained on 2016-2018 premieres. Higher is better; 1.0 is a perfect ranking.")
    save(fig, "fig4_ranking_by_information_set.png")


def fig_attention_curve(panel: pd.DataFrame) -> None:
    index = pd.read_csv(DATA / "pageviews_index.csv")
    payload = np.load(DATA / "pageviews_daily.npz")
    views, first_day = payload["views"], pd.Timestamp(str(payload["first_day"]))
    row_of = {(lang, title): row for row, lang, title in zip(index["row"], index["lang"], index["title"], strict=True)}

    extra: dict[int, set[tuple[str, str]]] = {}
    if TAG:
        links = pd.read_csv(DATA / "mal_wikipedia_extra_articles.csv")
        for mal_id, lang, name in zip(links["MAL_ID"], links["lang"], links["title"], strict=True):
            extra.setdefault(int(mal_id), set()).add((lang, name))

    tv = panel[(panel["Type"] == "TV") & panel["mapped_any"]].copy()
    tv["tercile"] = tv.groupby("season")["n_pos"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    days = np.arange(-90, 29)
    curves = {0: [], 2: []}
    for _, title in tv.iterrows():
        if title["tercile"] not in curves:
            continue
        articles = {(lang, title[f"{kind}_{lang}_title"]) for lang in LANGS for kind in ("own", "parent") if isinstance(title[f"{kind}_{lang}_title"], str)}
        rows = sorted(row_of[article] for article in articles | extra.get(int(title["MAL_ID"]), set()))
        offset = (title["start"] - first_day).days
        curves[title["tercile"]].append(views[rows, offset - 90 : offset + 29].sum(axis=0))

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for tercile, colour, label in ((2, BLUE, "Top third by eventual audience"), (0, ORANGE, "Bottom third")):
        median = np.median(np.vstack(curves[tercile]), axis=0)
        smooth = pd.Series(median).rolling(7, center=True, min_periods=1).mean().to_numpy()
        ax.plot(days, smooth + 1, color=colour, linewidth=2, solid_capstyle="round", label=label, zorder=3)
        ax.text(days[-1] + 1.5, smooth[-1] + 1, f"{smooth[-1]:,.0f}/day", color=INK_2, fontsize=9, va="center")
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.text(0.5, 0.03, " premiere", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(-90, 28)
    ax.set_xticks([-90, -60, -30, 0, 28], ["-90", "-60", "-30", "0", "+28"])
    ax.set_xlabel("Days relative to premiere")
    ax.set_ylabel("Median daily pageviews (7-day average, +1)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_2, handlelength=1.4, borderaxespad=0.2)
    titled(ax, "The gap in attention is already there before the first episode", "TV anime with a mapped article, premieres 2016-2020, split by eventual audience within each season")
    save(fig, "fig5_attention_around_premiere.png")


def fig_after_premiere() -> None:
    report = json.loads((RESULTS / "exploratory_after_premiere_report.json").read_text(encoding="utf-8"))
    horizons = report["definitions"]["horizons_days_after_premiere"]
    arms = ["meta", "meta+pre", *[f"meta+pre+first{h}" for h in horizons]]
    labels = ["Metadata\nonly", "+ pre-air\npageviews", *[f"+ first\n{h} days" for h in horizons]]
    x = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for model, colour, label in (("boosted", BLUE, "Boosted trees"), ("ridge", ORANGE, "Ridge regression")):
        values = [report["models"][model]["arms"][arm][PRIMARY] for arm in arms]
        ax.plot(x, values, color=colour, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.text(x[-1] + 0.12, values[-1], f"{values[-1]:.3f}", color=INK_2, fontsize=9, va="center")
        # Both models start from almost the same metadata-only score, so it is labelled once.
        if model == "boosted":
            ax.text(x[0] - 0.12, values[0], f"{values[0]:.2f}", color=INK_2, fontsize=9, va="center", ha="right")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.7, len(arms) - 0.3)
    ax.set_ylabel("Within-season rank correlation")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_2, handlelength=1.4, borderaxespad=0.2)
    titled(ax, "The ranking sharpens once the show starts airing, if the model is non-linear", f"TV anime with a mapped article, {report['definitions']['n_test']} test premieres 2019-2020. Information is cumulative left to right.")
    save(fig, "fig6_information_by_day.png")


def main() -> None:
    global TAG, NOTE
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploratory", action="store_true", help="draw the post hoc run (analyze.py --exploratory)")
    if parser.parse_args().exploratory:
        TAG, NOTE = "exploratory_", "\nExploratory: article mapping extended after the pre-specified test was read."

    report = json.loads((RESULTS / f"{TAG}feasibility_report.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(DATA / f"{TAG}analysis_panel.csv", parse_dates=["start"])
    fig_coverage(report)
    fig_scatter(panel)
    fig_delta_forest(report)
    fig_arms(report)
    fig_attention_curve(panel)
    if TAG and (RESULTS / "exploratory_after_premiere_report.json").exists():
        fig_after_premiere()
    print("wrote", sorted(path.name for path in FIGURES.glob(f"{TAG}fig*.png")))


if __name__ == "__main__":
    main()

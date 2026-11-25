"""README のターンテイキングの説明図（p_now / p_future の模式図）を作る。

使い方: python scripts/make_turn_taking_figure.py docs/img（Windows のメイリオを使う）
出力: turn-taking-{ja,en}-{light,dark}.svg

色と軸の向きは MaAI プロット（src/maai_sds/maai/plot_window.py）に合わせる。
- 上がユーザー側（入力波形1・黄）、下がシステム側（入力波形2・青）
- p_now / p_future は上が 0.0、下が 1.0。0.5 より上（ユーザー側）を黄、下（システム側）を青で塗る
- ターンのしきい値は緑の破線
曲線は説明用に作った模式的なもので、実測ではない。
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

for f in ("meiryo.ttc", "meiryob.ttc"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}")
plt.rcParams.update({
    "font.family": "Meiryo",
    "svg.fonttype": "path",
    "svg.hashsalt": "maai-sds",
    "axes.unicode_minus": False,
})

# MaAI プロットの色（plot_window.py）：ユーザー側 (245,189,0)、p_future の線 (245,120,0)、
# システム側 (80,160,240)、ターンしきい値 (80,200,120)、0.5 の線 (120,120,120)
USER, USER_PFUT, SYS, THR_C = "#f5bd00", "#f57800", "#50a0f0", "#50c878"
THEMES = {
    "light": dict(bg="#ffffff", text="#191919", sub="#666666", grid="#e6e6e6", lane="#f2f2f2", mid="#9a9a9a",
                  on_bar="#191919"),
    "dark": dict(bg="#0d1117", text="#efefef", sub="#9c9c9c", grid="#262c36", lane="#161b22", mid="#787878",
                 on_bar="#0d1117"),
}

TXT = {
    "ja": dict(
        user="ユーザー", system="システム", nopref="先読みなし",
        sys_pnow="システム\n（p_now）", sys_and="システム\n（p_now_and_future）",
        thr="ターンしきい値", user_side="ユーザー側", sys_side="システム側", xlabel="時間（秒）",
        a_title="A. 先読みで応答を早める",
        a_cap="① p_future がしきい値を超える → 先読み開始（LLM・TTS を動かし、音声は保持）\n"
              "② p_now がしきい値を超える → ターンを取り、保持した音声をすぐ再生",
        a_pref="先読み", a_speech="システムの発話", a_late="ここから生成を始めるので遅れる",
        b_title="B. 言いよどみでは先読みを捨てる",
        b_cap="① 話の途中の間で p_future がしきい値を超える → 先読み開始\n"
              "② ユーザーが話し始め、p_future がしきい値を下回る → 先読みを破棄"
              "（p_now は超えないのでターンは取らない）",
        b_pref="先読み（破棄）",
        c_title="C. ターンを取る条件（p_now / p_now_and_future）",
        c_cap="① p_now だけがしきい値を超える（p_future は未満）→ p_now：ターンを取る（ユーザーが続けると割り込みで停止）"
              "／p_now_and_future：取らない\n"
              "② p_now がしきい値を超え、p_future もしきい値以上 → どちらの条件でもターンを取る",
        c_cut="割り込みで停止",
        note="曲線は説明用の模式図で、実測ではありません。色と軸の向きは MaAI プロットと同じ（上がユーザー側、下がシステム側）",
    ),
    "en": dict(
        user="User", system="System", nopref="No prefetch",
        sys_pnow="System\n(p_now)", sys_and="System\n(p_now_and_future)",
        thr="turn threshold", user_side="user side", sys_side="system side", xlabel="Time (s)",
        a_title="A. Prefetching makes the response faster",
        a_cap="① p_future goes past the threshold → prefetch starts (LLM and TTS run, audio is held)\n"
              "② p_now goes past the threshold → take the turn and play the held audio at once",
        a_pref="prefetch", a_speech="system speech", a_late="generation starts here (late)",
        b_title="B. A hesitation discards the prefetch",
        b_cap="① a pause in the user's speech makes p_future go past the threshold → prefetch starts\n"
              "② the user speaks again and p_future falls back → the prefetch is discarded "
              "(p_now never goes past, so no turn)",
        b_pref="prefetch (discarded)",
        c_title="C. Turn condition (p_now / p_now_and_future)",
        c_cap="① only p_now goes past the threshold (p_future does not) → p_now: take the turn "
              "(stopped by barge-in when the user goes on) / p_now_and_future: do not\n"
              "② p_now goes past the threshold and p_future is past it too → both conditions take the turn",
        c_cut="barge-in",
        note="The curves are an illustration, not measurements. Colors and axis direction follow the MaAI plot "
             "(user side up, system side down)",
    ),
}

T = np.linspace(0.0, 4.0, 321)
THR = 0.55  # 見本の設定の「MaAIターンしきい値」


def sig(t, c, w):
    return 1.0 / (1.0 + np.exp(-(t - c) / w))


def bump(t, up, down, w):
    return sig(t, up, w) * (1.0 - sig(t, down, w))


def cross_up(y):
    i = np.where((y[:-1] < THR) & (y[1:] >= THR))[0]
    return [float(T[k]) for k in i]


def cross_down(y):
    i = np.where((y[:-1] >= THR) & (y[1:] < THR))[0]
    return [float(T[k]) for k in i]


def scenario_a():
    pfut = 0.12 + 0.72 * sig(T, 1.55, 0.13)
    pnow = 0.08 + 0.80 * sig(T, 2.25, 0.07)
    return pnow, pfut


def scenario_b():
    pfut = 0.12 + 0.58 * bump(T, 1.35, 2.0, 0.09)
    pnow = 0.08 + 0.27 * bump(T, 1.45, 1.85, 0.07)
    return pnow, pfut


def scenario_c():
    pnow = 0.08 + 0.64 * bump(T, 1.38, 1.72, 0.05) + 0.80 * sig(T, 2.85, 0.07)
    pnow = np.minimum(pnow, 0.9)
    pfut = 0.14 + 0.14 * bump(T, 1.2, 1.8, 0.1) + 0.70 * sig(T, 2.45, 0.12)
    return pnow, pfut


def draw(lang: str, theme: str, out: Path) -> None:
    c = THEMES[theme]
    s = TXT[lang]
    fig = plt.figure(figsize=(10.5, 12.0), facecolor=c["bg"])
    # 段の高さをそろえるため、行の高さを中身（発話の帯の数＋確率の 2 段）に比例させる
    outer = fig.add_gridspec(3, 1, height_ratios=[0.3 * 3 + 1.6, 0.3 * 2 + 1.6, 0.3 * 3 + 1.6], hspace=0.3,
                             left=0.155, right=0.885, top=0.955, bottom=0.065)

    def lanes(spec, rows):
        """発話の帯。rows は [(ラベル, [(開始, 終了, 種類, 文字), ...]), ...]"""
        ax = fig.add_subplot(spec)
        ax.set_facecolor(c["bg"])
        n = len(rows)
        for i, (label, bars) in enumerate(rows):
            y = n - 1 - i
            ax.add_patch(plt.Rectangle((0, y + 0.12), 4.0, 0.76, color=c["lane"], lw=0, zorder=0))
            ax.text(-0.08, y + 0.5, label, ha="right", va="center", fontsize=9.5, color=c["sub"],
                    linespacing=1.1)
            for (x0, x1, kind, txt) in bars:
                if kind == "user":
                    fc, ec, hatch = USER, "none", None
                elif kind == "sys":
                    fc, ec, hatch = SYS, "none", None
                elif kind == "pref":
                    fc, ec, hatch = "none", SYS, "////"
                else:  # ghost
                    fc, ec, hatch = "none", c["sub"], None
                ax.add_patch(FancyBboxPatch(
                    (x0, y + 0.2), x1 - x0, 0.6, boxstyle="round,pad=0,rounding_size=0.04",
                    fc=fc, ec=ec, lw=1.2, hatch=hatch, zorder=2, ls=(0, (3, 2)) if kind == "ghost" else "-"))
                if txt:
                    ax.text((x0 + x1) / 2, y + 0.5, txt, ha="center", va="center", fontsize=9,
                            color=c["on_bar"], zorder=3, fontweight="bold")
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, n)
        ax.axis("off")
        return ax

    def prob(spec, share, y, name, upper_pen, show_x):
        """MaAI プロットと同じ向き：上が 0.0（ユーザー側）、下が 1.0（システム側）。"""
        ax = fig.add_subplot(spec, sharex=share)
        ax.set_facecolor(c["bg"])
        ax.fill_between(T, y, 0.5, where=y <= 0.5, interpolate=True, color=USER, alpha=0.35, lw=0, zorder=2)
        ax.fill_between(T, y, 0.5, where=y >= 0.5, interpolate=True, color=SYS, alpha=0.35, lw=0, zorder=2)
        yu = np.where(y <= 0.5, y, np.nan)
        ys = np.where(y >= 0.5, y, np.nan)
        ax.plot(T, yu, color=upper_pen, lw=1.8, zorder=3)
        ax.plot(T, ys, color=SYS, lw=1.8, zorder=3)
        ax.axhline(0.5, color=c["mid"], lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(THR, color=THR_C, lw=1.3, ls=(0, (5, 3)), zorder=1)
        ax.set_ylim(1.0, 0.0)  # 上が 0.0、下が 1.0
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlim(0, 4.0)
        ax.tick_params(colors=c["sub"], labelsize=8.5, length=0)
        ax.tick_params(axis="x", labelbottom=show_x)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color=c["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.text(-0.08, 0.5, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                color=c["text"], fontweight="bold")
        ax.text(4.06, 0.12, "↑ " + s["user_side"], ha="left", va="center", fontsize=8.5, color=c["sub"])
        ax.text(4.06, 0.88, "↓ " + s["sys_side"], ha="left", va="center", fontsize=8.5, color=c["sub"])
        return ax

    def panel(row, title, lane_rows, pnow, pfut, marks):
        sub = outer[row].subgridspec(3, 1, height_ratios=[0.3 * len(lane_rows), 0.8, 0.8], hspace=0.22)
        la = lanes(sub[0], lane_rows)
        an = prob(sub[1], la, pnow, "p_now", USER, False)
        af = prob(sub[2], la, pfut, "p_future", USER_PFUT, True)
        la.set_title(title, loc="left", fontsize=12, fontweight="bold", color=c["text"], pad=16, x=-0.135)
        for x, num in marks:
            for ax in (la, an, af):
                ax.axvline(x, color=c["text"], lw=1, ls=(0, (1, 2)), zorder=5)
            la.text(x, len(lane_rows) + 0.05, num, ha="center", va="bottom", fontsize=11, color=c["text"],
                    fontweight="bold")
        return la, an, af

    # A
    pnow, pfut = scenario_a()
    a1, a2 = cross_up(pfut)[0], cross_up(pnow)[0]
    late = a2 + 0.7
    la, _, _ = panel(0, s["a_title"],
                     [(s["user"], [(0.15, 2.0, "user", "")]),
                      (s["system"], [(a1, a2, "pref", ""), (a2, 3.95, "sys", s["a_speech"])]),
                      (s["nopref"], [(late, 3.95, "ghost", "")])],
                     pnow, pfut, [(a1, "①"), (a2, "②")])
    la.text(a1 + (a2 - a1) / 2, 1.5, s["a_pref"], fontsize=8.5, color=c["text"], ha="center", va="center",
            zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=c["bg"], ec="none"))
    la.annotate("", xy=(late, 0.5), xytext=(a2, 0.5), arrowprops=dict(arrowstyle="->", color=c["sub"], lw=1))
    la.text(late + 0.05, 0.5, s["a_late"], fontsize=8.5, color=c["sub"], va="center")

    # B
    pnow, pfut = scenario_b()
    b1, b2 = cross_up(pfut)[0], cross_down(pfut)[0]
    la, _, _ = panel(1, s["b_title"],
                     [(s["user"], [(0.15, 1.15, "user", ""), (1.95, 3.6, "user", "")]),
                      (s["system"], [(b1, b2, "pref", "")])],
                     pnow, pfut, [(b1, "①"), (b2, "②")])
    la.text(b1 + (b2 - b1) / 2, 0.5, s["b_pref"], fontsize=8.5, color=c["text"], ha="center", va="center",
            zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=c["bg"], ec="none"))

    # C
    pnow, pfut = scenario_c()
    c1 = cross_up(pnow)[0]
    c_pref = cross_up(pfut)[0]
    c2 = cross_up(pnow)[1]
    user_back = 1.85
    la, _, _ = panel(2, s["c_title"],
                     [(s["user"], [(0.15, 1.3, "user", ""), (user_back, 2.55, "user", "")]),
                      (s["sys_pnow"], [(c1, user_back + 0.06, "sys", ""), (c_pref, c2, "pref", ""),
                                       (c2, 3.95, "sys", "")]),
                      (s["sys_and"], [(c_pref, c2, "pref", ""), (c2, 3.95, "sys", "")])],
                     pnow, pfut, [(c1, "①"), (c2, "②")])
    la.text(user_back + 0.12, 1.5, s["c_cut"], fontsize=8.5, color=c["sub"], va="center")

    fig.axes[-1].set_xlabel(s["xlabel"], fontsize=9.5, color=c["sub"], labelpad=2)
    # 凡例（しきい値の線）
    fig.text(0.885, 0.99, f"— — {s['thr']} ({THR})", ha="right", va="top", fontsize=8.5, color=THR_C)
    fig.text(0.885, 0.006, s["note"], ha="right", va="bottom", fontsize=8, color=c["sub"])
    fig.savefig(out, facecolor=c["bg"], metadata={"Date": None})
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / (out.stem + ".png"), facecolor=c["bg"], dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    d.mkdir(parents=True, exist_ok=True)
    for lang in ("ja", "en"):
        for theme in ("light", "dark"):
            p = d / f"turn-taking-{lang}-{theme}.svg"
            draw(lang, theme, p)
            print(p, p.stat().st_size)

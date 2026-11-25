"""README のターンテイキングの説明図（p_now / p_future の模式図）を作る。

使い方: python scripts/make_turn_taking_figure.py docs/img（Windows のメイリオを使う）
出力: turn-taking-{ja,en}-{light,dark}.svg
曲線は説明用に作った模式的なもので、実測ではない。
"""
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

THEMES = {
    "light": dict(bg="#ffffff", text="#191919", sub="#666666", grid="#e6e6e6", lane="#f2f2f2",
                  pnow="#257de0", pfut="#f47230", sys="#00b894", user="#8c8c8c", thr="#666666"),
    "dark": dict(bg="#0d1117", text="#efefef", sub="#9c9c9c", grid="#262c36", lane="#161b22",
                 pnow="#338ced", pfut="#e76624", sys="#00a888", user="#7d8590", thr="#9c9c9c"),
}

TXT = {
    "ja": dict(
        user="ユーザー", system="システム", nopref="先読みなし",
        sys_pnow="システム\n（p_now）", sys_and="システム\n（p_now_and_future）",
        thr="しきい値", xlabel="時間（秒）",
        a_title="A. 先読みで応答を早める",
        a1="① p_future が上抜け\n先読み開始（LLM・TTS を動かし、音声は保持）",
        a2="② p_now が上抜け\nターンを取り、保持した音声をすぐ再生",
        a_pref="先読み", a_speech="システムの発話", a_late="ここから生成を始めるので遅れる",
        b_title="B. 言いよどみでは先読みを捨てる",
        b1="① p_future が上抜け\n先読み開始",
        b2="② ユーザーが話し始め、p_future が下抜け\n先読みを破棄（p_now は上抜けないのでターンは取らない）",
        b_pref="先読み（破棄）",
        c_title="C. ターンを取る条件（p_now / p_now_and_future）",
        c1="① p_now だけが上抜け\n（p_future はしきい値未満）\np_now：ターンを取る\np_now_and_future：取らない",
        c2="② p_now が上抜け\n（p_future もしきい値以上）\nどちらの条件でもターンを取る",
        c_cut="割り込みで停止", c_pref="先読み",
        note="曲線は説明用の模式図で、実測ではありません",
    ),
    "en": dict(
        user="User", system="System", nopref="No prefetch",
        sys_pnow="System\n(p_now)", sys_and="System\n(p_now_and_future)",
        thr="threshold", xlabel="Time (s)",
        a_title="A. Prefetching makes the response faster",
        a1="① p_future crosses up\nprefetch starts (LLM and TTS run, audio is held)",
        a2="② p_now crosses up\ntake the turn and play the held audio at once",
        a_pref="prefetch", a_speech="system speech", a_late="generation starts here (late)",
        b_title="B. A hesitation discards the prefetch",
        b1="① p_future crosses up\nprefetch starts",
        b2="② the user speaks again and p_future crosses down\nthe prefetch is discarded (p_now never crosses, so no turn)",
        b_pref="prefetch (discarded)",
        c_title="C. Turn condition (p_now / p_now_and_future)",
        c1="① only p_now crosses up\n(p_future is below the threshold)\np_now: take the turn\np_now_and_future: do not",
        c2="② p_now crosses up\n(p_future is above the threshold)\nboth conditions take the turn",
        c_cut="barge-in", c_pref="prefetch",
        note="The curves are an illustration, not measurements",
    ),
}

T = np.linspace(0.0, 4.0, 801)
THR = 0.5


def sig(t, c, w):
    return 1.0 / (1.0 + np.exp(-(t - c) / w))


def bump(t, up, down, w):
    return sig(t, up, w) * (1.0 - sig(t, down, w))


def cross_up(y, t=T):
    i = np.where((y[:-1] < THR) & (y[1:] >= THR))[0]
    return [float(t[k]) for k in i]


def cross_down(y, t=T):
    i = np.where((y[:-1] >= THR) & (y[1:] < THR))[0]
    return [float(t[k]) for k in i]


def scenario_a():
    pfut = 0.12 + 0.72 * sig(T, 1.55, 0.13)
    pnow = 0.08 + 0.80 * sig(T, 2.25, 0.07)
    return pnow, pfut


def scenario_b():
    pfut = 0.12 + 0.55 * bump(T, 1.35, 2.0, 0.09)
    pnow = 0.08 + 0.27 * bump(T, 1.45, 1.85, 0.07)
    return pnow, pfut


def scenario_c():
    pnow = 0.08 + 0.55 * bump(T, 1.38, 1.72, 0.05) + 0.80 * sig(T, 2.85, 0.07)
    pnow = np.minimum(pnow, 0.9)
    pfut = 0.14 + 0.14 * bump(T, 1.2, 1.8, 0.1) + 0.70 * sig(T, 2.45, 0.12)
    return pnow, pfut


def draw(lang: str, theme: str, out: Path) -> None:
    c = THEMES[theme]
    s = TXT[lang]
    fig = plt.figure(figsize=(10.5, 11.2), facecolor=c["bg"])
    outer = fig.add_gridspec(3, 1, hspace=0.42, left=0.175, right=0.97, top=0.965, bottom=0.07)

    def lanes(spec, rows):
        """上段：発話の帯。rows は [(ラベル, [(開始, 終了, 種類, 文字), ...]), ...]"""
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
                    fc, ec, hatch, tc = c["user"], "none", None, c["bg"]
                elif kind == "sys":
                    fc, ec, hatch, tc = c["sys"], "none", None, c["bg"]
                elif kind == "pref":
                    fc, ec, hatch, tc = "none", c["sys"], "////", c["text"]
                else:  # ghost
                    fc, ec, hatch, tc = "none", c["sub"], None, c["sub"]
                box = FancyBboxPatch((x0, y + 0.2), x1 - x0, 0.6, boxstyle="round,pad=0,rounding_size=0.04",
                                     fc=fc, ec=ec, lw=1.2, hatch=hatch, zorder=2,
                                     ls=(0, (3, 2)) if kind == "ghost" else "-")
                ax.add_patch(box)
                if txt:
                    inside = kind in ("user", "sys")
                    ax.text((x0 + x1) / 2 if inside else x1 + 0.05, y + 0.5, txt,
                            ha="center" if inside else "left", va="center", fontsize=9, color=tc,
                            zorder=3, fontweight="bold" if inside else "normal")
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, n)
        ax.axis("off")
        return ax

    def probs(spec, pnow, pfut, share, label_at):
        ax = fig.add_subplot(spec, sharex=share)
        ax.set_facecolor(c["bg"])
        ax.axhline(THR, color=c["thr"], lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(4.0, THR + 0.03, s["thr"], ha="right", va="bottom", fontsize=9, color=c["sub"])
        ax.plot(T, pfut, color=c["pfut"], lw=2, zorder=3, solid_capstyle="round")
        ax.plot(T, pnow, color=c["pnow"], lw=2, zorder=3, solid_capstyle="round")
        (xn, yn), (xf, yf) = label_at
        ax.text(xn, yn, "p_now", color=c["text"], fontsize=10, ha="left", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc=c["bg"], ec=c["pnow"], lw=1.2))
        ax.text(xf, yf, "p_future", color=c["text"], fontsize=10, ha="left", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc=c["bg"], ec=c["pfut"], lw=1.2))
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(0, 4.0)
        ax.tick_params(colors=c["sub"], labelsize=9, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color=c["grid"], lw=0.8)
        ax.set_axisbelow(True)
        return ax

    def marker(axes, x, num_y, text, side="right"):
        for ax in axes:
            ax.axvline(x, color=c["text"], lw=1, ls=(0, (1, 2)), zorder=5)
        ax = axes[-1]
        ax.annotate(text, xy=(x, num_y), xytext=(x + (0.06 if side == "right" else -0.06), num_y),
                    ha="left" if side == "right" else "right", va="center", fontsize=9, color=c["text"],
                    linespacing=1.3, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=c["bg"], ec="none", alpha=0.92))

    def panel(row, title, lane_rows, pnow, pfut, label_at, marks):
        sub = outer[row].subgridspec(2, 1, height_ratios=[0.34 * len(lane_rows), 1.0], hspace=0.08)
        la = lanes(sub[0], lane_rows)
        pa = probs(sub[1], pnow, pfut, la, label_at)
        la.set_title(title, loc="left", fontsize=12, fontweight="bold", color=c["text"], pad=6, x=-0.16)
        for x, y, text, side in marks:
            marker([la, pa], x, y, text, side)
        return pa

    # A
    pnow, pfut = scenario_a()
    a1, a2 = cross_up(pfut)[0], cross_up(pnow)[0]
    late = a2 + 0.7
    panel(0, s["a_title"],
          [(s["user"], [(0.15, 2.0, "user", "")]),
           (s["system"], [(a1, a2, "pref", ""), (a2, 3.95, "sys", s["a_speech"])]),
           (s["nopref"], [(late, 3.95, "ghost", "")])],
          pnow, pfut, ((3.3, 0.99), (3.3, 0.71)),
          [(a1, 0.62, s["a1"], "left"), (a2, 0.22, s["a2"], "right")])
    fig.axes[0].text(a1 + (a2 - a1) / 2, 1.5, s["a_pref"], fontsize=8.5, color=c["text"], ha="center",
                     va="center", zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=c["bg"], ec="none"))
    # 先読みなしの帯に「遅れる」を書く
    fig.axes[0].annotate("", xy=(late, 0.5), xytext=(a2, 0.5),
                         arrowprops=dict(arrowstyle="->", color=c["sub"], lw=1))
    fig.axes[0].text(late + 0.05, 0.5, s["a_late"], fontsize=8.5, color=c["sub"], va="center")

    # B
    pnow, pfut = scenario_b()
    b1, b2 = cross_up(pfut)[0], cross_down(pfut)[0]
    panel(1, s["b_title"],
          [(s["user"], [(0.15, 1.15, "user", ""), (1.95, 3.6, "user", "")]),
           (s["system"], [(b1, b2, "pref", "")])],
          pnow, pfut, ((0.95, 0.3), (0.35, 0.3)),
          [(b1, 0.82, s["b1"], "left"), (b2, 0.82, s["b2"], "right")])
    fig.axes[2].text(b1 + (b2 - b1) / 2, 0.5, s["b_pref"], fontsize=8.5, color=c["text"], ha="center",
                     va="center", zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=c["bg"], ec="none"))

    # C
    pnow, pfut = scenario_c()
    c1 = cross_up(pnow)[0]
    c_pref = cross_up(pfut)[0]
    c2 = cross_up(pnow)[1]
    user_back = 1.85
    panel(2, s["c_title"],
          [(s["user"], [(0.15, 1.3, "user", ""), (user_back, 2.55, "user", "")]),
           (s["sys_pnow"], [(c1, user_back + 0.06, "sys", ""), (c_pref, c2, "pref", ""),
                            (c2, 3.95, "sys", "")]),
           (s["sys_and"], [(c_pref, c2, "pref", ""), (c2, 3.95, "sys", "")])],
          pnow, pfut, ((3.3, 0.99), (3.3, 0.71)),
          [(c1, 0.80, s["c1"], "left"), (c2, 0.27, s["c2"], "right")])
    fig.axes[4].text(user_back + 0.12, 1.5, s["c_cut"], fontsize=8.5, color=c["sub"], va="center")

    for ax in fig.axes:
        if ax.get_subplotspec() is not None and ax.axison:
            pass
    fig.axes[-1].set_xlabel(s["xlabel"], fontsize=9.5, color=c["sub"])
    fig.text(0.97, 0.012, s["note"], ha="right", va="bottom", fontsize=8.5, color=c["sub"])
    fig.savefig(out, facecolor=c["bg"], metadata={"Date": None})
    import os
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

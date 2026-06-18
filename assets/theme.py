"""
Tema visual para matplotlib.
Importar no início de qualquer notebook:

    from assets.theme import apply, COLORS, squircle_ax
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paleta ──────────────────────────────────────────────────────────────
COLORS = {
    # Primárias
    "dark_blue":   "#1D4F91",
    "light_blue":  "#426DA9",
    "purple":      "#77127B",
    "raspberry":   "#C1188B",
    "magenta":     "#E80070",
    "white":       "#FFFFFF",
    # Status e dados
    "navy":        "#1A2B4B",
    "alert_green": "#0FAC67",
    "alert_red":   "#FA1320",
    "yellow":      "#FEDB00",
    "orange":      "#FF9735",
    "teal":        "#54C2B8",
    "lime":        "#AFB904",
    "sapphire":    "#3087AA",
    "grey":        "#495765",
    # Secundárias
    "blue_10":     "#F7F8FB",
    "blue_20":     "#E1E4EF",
    "blue_30":     "#C6CBE1",
    "grey_10":     "#F8F8F9",
    "grey_50":     "#8F97A0",
}

SCORE_PALETTE  = [COLORS["alert_red"], COLORS["yellow"], COLORS["alert_green"]]
SERIES_PALETTE = [
    COLORS["dark_blue"], COLORS["teal"],    COLORS["yellow"],
    COLORS["raspberry"], COLORS["sapphire"], COLORS["lime"],
]
SEQUENTIAL = [COLORS["blue_10"], COLORS["blue_20"], COLORS["blue_30"],
              COLORS["light_blue"], COLORS["dark_blue"], COLORS["navy"]]


# ── Tema matplotlib ──────────────────────────────────────────────────────
def apply():
    """Aplica o tema globalmente em todos os plots da sessão."""
    plt.rcParams.update({
        # Figura
        "figure.facecolor":     COLORS["white"],
        "figure.dpi":           150,
        "figure.figsize":       (10, 5),

        # Axes
        "axes.facecolor":       COLORS["blue_10"],
        "axes.edgecolor":       COLORS["blue_20"],
        "axes.labelcolor":      COLORS["navy"],
        "axes.titlecolor":      COLORS["navy"],
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.prop_cycle":      plt.cycler(color=SERIES_PALETTE),

        # Grid
        "grid.color":           COLORS["blue_20"],
        "grid.linewidth":       0.8,
        "grid.alpha":           0.6,

        # Ticks
        "xtick.color":          COLORS["grey"],
        "ytick.color":          COLORS["grey"],
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,

        # Legenda
        "legend.framealpha":    0.95,
        "legend.edgecolor":     COLORS["blue_20"],
        "legend.fontsize":      10,

        # Linhas
        "lines.linewidth":      2.2,
        "lines.markersize":     6,

        # Fontes — Roboto para materiais online, Arial como fallback
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Roboto", "Arial", "DejaVu Sans"],
    })


def squircle_ax(ax, radius: float = 0.04):
    """Arredonda as bordas do axes para efeito squircle (usa FancyBboxPatch)."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(COLORS["blue_10"])
    fancy = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=COLORS["blue_10"],
        edgecolor=COLORS["blue_20"],
        linewidth=1.2,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(fancy)


def loss_plot(train_losses: list[float], val_losses: list[float],
              eval_interval: int = 500, title: str = "Curvas de Loss") -> plt.Figure:
    """Plot padrão de loss de treino vs validação."""
    apply()
    fig, ax = plt.subplots()

    steps_train = list(range(1, len(train_losses) + 1))
    steps_val   = [i * eval_interval for i in range(1, len(val_losses) + 1)]

    ax.plot(steps_train, train_losses, color=COLORS["dark_blue"], label="Train loss", linewidth=2.2)
    ax.plot(steps_val,   val_losses,   color=COLORS["yellow"],    label="Val loss",   linewidth=2.2, linestyle="--")

    best_step = steps_val[int(np.argmin(val_losses))]
    ax.axvline(best_step, color=COLORS["alert_green"], linestyle=":", linewidth=1.5,
               label=f"Melhor val (step {best_step})")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend()
    squircle_ax(ax)
    fig.tight_layout()
    return fig


def score_bar(labels: list[str], values: list[float],
              title: str = "Distribuição de Score") -> plt.Figure:
    """Barras com paleta de score (vermelho → amarelo → verde)."""
    apply()
    fig, ax = plt.subplots()

    colors = [SCORE_PALETTE[int(v / (max(values) + 1e-9) * 2.99)] for v in values]

    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                color=COLORS["navy"], fontweight="bold")

    ax.set_title(title)
    squircle_ax(ax)
    fig.tight_layout()
    return fig

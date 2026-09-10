"""Create a grouped success-rate plot for up to ten named sub-policies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Each sub-policy needs one success rate per object, in the same order.
OBJECTS = ["Chair1", "Chair2", "Cart", "Bucket"]
SUB_POLICY_RESULTS = {
    # First three: warm-color group => PPO-i
    # "Sub-policy Alpha": [69.20, 61.47, 56.23, 85.38],
    # "Sub-policy Beta": [70.83, 65.94, 74.34, 84.39],
    # "Sub-policy Gamma": [70.48, 62.70, 69.41, 0.00],
    
    # Best $PPO_i$
    "Best $PPO_i$": [0.7083, 0.6594, 0.7434, 0.8538],
    # Last two: cool colors, with spacing between them => PPO random & Grasp-TAGS.
    "$PPO_g +$ random": [0.7083, 0.5344, 0.5757, 0.7715],
    "$\mathbf{Grasp-TAGS ({Ours})}$": [0.8699, 0.7937, 0.8354, 0.8994],
}

PLOT_TITLE = "Policy Success Rates by Object"
OUTPUT_FILE = Path(__file__).with_name("success_rate_bar_plot.png")
MAX_POLICIES = 10
WARM_GROUP_SIZE = 3

# This same gap is used after column 3 and after column 4.
GROUP_GAP_IN_BAR_WIDTHS = 0.45

# BAR_COLORS = [
#     "#D73027",  # warm red
#     "#F46D43",  # warm orange-red
#     "#FDAE61",  # warm orange
#     "#4575B4",  # cool blue
#     "#74ADD1",  # cool light blue
# ]

BAR_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
]

def validate_data() -> None:
    if not OBJECTS:
        raise ValueError("Add at least one object to OBJECTS.")
    if not SUB_POLICY_RESULTS:
        raise ValueError("Add at least one sub-policy to SUB_POLICY_RESULTS.")
    if len(SUB_POLICY_RESULTS) > MAX_POLICIES:
        raise ValueError(f"A maximum of {MAX_POLICIES} sub-policies is supported.")

    for name, rates in SUB_POLICY_RESULTS.items():
        if len(rates) != len(OBJECTS):
            raise ValueError(
                f"{name!r} has {len(rates)} rates, but there are "
                f"{len(OBJECTS)} objects."
            )
        if any(not 0 <= rate <= 1 for rate in rates):
            raise ValueError(f"All success rates for {name!r} must be in [0, 1].")


def create_plot() -> None:
    validate_data()

    x = np.arange(len(OBJECTS))
    names = list(SUB_POLICY_RESULTS)
    count = len(names)

    # Add one gap after bar 3 and another equal gap after bar 4 when present.
    first_gap_exists = count > WARM_GROUP_SIZE
    second_gap_exists = count > WARM_GROUP_SIZE + 1
    gap_count = int(first_gap_exists) + int(second_gap_exists)
    group_width = 0.82
    bar_width = group_width / (count + gap_count * GROUP_GAP_IN_BAR_WIDTHS)

    if count <= len(BAR_COLORS):
        colors = BAR_COLORS[:count]
    else:
        colors = plt.colormaps["tab10"].colors[:count]

    raw_offsets = []
    for index in range(count):
        offset_units = float(index)
        if first_gap_exists and index >= WARM_GROUP_SIZE:
            offset_units += GROUP_GAP_IN_BAR_WIDTHS
        if second_gap_exists and index >= WARM_GROUP_SIZE + 1:
            offset_units += GROUP_GAP_IN_BAR_WIDTHS
        raw_offsets.append(offset_units * bar_width)
    offsets = np.asarray(raw_offsets) - np.mean(raw_offsets)

    fig, ax = plt.subplots(figsize=(max(10, len(OBJECTS) * 2.1), 6))
    for index, (name, color) in enumerate(zip(names, colors)):
        bars = ax.bar(
            x + offsets[index],
            SUB_POLICY_RESULTS[name],
            width=bar_width * 0.94,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=2,
            fontsize=16 if count <= 5 else 12,
            rotation=0,
            fontweight="bold" if name == "$\mathbf{Grasp-TAGS ({Ours})}$" else "normal",
        )

    ax.set_title(PLOT_TITLE, fontsize=18, pad=12)
    ax.set_xlabel("Object", fontsize=15)
    ax.set_ylabel("Success Rate", fontsize=15)
    ax.set_xticks(x, OBJECTS)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=min(count, 5),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        fontsize=13,
        title_fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    create_plot()

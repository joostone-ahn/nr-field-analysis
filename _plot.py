
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import _common

def plot_kpi(df, grid_size, out_dir, title):
    os.makedirs(out_dir, exist_ok = True)
    plot_dir = os.path.join(out_dir, f"plot_{grid_size}m")
    os.makedirs(plot_dir, exist_ok=True)
    
    df_pair = _common.grid_kpi(df, grid_size=grid_size)

    rsrp_col = "RSRP_n28"
    rsrp_min = int(df_pair[rsrp_col].min())
    rsrp_max = int(df_pair[rsrp_col].max())
    bins = np.arange(rsrp_min, rsrp_max + 2, 1)  # 1 dB step
    df_pair["RSRP_bin"] = pd.cut(df_pair["RSRP_n28"], bins=bins, right=False)
    
    df_binned = (
        df_pair.groupby("RSRP_bin", observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    df_binned["RSRP_bin_tick"] = df_binned["RSRP_bin"].apply(lambda x: int(x.left))
    
    style_map = {
        "n26": {"marker": "o", "color": "blue"},
        "n28": {"marker": "s", "color": "red"},
    }
    
    # 공통 x축 설정 함수
    def apply_common_axis(ax, df, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.7)
    
        # min/max를 5dB 단위로 맞추기
        x_min = int(df["RSRP_bin_tick"].min())
        x_max = int(df["RSRP_bin_tick"].max())
        x_min_5 = (x_min // 5) * 5
        x_max_5 = (x_max // 5) * 5
    
        xticks = np.arange(x_min_5, x_max_5 + 1, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45)
    
    
    # Rx Quality (Absolute + Difference)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.32)
    
    # Absoulte
    metrics_group1 = [("RSRP", "RSRP [dBm]"),
                      ("RSRQ", "RSRQ [dB]"),
                      ("SINR", "SINR [dB]")]
    
    for ax, (metric, ylabel) in zip(axes[0], metrics_group1):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
                style = style_map[source]
                ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                        marker=style["marker"], markersize=3,
                        color=style["color"], label=source,
                        linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad = 8)
    
    # n28 - n26 Difference
    metrics_group_diff = [
        ("RSRP", "RSRP Δ [dB]"),
        ("RSRQ", "RSRQ Δ [dB]"),
        ("SINR", "SINR Δ [dB]")
    ]
    
    for ax, (metric, ylabel) in zip(axes[1], metrics_group_diff):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax.plot(df_binned["RSRP_bin_tick"], diff,
                    marker="^", markersize=3,
                    color="green", label=f"{metric} (n28-n26)",
                    linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.set_ylim(-5, 5)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad = 8)
    
    fig.suptitle(f"{title} Rx Quality (RSRP, RSRQ, SINR)", fontsize=14, y = 0.95)
    out_path = os.path.join(plot_dir, f"plot_rx_quality.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")
    
    
    # Link Adaptation (CQI, MCS, RI)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.32)

    metrics_abs = [
        ("RI", "RI"),
        ("CQI", "CQI"),
        ("DL_MCS", "DL MCS"),
    ]
    metrics_diff = [
        ("RI", "RI Δ"),
        ("CQI", "CQI Δ"),
        ("DL_MCS", "DL MCS Δ"),
    ]
    
    # Absolute
    for ax, (metric, ylabel) in zip(axes[0], metrics_abs):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
                style = style_map[source]
                ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                        marker=style["marker"], markersize=3,
                        color=style["color"], label=source,
                        linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad = 8)
    
    # delta Δ
    for ax, (metric, ylabel) in zip(axes[1], metrics_diff):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax.plot(df_binned["RSRP_bin_tick"], diff,
                    marker="^", markersize=3,
                    color="green", label=f"{metric} (n28−n26)",
                    linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
    
            if metric == "CQI":
                ax.set_ylim(-10, 10)
            elif metric == "DL_MCS":
                ax.set_ylim(-20, 20)
            elif metric == "RI":
                ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad = 8)
    
    fig.suptitle(f"{title} Link Adaptation (RI, CQI, DL MCS)", fontsize=14, y=0.95)
    out_path = os.path.join(plot_dir, f"plot_link_adaptation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")
    
    # Throughput
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.25)

    metric = "DL_Tput"
    col_n26 = f"{metric}_n26"
    col_n28 = f"{metric}_n28"
    
    # Absolute Throughput
    ax = axes[0]
    for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
        style = style_map[source]
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                marker=style["marker"], markersize=3,
                color=style["color"], label=source,
                linestyle="-", linewidth=1)
    apply_common_axis(ax, df_binned, "RSRP [dBm]", "DL Tput [Mbps]")
    ax.legend()
    # ax.set_title("Absolute DL Tput", fontsize=10, pad=7)
    
    # Relative Throughput
    ax = axes[1]
    if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
        rel_ratio = (df_binned[col_n28] / df_binned[col_n26]) * 100
        ax.plot(df_binned["RSRP_bin_tick"], rel_ratio,
                marker="s", markersize=3, color="green", label="n28/n26 [%]",
                linestyle="-", linewidth=1)
    apply_common_axis(ax, df_binned, "RSRP [dBm]", "DL Tput Ratio [%]")
    ax.legend()
    # ax.set_title("Relative DL Tput", fontsize=10, pad=7)
    
    fig.suptitle(f"{title} DL Throughput", fontsize=14, y=0.98)
    out_path = os.path.join(plot_dir, f"plot_{metric}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_tput_vs_sinr(df, grid_size, out_dir, title):

    df_pair = _common.grid_kpi(df, grid_size=grid_size)

    rsrp_col = "RSRP_n28"
    bins = np.arange(int(df_pair[rsrp_col].min()), int(df_pair[rsrp_col].max()) + 1, 1)
    df_pair["RSRP_bin"] = pd.cut(df_pair["RSRP_n28"], bins=bins, right=False)
    df_binned = df_pair.groupby("RSRP_bin", observed=True).mean(numeric_only=True).reset_index()
    df_binned["RSRP_bin_tick"] = df_binned["RSRP_bin"].apply(lambda x: int(x.left))

    style_map = {"n26": {"color": "blue", "label": "n26"},
                 "n28": {"color": "red", "label": "n28"},
                 "diff": {"color": "green", "label": "Δ / Ratio"}}

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", "DL Tput Ratio (n28/n26) [%]", "ratio"),
        ("SINR", "SINR [dB]", "SINR Δ [dB]", "diff"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3)

    for ax, (metric, ylabel, ylabel_2, mode) in zip(axes, metrics):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 not in df_binned.columns or col_n28 not in df_binned.columns:
            continue

        # --- 왼쪽 y축: n26 / n28 ---
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col_n26], marker='o', markersize=3,
                color=style_map["n26"]["color"], label=style_map["n26"]["label"], linewidth=1)
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col_n28], marker='o', markersize=3,
                color=style_map["n28"]["color"], label=style_map["n28"]["label"], linewidth=1)
        ax.set_xlabel("RSRP [dBm]", fontsize=11)
        ax.set_ylabel(f"{ylabel}", fontsize=11)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        ax.invert_xaxis()

        # --- 오른쪽 y축: Δ or Ratio ---
        ax2 = ax.twinx()
        if mode == "diff":
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax2.plot(df_binned["RSRP_bin_tick"], diff,
                     color=style_map["diff"]["color"], linestyle="--", linewidth=0.8,
                     label=f"{metric} Δ (n28−n26)")
            absmax = np.nanmax(np.abs(diff))
            absmax = np.ceil(absmax)
            ax2.set_ylim(-absmax, absmax)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        else:
            ratio = (df_binned[col_n28] / df_binned[col_n26]) * 100
            diff_from_100 = ratio - 100
            absmax = np.nanmax(np.abs(diff_from_100))
            absmax = np.ceil(absmax)
            ax2.set_ylim(- absmax, absmax)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax2.plot(df_binned["RSRP_bin_tick"], diff_from_100,
                     color=style_map["diff"]["color"], linestyle="--", linewidth=0.8,
                     label=f"{metric} Ratio [%]")

        # --- 오른쪽 축 스타일 ---
        ax2.set_ylabel(ylabel_2, color=style_map["diff"]["color"], fontsize=11)
        ax2.tick_params(axis='y', labelcolor=style_map["diff"]["color"])

        # --- 범례 병합 ---
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

        # ax.set_title(ylabel, fontsize=12, pad=8)

    fig.suptitle(f"{title} KPI Trend : DL Tput vs SINR", fontsize=14, y=0.93)
    out_path = os.path.join(out_dir, "plot_DL_Tput_vs_SINR.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os
import _common


def split_band_df(df_pair):
    common_cols = [c for c in df_pair.columns if not any(s in c for s in ["_n26", "_n28", "_diff"])]
    n26_cols = [c for c in df_pair.columns if c.endswith("_mean_n26")]
    n28_cols = [c for c in df_pair.columns if c.endswith("_mean_n28")]

    df_n26 = df_pair[common_cols + n26_cols].copy()
    df_n28 = df_pair[common_cols + n28_cols].copy()

    df_n26.columns = [c.replace("_mean_n26", "") for c in df_n26.columns]
    df_n28.columns = [c.replace("_mean_n28", "") for c in df_n28.columns]

    df_n26["Band"] = "n26"
    df_n28["Band"] = "n28"

    return df_n26, df_n28

def scat_kpi_by_UHD(df, out_dir, grid_size=25, rb_min=48, sample_min=10, band="n28"):

    metrics = [
        "SINR_TRS",
        "DL_Tput",
    ]

    for metric in metrics:
        df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
        df_n26, df_n28 = split_band_df(df_pair)
        plot_df = df_n26.copy() if band == "n26" else df_n28.copy()

        plot_df = plot_df[(plot_df["RSRP"] <= -60) & (plot_df["RSRP"] >= -120)]

        color_col = "uhd_max"
        valid_vals = plot_df[color_col].dropna()

        if len(valid_vals) > 0:
            q1, q2, q3 = valid_vals.quantile([0.25, 0.5, 0.75])

            def color_by_uhd(v):
                if pd.isna(v):
                    return "gray", "null"
                elif v >= q3:
                    return "#FF4500", f"PWR ≥ {q3:.0f}"  # 빨강 (높음)
                elif v >= q2:
                    return "#FFD700", f"{q2:.0f} ≤ PWR < {q3:.0f}"  # 노랑
                elif v >= q1:
                    return "#32CD32", f"{q1:.0f} ≤ PWR < {q2:.0f}"  # 초록
                else:
                    return "#1E90FF", f"PWR < {q1:.0f}"  # 파랑 (낮음)

            plot_df[["color", "color_label"]] = plot_df[color_col].apply(lambda v: pd.Series(color_by_uhd(v)))
        else:
            plot_df["color"] = "gray"
            plot_df["color_label"] = "null"

        order = [
            f"PWR ≥ {q3:.0f}",
            f"{q2:.0f} ≤ PWR < {q3:.0f}",
            f"{q1:.0f} ≤ PWR < {q2:.0f}",
            f"PWR < {q1:.0f}",
            "null",
        ]
        plot_df["color_label"] = pd.Categorical(plot_df["color_label"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("color_label")

        def make_hover_text(row):
            lines = [
                # "───────────────",
                f"loc_id_({grid_size}m): {row['loc_id']}",
                f"UHD_PWR: {row['uhd_max']:.1f}" if not pd.isna(row['uhd_max']) else "UHD Max: N/A",
                # "───────────────",
                f"RSRP: {row['RSRP']:.1f}",
                f"DL_Tput: {row['DL_Tput']:.1f}",
                f"SINR_TRS: {row['SINR_TRS']:.1f}",
                # "───────────────",
                # f"SINR: {row['SINR']:.1f}",
                # f"CQI: {row['CQI']:.1f}",
                # f"RI: {row['RI']:.1f}",
                # f"DL MCS: {row['DL_MCS']:.1f}",
                # f"DL BLER: {row['DL_BLER']:.1f}",
                # f"UL BLER: {row['UL_BLER']:.1f}",
                # "───────────────",
            ]
            return "<br>".join(lines)

        plot_df["hover_text"] = plot_df.apply(make_hover_text, axis=1)

        fig = go.Figure()

        for label in order:
            group = plot_df[plot_df["color_label"] == label]
            if group.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group["RSRP"],
                    y=group[metric],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    marker=dict(size=10, color=group["color"].iloc[0]),
                    text=group["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        ## Linear Regression (개별 커브)
        valid_df = plot_df.dropna(subset=["RSRP", metric])
        x_range = np.linspace(valid_df["RSRP"].min(), valid_df["RSRP"].max(), 200).reshape(-1, 1)

        for label, group in valid_df.groupby("color_label", observed=True):
            if len(group) < 2:
                continue  # 데이터 너무 적으면 스킵

            X = group["RSRP"].values.reshape(-1, 1)
            y = group[metric].values

            model = LinearRegression().fit(X, y)
            y_pred = model.predict(x_range)

            color = group["color"].iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=x_range.flatten(),
                    y=y_pred,
                    mode="lines",
                    name=f"Trend ({label})",
                    line=dict(color=color, width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=label,
                )
            )

        ## Linear Regression (단일 커브)
        # valid = plot_df.dropna(subset=["RSRP", "DL_Tput"])

        # X = valid["RSRP"].values.reshape(-1, 1)
        # y = valid["DL_Tput"].values

        # model = LinearRegression().fit(X, y)
        # x_range = np.linspace(valid["RSRP"].min(), valid["RSRP"].max(), 200)
        # y_pred = model.predict(x_range.reshape(-1, 1))

        # fig.add_trace(
        #     go.Scatter(
        #         x=x_range,
        #         y=y_pred,
        #         mode="lines",
        #         name="Linear Regression",
        #         line=dict(color="black", width=2, dash="dot"),
        #         hoverinfo="skip",
        #         showlegend=True,
        #         legendgroup="trend",
        #     )
        # )

        # # RSRP bin 평균
        # bins = np.arange(plot_df["RSRP"].min(), plot_df["RSRP"].max() + 1, 1)
        # plot_df["RSRP_bin"] = pd.cut(plot_df["RSRP"], bins=bins, right=False)
        # plot_df["RSRP_bin_center"] = plot_df["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)

        # avg_df = (
        #     plot_df.groupby("RSRP_bin_center", observed=True)["DL_Tput"]
        #     .mean()
        #     .reset_index()
        #     .sort_values("RSRP_bin_center")
        # )

        # fig.add_trace(
        #     go.Scatter(
        #         x=avg_df["RSRP_bin_center"],
        #         y=avg_df["DL_Tput"],
        #         # mode="lines+markers",
        #         mode='lines',
        #         name="Average Trend",
        #         # line=dict(color="black", width=2),
        #         # marker=dict(size=5, color="black"),
        #         line=dict(color="black", width=2, dash="dot"),
        #         hoverinfo="skip",
        #         showlegend=True,
        #         legendgroup="trend"
        #     )
        # )

        ## 3차 다항식 (단일 커브)
        # valid = plot_df.dropna(subset=["RSRP", "DL_Tput"])

        # X = valid["RSRP"].values.reshape(-1, 1)
        # y = valid["DL_Tput"].values

        # poly = PolynomialFeatures(degree=3)
        # X_poly = poly.fit_transform(X)
        # model = LinearRegression().fit(X_poly, y)

        # x_range = np.linspace(valid["RSRP"].min(), valid["RSRP"].max(), 200)
        # y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))

        # fig.add_trace(
        #     go.Scatter(
        #         x=x_range,
        #         y=y_pred,
        #         mode="lines",
        #         name="3rd Polynomial Curve",
        #         line=dict(color="black", width=2, dash="dot"),
        #         hoverinfo="skip",
        #         showlegend=True,
        #         legendgroup="trend",
        #     )
        # )

        ## 3차 다항식 (개별 커브)
        # x_range = np.linspace(plot_df["RSRP"].min(), plot_df["RSRP"].max(), 200)

        # for label, group in plot_df.groupby("color_label", observed=True):
        #     color = group["color"].iloc[0]
        #     valid = group[["RSRP", "DL_Tput"]].dropna().sort_values("RSRP")
        #     if len(valid) < 5:
        #         continue

        #     X = valid["RSRP"].values.reshape(-1, 1)
        #     y = valid["DL_Tput"].values

        #     poly = PolynomialFeatures(degree=3)
        #     X_poly = poly.fit_transform(X)
        #     model = LinearRegression().fit(X_poly, y)

        #     # 전체 범위(x_range)에 대해 예측 수행
        #     y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))

        #     fig.add_trace(
        #         go.Scatter(
        #             x=x_range,
        #             y=y_pred,
        #             mode="lines",
        #             line=dict(color=color, width=2, dash="dot"),
        #             hoverinfo="skip",
        #             showlegend=False,
        #             legendgroup=label
        #         )
        #     )

        if band == "n28":
            map_url = "https://joostone-ahn.github.io/nr-field-analysis/results/All/map_25m_DL_Tput_n28.html"
            title_text = (
                f"{metric.replace("_"," ")} over RSRP Scatter ({band}) "
                f"<a href='{map_url}' target='_blank' style='text-decoration:none; font-size:14px;'> [View Map]</a>"
            )
        else:
            title_text = f"{metric.replace("_", " ")} over RSRP Scatter ({band})"

        fig.update_layout(
            title=title_text,
            template="plotly_white",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="gray",
                font=dict(size=10),
                align="left",
            ),
            legend=dict(
                title=dict(
                    text="<span><b>  UHD Power [dBm]</b></span><br>",
                    font=dict(size=13),
                    side="top"
                ),
                font=dict(size=13),
                itemsizing="constant",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                tracegroupgap=8,
                yanchor="top",
                y=1.0,
                xanchor="left",
            )
        )
        fig.update_xaxes(
            title="RSRP [dBm]",
            autorange="reversed",
            dtick=5,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
            griddash="dot",
        )

        if metric == "SINR_TRS":
            y_title = "SINR TRS [dB]"
        elif metric == "DL_Tput":
            y_title = "DL Throughput [Mbps]"
        fig.update_yaxes(
            title=y_title,
            dtick=10,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
            griddash="dot",
        )
        os.makedirs(out_dir, exist_ok=True)
        if metric == "SINR_TRS":
            metric_title = "SINR"
        elif metric == "DL_Tput":
            metric_title = "Tput"
        out_path = os.path.join(out_dir, f"plot_{grid_size}m_{metric_title}_by_UHD_{band}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

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

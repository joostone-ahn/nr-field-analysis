import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import pandas as pd
import _common

def split_band_df(df_pair):
    common_cols = [c for c in df_pair.columns if not any(s in c for s in ["_n26", "_n28", "_diff"])]

    def extract_band(df, band):
        band_cols = [c for c in df.columns if c.endswith(band)]
        df_band = df[common_cols + band_cols].copy()

        new_cols = []
        for col in df_band.columns:
            if col.endswith(f"_mean_{band}"):
                new_cols.append(col.replace(f"_mean_{band}", ""))
            elif col.endswith(f"_std_{band}"):
                new_cols.append(col.replace(f"_std_{band}", "_std"))
            elif col.endswith(f"sample_count_{band}"):
                new_cols.append(col.replace(f"sample_count_{band}", "count"))
            else:
                new_cols.append(col)
        df_band.columns = new_cols

        df_band["Band"] = band
        return df_band

    df_n26 = extract_band(df_pair, "n26")
    df_n28 = extract_band(df_pair, "n28")

    return df_n26, df_n28

def plot_kpis_group_by_band(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -115
    RSRP_HIGH = -65

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SSB SINR [dB]", [-10, 35]),
        ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-18, -10]),
        ("RI", "Rank Indicator", [1, 2]),
        ("CQI", "CQI Index", [0, 15]),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]


    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for route_name in route_list:
        route_df = plot_df if route_name == "All" else plot_df[plot_df["route"] == route_name]

        for i, (metric, y_title, y_range) in enumerate(metrics, start=1):
            for band in order:
                group = route_df[route_df["Band"] == band]
                color = band_colors[band]
                valid = group.copy()

                bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
                valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)

                stats = valid.groupby("RSRP_bin", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
                stats["RSRP_center"] = stats["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                stats["RSRP_range"] = stats["RSRP_bin"].apply(lambda x: f"{x.right:.0f} ~ {x.left:.0f}")
                stats["SE"] = stats["std"] / np.sqrt(stats["count"])
                stats["CI"] = 1.96 * stats["SE"]

                stats["hover_text"] = stats.apply(
                    lambda r: (
                        f"<b>RSRP</b>: {r['RSRP_range']}<br>"
                        f"<b>{metric.replace('_', ' ')}</b>: {r['mean']:.2f}<br>"
                        f"<b>95% CI</b>: ±{r['CI']:.2f}<br>"
                        f"<b>counts</b>: {int(r['count'])}"
                    ),
                    axis=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=stats["RSRP_center"],
                        y=stats["mean"],
                        mode="lines+markers",
                        name=f"{route_name} | {band} | Avg ±95% CI",
                        legendgroup=f"{band}",
                        showlegend=(i == 1),
                        line=dict(color=color, width=1.3),
                        marker=dict(size=5, color=color),
                        text=stats["hover_text"],
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        )
                    ),
                    row=i, col=1
                )

                ci_df = stats.copy()
                ci_df["upper_CI"] = ci_df["mean"] + ci_df["CI"]
                ci_df["lower_CI"] = ci_df["mean"] - ci_df["CI"]

                fig.add_trace(
                    go.Scatter(
                        x=ci_df["RSRP_center"],
                        y=ci_df["upper_CI"],
                        mode="lines",
                        name=f"{route_name} | {band} | ±95% CI upper",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=i, col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=ci_df["RSRP_center"],
                        y=ci_df["lower_CI"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j + 2], 16) for j in (0, 2, 4)) + (0.2,)}",
                        name=f"{route_name} | {band} | ±95% CI lower",
                        legendgroup=f"{band}",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=i, col=1,
                )

                # median_df = valid.groupby("RSRP_bin", observed=True)[metric].median().reset_index()
                # median_df["RSRP_center"] = median_df["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                # fig.add_trace(
                #     go.Scatter(
                #         x=median_df["RSRP_center"],
                #         y=median_df[metric],
                #         mode="lines+markers",
                #         name=f"{route_name} | {band} | median",
                #         legendgroup=f"{band}_median",
                #         showlegend=(i == 1),
                #         line=dict(color=color, width=1, dash='dot'),
                #         marker=dict(size=5, color=color, symbol='square'),
                #         hoverinfo="skip",
                #     ),
                #     row=i, col=1
                # )

                # # raw data
                # fig.add_trace(
                #     go.Scatter(
                #         x=valid["RSRP"],
                #         y=valid[metric],
                #         mode="markers",
                #         name=f"{route_name} | {band} | raw",
                #         legendgroup=f"{band}_raw",
                #         showlegend=(i == 1),
                #         marker=dict(size=2, color=color, opacity=0.15),
                #         hoverinfo="skip",
                #     ),
                #     row=i, col=1
                # )

            fig.update_xaxes(
                title="RSRP [dBm]",
                autorange="reversed",
                dtick=5,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )
            fig.update_yaxes(
                title=y_title,
                range=y_range,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )

    buttons = []
    for route_name in route_list:
        visible_array = [route_name in trace.name for trace in fig.data]
        buttons.append(
            dict(
                label=route_name,
                method="update",
                args=[{"visible": visible_array}],
            )
        )
    initial_visible = ["All" in trace.name for trace in fig.data]
    for trace, visible in zip(fig.data, initial_visible):
        trace.visible = visible

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=LEGEND_Y,
                xanchor="left",
                buttons=buttons,
                showactive=True,
                bgcolor="white",
                bordercolor="gray",
            )
        ],
        height=SUBPLOT_HEIGHT * len(metrics),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
            itemsizing="constant",
        ),
        margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kpis_group_by_band.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")

def plot_kpis_group_by_site(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -115
    RSRP_HIGH = -65

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SSB SINR [dB]", [-10, 35]),
        ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-18, -10]),
        ("RI", "Rank Indicator", [1, 2]),
        ("CQI", "CQI Index", [0, 15]),
    ]

    route_colors = {
        # "All": "#808080",
        "All": "#1E90FF",
        "Namsan": "#FF4500",
        "Huam345-5": "#FFD700",
        "Huam415-1": "#32CD32",
    }
    route_list = list(route_colors.keys())
    band_list = ["n28", "n26"]

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for band_name in band_list:
        band_df = plot_df[plot_df["Band"] == band_name]

        for i, (metric, y_title, y_range) in enumerate(metrics, start=1):
            for route_name, color in route_colors.items():
                group = band_df if route_name == "All" else band_df[band_df["route"] == route_name]
                valid = group.copy()
                if valid.empty:
                    continue

                bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
                valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)

                stats = valid.groupby("RSRP_bin", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
                stats["RSRP_center"] = stats["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                stats["RSRP_range"] = stats["RSRP_bin"].apply(lambda x: f"{x.right:.0f} ~ {x.left:.0f}")
                stats["SE"] = stats["std"] / np.sqrt(stats["count"])
                stats["CI"] = 1.96 * stats["SE"]

                stats["hover_text"] = stats.apply(
                    lambda r: (
                        f"<b>RSRP</b>: {r['RSRP_range']}<br>"
                        f"<b>{metric.replace('_', ' ')}</b>: {r['mean']:.2f}<br>"
                        f"<b>95% CI</b>: ±{r['CI']:.2f}<br>"
                        f"<b>counts</b>: {int(r['count'])}"
                    ),
                    axis=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=stats["RSRP_center"],
                        y=stats["mean"],
                        mode="lines+markers",
                        name=f"{band_name} | {route_name} | Avg ±95% CI",
                        legendgroup=f"{route_name}",
                        showlegend=(i == 1),
                        line=dict(color=color, width=1.3),
                        marker=dict(size=5, color=color),
                        text=stats["hover_text"],
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color),
                    ),
                    row=i, col=1
                )

                ci_df = stats.copy()
                ci_df["upper_CI"] = ci_df["mean"] + ci_df["CI"]
                ci_df["lower_CI"] = ci_df["mean"] - ci_df["CI"]

                fig.add_trace(
                    go.Scatter(
                        x=ci_df["RSRP_center"],
                        y=ci_df["upper_CI"],
                        mode="lines",
                        name=f"{band_name} | {route_name} | ±95% CI upper",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=i, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=ci_df["RSRP_center"],
                        y=ci_df["lower_CI"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j + 2], 16) for j in (0, 2, 4)) + (0.2,)}",
                        name=f"{band_name} | {route_name} | ±95% CI lower",
                        legendgroup=f"{route_name}",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title="RSRP [dBm]",
                autorange="reversed",
                dtick=5,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )
            fig.update_yaxes(
                title=y_title,
                range=y_range,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )

    # dropdown for band selection
    buttons = []
    for band_name in band_list:
        visible_array = [band_name in trace.name for trace in fig.data]
        buttons.append(
            dict(
                label=band_name,
                method="update",
                args=[{"visible": visible_array}],
            )
        )

    initial_visible = ["n28" in (trace.name or "") for trace in fig.data]
    for trace, visible in zip(fig.data, initial_visible):
        trace.visible = visible

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=LEGEND_Y,
                xanchor="left",
                buttons=buttons,
                showactive=True,
                bgcolor="white",
                bordercolor="gray",
            )
        ],
        height=SUBPLOT_HEIGHT * len(metrics),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
            itemsizing="constant",
        ),
        margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kpis_group_by_site.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")


def plot_kpis_each_test(df, out_dir, grid_size, rb_min, sample_min):
    metrics = [
        "RSRP", "RSRQ",
        "SINR_SSB", "SINR_TRS",
        "DL_RB",
        "DL_Tput",
        "DL_Tput_per_RB",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
    ]

    lat_factor, lon_factor = 111320, 88000
    df[f"lat_bin"] = (df["Lat"] * lat_factor // grid_size).astype(int)
    df[f"lon_bin"] = (df["Lon"] * lon_factor // grid_size).astype(int)

    df_map = df[df["route"].isin(['Namsan','Huam415-1','Huam345-5'])].copy()
    df_grid = _common.grid_kpi(df_map, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
    df = df.merge(
        df_grid[[f"lat_bin", f"lon_bin", f"loc_id"]],
        on=[f"lat_bin", f"lon_bin"],
        how="left"
    )
    df[f"loc_id"] = df[f"loc_id"].astype("Int64")

    test_list = sorted(df["test_no"].unique())
    for target_no in test_list:
        df_sub = df[df["test_no"] == target_no].copy()
        total_rows = len(metrics)

        fig = make_subplots(
            rows=total_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            specs=[[{"secondary_y": True}] for _ in range(total_rows)]
        )

        for i, metric in enumerate(metrics, start=1):
            df_pivot = (
                df_sub.pivot_table(index="TIME", columns="Band", values=metrics)
                .dropna()
                .reset_index()
            )

            df_pivot.columns = [
                f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != "" else col[0]
                for col in df_pivot.columns
            ]
            df_pivot = df_pivot.merge(
                df_sub[["TIME", f"loc_id"]],
                on="TIME",
                how="left"
            )

            for m in metrics:
                df_pivot[f"{m}_delta"] = df_pivot[f"{m}_n28"] - df_pivot[f"{m}_n26"]

            hover_texts = []
            for _, row in df_pivot.iterrows():
                time_val = row["TIME"].strftime("%H:%M:%S")
                loc_id_val = row[f"loc_id"]
                f"<b>loc_id:</b> {loc_id_val} "

                lines = [
                    f"<b>time:</b> {time_val}<br>"
                    f"<b>loc_id:</b> {loc_id_val}<br>",
                    "<b>Metric</b> | <b>n26</b> | <b>n28</b> | <b>Δ(n28−n26)</b>",
                    "--------------------------------------------"
                ]
                for m in metrics:
                    color = "#009900" if m == metric else "#000000"
                    delta_val = row[f"{m}_delta"]
                    delta_color = "blue" if delta_val > 0 else "red" if delta_val < 0 else "black"
                    line = (
                        f"<span style='color:{color};'>{m}</span> | "
                        f"{row[f'{m}_n26']:.2f} | {row[f'{m}_n28']:.2f} | "
                        f"<span style='color:{delta_color};'>{delta_val:+.2f}</span>"
                    )
                    lines.append(line)
                hover_texts.append("<br>".join(lines))

            fig.add_trace(go.Scatter(
                x=df_pivot["TIME"],
                y=df_pivot[f"loc_id"],
                mode="lines+markers",
                line=dict(color="gray", width=0.8),
                marker=dict(size=3),
                name=f"loc_id",
                legendgroup="loc_id_group",
                showlegend=(i == 1),
                hoverinfo="skip"
            ), row=i, col=1, secondary_y=True)

            for band, color in zip(["n26", "n28"], ["blue", "red"]):
                fig.add_trace(go.Scatter(
                    x=df_pivot["TIME"],
                    y=df_pivot[f"{metric}_{band}"],
                    mode="lines+markers",
                    line=dict(color=color, width=1),
                    marker=dict(size=3),
                    name=band,
                    legendgroup=f"{band}_group",
                    showlegend=(i == 1),
                    hoverinfo="text" if band == "n28" else "skip",
                    text=hover_texts if band == "n28" else None
                ), row=i, col=1, secondary_y=False)

            fig.update_yaxes(
                title_text=metric,
                row=i, col=1,
                secondary_y=False,
                showgrid=True,
                zeroline=False,
                gridcolor="rgba(200, 200, 200, 0.5)",
                gridwidth=0.8,
            )
            fig.update_yaxes(
                title_text=f"loc_id",
                color="gray",
                row=i, col=1,
                secondary_y=True,
                showgrid=True,
                zeroline=False,
                gridcolor="rgba(200, 200, 200, 0.4)",
                gridwidth=0.8,
                griddash="dot"
            )

        fig.update_layout(
            title=f"[{target_no}] KPI trends (n26 vs n28)",
            height=300 * total_rows,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="top", y=1.02,
                xanchor="center", x=0.5,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(200,200,200,0.4)",
                borderwidth=1
            ),
            margin=dict(t=150, b=60),
            uirevision=True
        )

        date = target_no.split("_")[0]
        test_num = target_no.split("_")[1]
        route = target_no.split("_")[2]

        os.makedirs(out_dir, exist_ok=True)
        save_dir = os.path.join(out_dir, f"grid_{grid_size}m", date, route)
        os.makedirs(save_dir, exist_ok=True)
        out_path_html = os.path.join(save_dir, f"TEST_{test_num}.html")
        pio.write_html(fig, file=out_path_html, include_plotlyjs="cdn", full_html=True)
        print(f"Saved HTML: {out_path_html}")

def rb_each_test(df, out_dir, rb_min):
    metric = "DL_RB"
    test_list = sorted(df["test_no"].unique())
    date_list = sorted(set([t.split("_")[0] for t in test_list]))

    for date in date_list:
        date_tests = [t for t in test_list if t.startswith(date)]
        fig, axes = plt.subplots(len(date_tests), 1, figsize=(16, 4 * len(date_tests)), sharex=False)

        if len(date_tests) == 1:
            axes = [axes]

        for i, target_no in enumerate(date_tests):
            ax = axes[i]
            df_sub = df[df["test_no"] == target_no]

            # pivot: Band별 DL_RB
            df_pivot = (
                df_sub.pivot_table(index="TIME", columns="Band", values=metric)
                .dropna()
                .reset_index()
            )
            df_pivot["idx"] = range(len(df_pivot))

            ymin = df_pivot[["n26", "n28"]].min().min()
            ymax = df_pivot[["n26", "n28"]].max().max()
            if metric == 'DL_RB':
                ymax = 50
                ymin = rb_min

            # n26 / n28 plot
            ax.plot(df_pivot["idx"], df_pivot["n26"], label="n26", color="blue", linewidth=0.8, alpha=0.7)
            ax.plot(df_pivot["idx"], df_pivot["n28"], label="n28", color="red", linewidth=0.8, alpha=0.7)

            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(f"[{target_no}] DL_RB (n26 vs n28)", fontsize=11, pad=5)
            ax.set_xlabel("Time Index")
            ax.set_ylabel("DL_RB")
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        os.makedirs(out_dir, exist_ok=True)
        save_dir = os.path.join(out_dir, f"plot_RB_each_test")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{date}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        # plt.show()
        print(f"Saved: {out_path}")

def plot_grid_kpi(df, out_dir, grid_size, rb_min, sample_min):
    df_map = df[df['route'].isin(['Namsan','Huam415-1','Huam345-5'])]
    df_pair = _common.grid_kpi(df_map, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
    df_n26, df_n28 = split_band_df(df_pair)
    plot_df = pd.concat([df_n26, df_n28], axis=0)
    plot_df = plot_df[(plot_df["RSRP"] <= -60) & (plot_df["RSRP"] >= -120)]

    df_fixed = df[df["route"] == "Fixed-point"].copy()

    if grid_size == 30:
        marker_size = 15
    elif grid_size == 5:
        marker_size = 5

    def make_hover_text(row):
        lines = [
            "────────────────────────",
            f"<b>band</b> : {row['Band']}",
            f"<b>route</b> : {row['route']}",
            "────────────────────────",
            f"<b>DL_Tput</b> : {row['DL_Tput']:.1f} Mbps",
            f"<b>DL_RB</b> : {row['DL_RB']:.1f}",
            "────────────────────────",
            f"<b>RSRP</b> : {row['RSRP']:.1f} dBm",
            f"<b>SINR_SSB</b> : {row['SINR_SSB']:.1f} dB",
            f"<b>SINR_TRS</b> : {row['SINR_TRS']:.1f} dB",
            f"<b>RSRQ</b> : {row['RSRQ']:.1f} dB",
            "────────────────────────",
        ]
        if "loc_id" in row.index:
            lines[2] = f"<b>route / loc_id</b> : {row['route']} / {row['loc_id']}"
        return "<br>".join(lines)

    plot_df["hover_text"] = plot_df.apply(make_hover_text, axis=1)
    df_fixed["hover_text"] = df_fixed.apply(make_hover_text, axis=1)

    metrics = [
        "SINR_SSB",
        "SINR_TRS",
        "RSRQ",
        "DL_Tput"
    ]

    # 색상 정의
    band_colors = {
        "n28": "#FF4500", # 빨강
        "n26": "#1E90FF"  # 파랑
    }
    fixed_colors = {
        "n28": "#FF8C00", # 주황
        "n26": "#228B22"  # 초록
    }
    order = ["n28", "n26"]

    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    for metric in metrics:
        fig = go.Figure()

        for route_name in route_list:
            if route_name == "All":
                route_df = plot_df.copy()
            else:
                route_df = plot_df[plot_df["route"] == route_name]

            for band in order:
                group = route_df[route_df["Band"] == band]
                if group.empty:
                    continue

                color = band_colors.get(band, "gray")

                fig.add_trace(go.Scatter(
                    x=group["RSRP"],
                    y=group[metric],
                    mode="markers",
                    name=f"{band} raw ({route_name})",
                    legendgroup=f"{band}_{route_name}",
                    marker= dict(size=marker_size, color=color, opacity=0.3),
                    text=group["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="white",
                        bordercolor=color,
                        font=dict(color="gray")
                    ),
                    visible=(route_name == "All"),
                ))

                valid = group.dropna(subset=["RSRP", metric])
                if not valid.empty:
                    bins = np.arange(-120, -59, 5)
                    valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)
                    mean_df = (
                        valid.groupby("RSRP_bin", observed=True)[metric]
                        .mean()
                        .reset_index()
                    )
                    mean_df["RSRP_center"] = mean_df["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)

                    fig.add_trace(go.Scatter(
                        x=mean_df["RSRP_center"],
                        y=mean_df[metric],
                        mode="lines+markers",
                        name=f"{band} avg ({route_name})",
                        legendgroup=f"{band}_{route_name}",
                        line= dict(color=color, width=3, dash="dot"),
                        marker= dict(size=7, color=color),
                        hoverinfo="skip",
                        visible=(route_name == "All"),
                    ))

        for band in order:
            fixed_group = df_fixed[df_fixed["Band"] == band]
            if not fixed_group.empty:
                fixed_color = fixed_colors[band]
                fig.add_trace(go.Scatter(
                    x=fixed_group["RSRP"],
                    y=fixed_group[metric],
                    mode="markers",
                    name=f"{band} raw (Fixed-point)",
                    legendgroup=f"Fixed-{band}",
                    marker=dict(size=3, color=fixed_color, opacity=0.8),
                    text=fixed_group["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="white",
                        bordercolor=fixed_color,
                        font=dict(color="gray")
                    ),
                    visible=True,
                ))

        buttons = []
        for route_name in route_list:
            visible_flags = []
            for trace in fig.data:
                trace_name = trace.name
                if "(Fixed-point)" in trace_name:
                    visible_flags.append(True)
                elif route_name == "All":
                    visible_flags.append("(All)" in trace_name)
                else:
                    visible_flags.append(f"({route_name})" in trace_name)
            buttons.append(dict(
                label=route_name,
                method="update",
                args=[
                    {"visible": visible_flags},
                    {"title.text": f"{metric.replace('_',' ')} over RSRP ({route_name})"}
                ]
            ))

        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.07, y=1.05,
                xanchor="center",
                yanchor="top",
                bgcolor="white",
                bordercolor="lightgray",
                borderwidth=1,
                pad=dict(r=10, t=5, b=5),
            )],
            title=f"{metric.replace('_',' ')} over RSRP",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", bordercolor="gray", font=dict(size=10)),
            legend=dict(
                title="<b>Field Band</b>",
                font=dict(size=12),
                itemsizing="constant",
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=1.15,
                # bordercolor="lightgray",
                # borderwidth=1
            ),
            margin=dict(l=60, r=160, t=100, b=60),
        )

        fig.update_xaxes(
            title="RSRP [dBm]",
            autorange="reversed",
            dtick=5,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
        )

        if metric == "SINR_TRS":
            y_title = "SINR TRS [dB]"
        elif metric == "SINR_SSB":
            y_title = "SINR SSB [dB]"
        elif metric == "DL_Tput":
            y_title = "DL Throughput [Mbps]"
        else:
            y_title = metric
        fig.update_yaxes(title=y_title, gridcolor="rgba(0,0,0,0.15)")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"cmpr_{metric}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

def plot_grid_kpi_group_by_uhd(df, out_dir, grid_size, rb_min, sample_min, band):

    if band not in ["n26", "n28"]:
        ValueError("band must be n28 or n26")

    metrics = [
        "SINR_TRS",
        "DL_Tput",
        # "DL_Tput_per_RB",
    ]

    for metric in metrics:
        df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
        df_n26, df_n28 = split_band_df(df_pair)
        plot_df = df_n28.copy() if band == "n28" else df_n26.copy()
        plot_df = plot_df[(plot_df["RSRP"] <= -60) & (plot_df["RSRP"] >= -120)]
        # print(plot_df.info())

        group_name = 'UHD Power'
        color_col = "uhd_max"

        # valid_vals = plot_df[color_col].dropna()
        # if len(valid_vals) > 0:
        plot_df = plot_df.dropna(subset=[color_col])
        if not plot_df.empty:
            q1, q2, q3 = -40, -35, -30
            def color_by_uhd(v):
                if v >= q3:
                    return "#FF4500", f"PWR ≥ {q3:.0f}"  # 빨강 (높음)
                elif v >= q2:
                    return "#FFD700", f"{q2:.0f} ≤ PWR < {q3:.0f}"  # 노랑
                elif v >= q1:
                    return "#32CD32", f"{q1:.0f} ≤ PWR < {q2:.0f}"  # 초록
                else:
                    return "#1E90FF", f"PWR < {q1:.0f}"  # 파랑 (낮음)

            plot_df[["color", "color_label"]] = plot_df[color_col].apply(lambda v: pd.Series(color_by_uhd(v)))
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
            def ci95(std, n):
                if pd.isna(std) or pd.isna(n) or n <= 1:
                    return None
                return 1.96 * std / np.sqrt(n)

            ci = {
                "DL_Tput": ci95(row.get("DL_Tput_std"), row.get("count")),
                "DL_RB": ci95(row.get("DL_RB_std"), row.get("count")),
                "DL_Tput_per_RB": ci95(row.get("DL_Tput_per_RB_std"), row.get("count")),
                "RSRP": ci95(row.get("RSRP_std"), row.get("count")),
                "SINR_TRS": ci95(row.get("SINR_TRS_std"), row.get("count")),
            }

            def fmt(val, ci_val):
                if pd.isna(val):
                    return "null"
                if ci_val is None:
                    return f"{val:.1f}"
                return f"{val:.1f} <span style='color:#777;'>(±{ci_val:.2f})</span>"

            lines = [
                "────────────────────────",
                f"<b>UHD_PWR</b> : {row['uhd_max']:.1f}" if not pd.isna(row['uhd_max']) else "<b>UHD_PWR</b>: null",
                f"<b>route / loc_id</b> : {row['route']} / {row['loc_id']}",
                "────────────────────────",
                f"<b>samples</b> : {row['count']}",
                f"<b>RSRP</b> : {fmt(row['RSRP'], ci['RSRP'])}",
                f"<b>SINR_TRS</b> : {fmt(row['SINR_TRS'], ci['SINR_TRS'])}",
                f"<b>DL_Tput</b> : {fmt(row['DL_Tput'], ci['DL_Tput'])}",
                f"<b>DL_RB</b> : {fmt(row['DL_RB'], ci['DL_RB'])}",
                f"<b>DL_Tput_per_RB</b> : {fmt(row['DL_Tput_per_RB'], ci['DL_Tput_per_RB'])}",
                "────────────────────────",
            ]
            return "<br>".join(lines)

        plot_df["hover_text"] = plot_df.apply(make_hover_text, axis=1)

        fig = go.Figure()

        for label in order:
            group = plot_df[plot_df["color_label"] == label]
            if group.empty:
                continue

            color = group["color"].iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=group["RSRP"],
                    y=group[metric],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    marker=dict(size=10, color=color),
                    text=group["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

            # valid = (
            #     group.dropna(subset=["RSRP", metric])
            #     .replace([np.inf, -np.inf], np.nan)
            #     .dropna(subset=[metric, "RSRP"])
            #     .copy()
            # )
            # if len(valid) < 5:
            #     continue
            #
            # bin_size = 1
            # bins = np.arange(-120, -59, bin_size)
            # valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)
            #
            # mean_df = (
            #     valid.groupby("RSRP_bin", observed=True)[metric]
            #     .mean()
            #     .reset_index()
            #     .dropna()
            # )
            # mean_df["RSRP_center"] = mean_df["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
            #
            # if not mean_df.empty:
            #     fig.add_trace(
            #         go.Scatter(
            #             x=mean_df["RSRP_center"],
            #             y=mean_df[metric],
            #             mode="lines+markers",
            #             name=f"{label} avg({bin_size}dB)",
            #             legendgroup=label,
            #             line=dict(color=color, width=3, dash="dot"),
            #             marker=dict(size=10, color=color),
            #             hoverinfo="skip",
            #             showlegend=True,
            #         )
            #     )

        title_text = f"{metric.replace("_", " ")} over RSRP group by {group_name} ({band})"

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
                    text=f"<span><b>  {group_name}</b></span><br>",
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
        elif metric == "DL_Tput_per_RB":
            y_title = "DL Throughput per RB [Mbps]"
        fig.update_yaxes(
            title=y_title,
            # dtick=10,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
            griddash="dot",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{band}_{metric}_by_uhd.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import pandas as pd
import _common

def dist_kpis_pdf_group_by_band(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SINR [dB]", [-5, 45]),
        # ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)

    for b_idx, b in enumerate(bins[:-1]):
        rsrp_min, rsrp_max = b, b + rsrp_bin
        bin_df = plot_df[(plot_df["RSRP"] >= rsrp_min) & (plot_df["RSRP"] < rsrp_max)].copy()

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=VERTICAL_SPACING,
        )

        for route_name in route_list:
            route_df = bin_df if route_name == "All" else bin_df[bin_df["route"] == route_name]

            for i, (metric, x_title, x_range) in enumerate(metrics, start=1):
                x_min, x_max = None, None

                for band in order:
                    group = route_df[route_df["Band"] == band]
                    if len(group) < 5:
                        continue

                    data = group[metric].dropna().values
                    counts, bin_edges = np.histogram(data, bins=30, density=True)
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)
                    total_count = len(data)

                    if x_min is None or centers.min() < x_min:
                        x_min = centers.min()
                    if x_max is None or centers.max() > x_max:
                        x_max = centers.max()

                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=counts,
                            mode="lines+markers",
                            name=f"{route_name} | {band}",
                            legendgroup=f"{band}",
                            line=dict(color=band_colors[band], width=2),
                            marker=dict(size=4, color=band_colors[band]),
                            hovertemplate=(
                                f"<b>{band}</b><br>"
                                f"{metric}: %{{x:.1f}}<br>"
                                f"Density: %{{y:.4f}}"
                                f" (%{{customdata[0]}} / {total_count})<extra></extra>"
                            ),
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
                            customdata=np.array(raw_counts).reshape(-1, 1),  # 👈 카운트 전달
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1,
                    )

                if x_min is not None and x_max is not None:
                    target_ticks = 15
                    dtick = round((x_max - x_min) / target_ticks, 1)
                    dtick = max(dtick, 0.1)
                else:
                    dtick = None

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    dtick=dtick,
                    range=x_range,
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="PDF (Measured Density)",
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
            ),
            margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        )

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "group_by_band", "PDF")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"PDF_RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

def plot_kpis_cdf_group_by_band(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SINR [dB]", [-5, 45]),
        # ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)

    for b_idx, b in enumerate(bins[:-1]):
        rsrp_min, rsrp_max = b, b + rsrp_bin
        bin_df = plot_df[(plot_df["RSRP"] >= rsrp_min) & (plot_df["RSRP"] < rsrp_max)].copy()

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=VERTICAL_SPACING,
        )

        for route_name in route_list:
            route_df = bin_df if route_name == "All" else bin_df[bin_df["route"] == route_name]

            for i, (metric, x_title, x_range) in enumerate(metrics, start=1):
                x_min, x_max = None, None

                for band in order:
                    group = route_df[route_df["Band"] == band]
                    if len(group) < 5:
                        continue

                    data = group[metric].dropna().values
                    counts, bin_edges = np.histogram(data, bins=30, density=True)
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)
                    total_count = len(data)

                    cdf = np.cumsum(counts * np.diff(bin_edges))
                    cdf = np.clip(cdf, 0, 1)

                    if x_min is None or centers.min() < x_min:
                        x_min = centers.min()
                    if x_max is None or centers.max() > x_max:
                        x_max = centers.max()

                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=cdf,
                            mode="lines+markers",
                            name=f"{route_name} | {band}",
                            legendgroup=f"{band}",
                            line=dict(color=band_colors[band], width=2),
                            marker=dict(size=4, color=band_colors[band]),
                            hovertemplate=(
                                f"<b>{band}</b><br>"
                                f"{metric}: %{{x:.1f}}<br>"
                                f"CDF: %{{y:.3f}}"
                                f" (%{{customdata[0]}} / {total_count})<extra></extra>"
                            ),
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
                            customdata=np.array(raw_counts).reshape(-1, 1),  # 👈 카운트 전달
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1,
                    )

                if x_min is not None and x_max is not None:
                    target_ticks = 15
                    dtick = round((x_max - x_min) / target_ticks, 1)
                    dtick = max(dtick, 0.1)
                else:
                    dtick = None

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    dtick=dtick,
                    range=x_range,
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="CDF (Cumulative Probability)",
                    gridcolor="rgba(0,0,0,0.15)",
                    range=[0, 1.05],
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
            ),
            margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        )

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "group_by_band", "CDF")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"CDF_RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")
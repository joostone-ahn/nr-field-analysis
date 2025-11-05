import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import pandas as pd
import _common

def dist_kpis_group_by_site(df, out_dir, rb_min, rsrp_bin):
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
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
    ]

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFD700",
        "Huam415-1": "#32CD32",
    }
    route_list = list(route_colors.keys())
    band_list = ["n28", "n26"]

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
            specs=[[{"secondary_y": True}] for _ in metrics],
        )

        for band_name in band_list:
            band_df = bin_df[bin_df["Band"] == band_name]

            for i, (metric, x_title, x_range) in enumerate(metrics, start=1):
                for route_name, color in route_colors.items():
                    group = band_df[band_df["route"] == route_name]
                    if len(group) < 5:
                        continue

                    data = group[metric].dropna().values
                    counts, bin_edges = np.histogram(data, bins=30, density=True)
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)
                    total_count = len(data)
                    cdf = np.cumsum(counts * np.diff(bin_edges))
                    cdf = np.clip(cdf, 0, 1)

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means, cqi_means = [],[]
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                                cqi_means.append(group.loc[in_bin, "CQI"].mean())
                            else:
                                sinr_means.append(np.nan)
                                cqi_means.append(np.nan)
                        customdata = np.stack((raw_counts, sinr_means, cqi_means), axis=-1)
                        hovertemplate_pdf = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CQI: %{{customdata[2]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                        hovertemplate_cdf = hovertemplate_pdf.replace("PDF", "CDF")
                    else:
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        tput_means = []

                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                            else:
                                tput_means.append(np.nan)

                        customdata = np.stack((raw_counts, tput_means), axis=-1)

                        hovertemplate_pdf = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"DL Tput: %{{customdata[1]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                        hovertemplate_cdf = hovertemplate_pdf.replace("PDF", "CDF")

                    # PDF
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=counts,
                            mode="lines+markers",
                            name=f"{band_name} | {route_name} | PDF",
                            legendgroup=f"{route_name}_pdf",
                            line=dict(color=color, width=0.8),
                            marker=dict(size=5, color=color),
                            customdata=customdata,
                            hovertemplate=hovertemplate_pdf,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
                            visible=(band_name == "n28"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1, secondary_y=False,
                    )

                    # CDF
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=cdf,
                            mode="lines+markers",
                            name=f"{band_name} | {route_name} | CDF",
                            legendgroup=f"{route_name}_cdf",
                            line=dict(color=color, width=2.0, dash="dash"),
                            marker=dict(size=5, color=color, symbol="square"),
                            customdata=customdata,
                            hovertemplate=hovertemplate_cdf,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
                            visible=(band_name == "n28"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1, secondary_y=True,
                    )

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="PDF (Probability Density Function)",
                    gridcolor="rgba(0,0,0,0.25)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="CDF (Cumulative Distribution Function)",
                    gridcolor="rgba(0,0,0,0.15)",
                    griddash="dot",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    tickformat=".2f",
                    row=i, col=1,
                    secondary_y=True,
                )

        # Band 드롭다운
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
        out_path = os.path.join(out_dir, "kpis_group_by_site", f"RSRP_bin_{rsrp_bin}dB")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

def dist_kpis_group_by_band(df, out_dir, rb_min, rsrp_bin):
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
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
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
            specs=[[{"secondary_y": True}] for _ in metrics],
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

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means, cqi_means = [],[]
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                                cqi_means.append(group.loc[in_bin, "CQI"].mean())
                            else:
                                sinr_means.append(np.nan)
                                cqi_means.append(np.nan)
                        customdata = np.stack((raw_counts, sinr_means, cqi_means), axis=-1)
                        hovertemplate_pdf = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CQI: %{{customdata[2]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                        hovertemplate_cdf = hovertemplate_pdf.replace("PDF", "CDF")
                    else:
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        tput_means = []

                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                            else:
                                tput_means.append(np.nan)

                        customdata = np.stack((raw_counts, tput_means), axis=-1)

                        hovertemplate_pdf = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"DL Tput: %{{customdata[1]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                        hovertemplate_cdf = hovertemplate_pdf.replace("PDF", "CDF")

                    # PDF
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=counts,
                            mode="lines+markers",
                            name=f"{route_name} | {band} | PDF",
                            legendgroup=f"{band}_pdf",
                            line=dict(color=band_colors[band], width=0.8),
                            marker=dict(size=5, color=band_colors[band]),
                            customdata=customdata,
                            hovertemplate=hovertemplate_pdf,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1, secondary_y=False,
                    )

                    # CDF
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=cdf,
                            mode="lines+markers",
                            name=f"{route_name} | {band} | CDF",
                            legendgroup=f"{band}_cdf",
                            line=dict(color=band_colors[band], width=2.0, dash="dash"),
                            marker=dict(size=5, color=band_colors[band], symbol="square"),
                            customdata=customdata,
                            hovertemplate=hovertemplate_cdf,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1, secondary_y=True,
                    )

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="PDF (Probability Density Function)",
                    gridcolor="rgba(0,0,0,0.25)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="CDF (Cumulative Distribution Function)",
                    gridcolor="rgba(0,0,0,0.15)",
                    griddash="dot",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    tickformat=".2f",
                    row=i, col=1,
                    secondary_y=True,
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
        out_path = os.path.join(out_dir, "kpis_group_by_band", f"RSRP_bin_{rsrp_bin}dB")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

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

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means = []
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                            else:
                                sinr_means.append(np.nan)
                        customdata = np.stack((raw_counts, sinr_means), axis=-1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<br><extra></extra>"
                        )
                    else:
                        customdata = np.array(raw_counts).reshape(-1, 1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"CDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=counts,
                            mode="lines+markers",
                            name=f"{route_name} | {band}",
                            legendgroup=f"{band}",
                            line=dict(color=band_colors[band], width=2),
                            marker=dict(size=4, color=band_colors[band]),
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
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
                    # range=x_range,
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

def dist_kpis_cdf_group_by_band(df, out_dir, rb_min, rsrp_bin):
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

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means = []
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                            else:
                                sinr_means.append(np.nan)
                        customdata = np.stack((raw_counts, sinr_means), axis=-1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<br><extra></extra>"
                        )
                    else:
                        customdata = np.array(raw_counts).reshape(-1, 1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"CDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )

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
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=band_colors[band]
                            ),
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
                    # range=x_range,
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
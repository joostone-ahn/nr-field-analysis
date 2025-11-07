import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import pandas as pd
import _common

def dist_uhd_by_minmax(df, out_dir):
    SUBPLOT_HEIGHT = 600
    TOP_MARGIN = 70
    LEGEND_Y = 1.08
    LEGEND_FONT_SIZE = 13

    metrics = [
        {"name": "uhd_avg", "label": "AVG", "color": "#1976D2"},
        {"name": "uhd_max", "label": "MAX", "color": "#D32F2F"},
        {"name": "uhd_min", "label": "MIN", "color": "#388E3C"},
    ]
    fig = go.Figure()

    for m in metrics:
        metric = m["name"]
        label = m["label"]
        color = m["color"]

        data = df[metric].dropna().values
        total = len(data)

        bins = np.arange(np.floor(data.min())-0.5, np.ceil(data.max())+0.5, 1)
        counts, bin_edges = np.histogram(data, bins=bins)
        cdf = np.cumsum(counts) / total
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hover_text = [f"{int(x)}/{total}" for x in np.cumsum(counts)]

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=cdf,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=1.2),
                text=hover_text,
                hovertemplate=(
                    "Power: %{x:.1f} dBm<br>"
                    "CDF: %{y:.2f}<br>"
                    "Count: %{text}<extra></extra>"
                ),
                hoverlabel=dict(
                    font=dict(size=11, color="white"),
                    bgcolor=color,
                ),
            )
        )

    center_x = -30
    x_min, x_max = df["uhd_min"].min(), df["uhd_max"].max()
    left_dist = abs(center_x - x_min)
    right_dist = abs(x_max - center_x)
    half_range = max(left_dist, right_dist)
    x_lower = center_x - half_range -2
    x_upper = center_x + half_range

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=SUBPLOT_HEIGHT,
        margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        xaxis=dict(
            title="UHD Power [dBm/12MHz]",
            range=[x_lower, x_upper],
            gridcolor="rgba(0,0,0,0.15)",
            dtick=3,
        ),
        yaxis=dict(
            title_text="CDF (Cumulative Distribution Function)",
            gridcolor="rgba(0,0,0,0.15)",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            tickformat=".2f",
            range=[-0.05, 1.05],
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    fig.add_vline(
        x=center_x,
        line=dict(color="black", width=1, dash="dot"),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"uhd_minmax.html")
    fig.write_html(out_file)
    print(f"✅ Saved: {out_file}")
def dist_uhd_by_site(df, out_dir, grid_size):
    SUBPLOT_HEIGHT = 600
    TOP_MARGIN = 70
    LEGEND_Y = 1.08
    LEGEND_FONT_SIZE = 13

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFD700",
        "Huam415-1": "#32CD32",
    }

    if grid_size == 30:
        loc_ranges = {
            "Huam345-5": (1, 82),
            "Huam415-1": (83, 121),
            "Namsan": (122, 145),
        }
    else:
        raise ValueError("grid_size must be 30")

    fig = go.Figure()
    metric = "uhd_avg"

    for route, (min_id, max_id) in loc_ranges.items():
        color = route_colors.get(route, "#000000")
        subset = df[(df["loc_id"] >= min_id) & (df["loc_id"] <= max_id)]
        data = subset[metric].dropna().values

        if len(data) == 0:
            continue

        total = len(data)
        bins = np.arange(np.floor(data.min()) - 0.5, np.ceil(data.max()) + 0.5, 1)
        counts, bin_edges = np.histogram(data, bins=bins)
        cdf = np.cumsum(counts) / total
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hover_text = [f"{int(x)}/{total}" for x in np.cumsum(counts)]

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=cdf,
                mode="lines+markers",
                name=f"{route}",
                line=dict(color=color, width=1.2),
                text=hover_text,
                hovertemplate=(
                    "Power: %{x:.1f} dBm<br>"
                    "CDF: %{y:.2f}<br>"
                    "Count: %{text}<extra></extra>"
                ),
                hoverlabel=dict(
                    font=dict(size=11, color="white"),
                    bgcolor=color,
                ),
            )
        )

    center_x = -30
    x_min, x_max = df["uhd_avg"].min(), df["uhd_avg"].max()
    left_dist = abs(center_x - x_min)
    right_dist = abs(x_max - center_x)
    half_range = max(left_dist, right_dist)
    x_lower = center_x - half_range -2
    x_upper = center_x + half_range

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=SUBPLOT_HEIGHT,
        margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        xaxis=dict(
            title="UHD Power [dBm/12MHz]",
            range=[x_lower, x_upper],
            gridcolor="rgba(0,0,0,0.15)",
            dtick=3,
        ),
        yaxis=dict(
            title="CDF (Cumulative Distribution Function)",
            gridcolor="rgba(0,0,0,0.15)",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            tickformat=".2f",
            range=[-0.05, 1.05],
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    fig.add_vline(
        x=center_x,
        line=dict(color="black", width=1, dash="dot"),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"uhd_by_site.html")
    fig.write_html(out_file)
    print(f"✅ Saved: {out_file}")
def dist_uhd_by_site_pdf_cdf(df, out_dir, grid_size):
    SUBPLOT_HEIGHT = 600
    TOP_MARGIN = 70
    LEGEND_Y = 1.08
    LEGEND_FONT_SIZE = 13

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFD700",
        "Huam415-1": "#32CD32",
    }

    # --- loc_id 범위 설정 ---
    if grid_size == 30:
        loc_ranges = {
            "Huam345-5": (1, 82),
            "Huam415-1": (83, 121),
            "Namsan": (122, 145),
        }
    elif grid_size == 5:
        loc_ranges = {
            "Huam345-5": (1, 503),
            "Huam415-1": (504, 726),
            "Namsan": (727, 933),
        }
    else:
        raise ValueError("grid_size must be either 5 or 30")

    # --- Subplot 생성 (PDF + CDF) ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09
    )

    metric = "uhd_avg"

    for route, (min_id, max_id) in loc_ranges.items():
        color = route_colors.get(route, "#000000")
        subset = df[(df["loc_id"] >= min_id) & (df["loc_id"] <= max_id)]
        data = subset[metric].dropna().values

        if len(data) == 0:
            continue

        total = len(data)
        bins = np.arange(np.floor(data.min()) - 0.5, np.ceil(data.max()) + 0.5, 1)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        pdf = counts / counts.sum()
        cdf = np.cumsum(counts) / total
        hover_text_cdf = [f"{int(x)}/{total}" for x in np.cumsum(counts)]

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=pdf,
                mode="lines+markers",
                name=f"{route}",
                legendgroup=f"{route}",
                showlegend=True,
                line=dict(color=color, width=1.4),
                text=[f"{v:.3f}" for v in pdf],
                hovertemplate="Power: %{x:.1f} dBm<br>PDF(norm): %{y:.3f}<extra></extra>",
                hoverlabel=dict(font=dict(size=11, color="white"), bgcolor=color),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=cdf,
                mode="lines+markers",
                name=f"{route}_CDF",
                legendgroup=f"{route}",
                showlegend=False,
                line=dict(color=color, width=1.4),
                text=hover_text_cdf,
                hovertemplate="Power: %{x:.1f} dBm<br>CDF: %{y:.2f}<br>Count: %{text}<extra></extra>",
                hoverlabel=dict(font=dict(size=11, color="white"), bgcolor=color),
            ),
            row=2, col=1
        )

    center_x = -30
    x_min, x_max = df["uhd_avg"].min(), df["uhd_avg"].max()
    left_dist = abs(center_x - x_min)
    right_dist = abs(x_max - center_x)
    half_range = max(left_dist, right_dist)
    x_lower = center_x - half_range - 2
    x_upper = center_x + half_range

    fig.update_layout(
        template="plotly_white",
        height=SUBPLOT_HEIGHT * 1.6,
        margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    fig.update_xaxes(
        title="UHD Power [dBm/12MHz]",
        range=[x_lower, x_upper],
        gridcolor="rgba(0,0,0,0.15)",
        dtick=3,
        row=1, col=1
    )
    fig.update_yaxes(
        title="Probability Density Function (PDF)",
        gridcolor="rgba(0,0,0,0.15)",
        row=1, col=1
    )

    fig.update_xaxes(
        title="UHD Power [dBm/12MHz]",
        range=[x_lower, x_upper],
        gridcolor="rgba(0,0,0,0.15)",
        dtick=3,
        row=2, col=1
    )
    fig.update_yaxes(
        title="Cumulative Distribution Function (CDF)",
        gridcolor="rgba(0,0,0,0.15)",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        range=[-0.05, 1.05],
        row=2, col=1
    )

    for r in [1, 2]:
        fig.add_vline(x=center_x, line=dict(color="black", width=1, dash="dot"), row=r, col=1)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "uhd_by_site.html")
    fig.write_html(out_file)
    print(f"✅ Saved: {out_file}")

def dist_kpis_by_site_rsrp_bin(df, out_dir, rb_min, sample_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFC107",
        "Huam415-1": "#32CD32",
    }
    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    route_list = list(route_colors.keys())
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

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

            for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
                x_min, x_max = x_range[0], x_range[1]
                bin_size = 10

                for route_name, color in route_colors.items():
                    group = band_df[band_df["route"] == route_name]
                    data = group[metric].dropna().values
                    total_count = len(data)
                    if total_count < sample_min:
                        continue

                    counts, bin_edges = np.histogram(data, bins=bin_size, density=False)
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    pdf = counts / counts.sum()

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means, cqi_means = [], []
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                                cqi_means.append(group.loc[in_bin, "CQI"].mean())
                            else:
                                sinr_means.append(np.nan)
                                cqi_means.append(np.nan)
                        customdata = np.stack((counts, sinr_means, cqi_means), axis=-1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CQI: %{{customdata[2]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                    else:
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        tput_means = []

                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                            else:
                                tput_means.append(np.nan)

                        customdata = np.stack((counts, tput_means), axis=-1)

                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"DL Tput: %{{customdata[1]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=pdf,
                            mode="lines+markers",
                            name=f"{band_name} | {route_name}",
                            legendgroup=f"{route_name}_pdf",
                            line=dict(color=color, width=1.2),
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
                            visible=(band_name == "n28"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1
                    )

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="Probability Density Function (PDF)",
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )

        # Band 드롭 다운
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
            autosize=True,
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
        out_path = os.path.join(out_dir, f"kpis_by_site")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")
def dist_kpis_by_site_pdf(df, out_dir, rb_min):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFC107",
        "Huam415-1": "#32CD32",
    }
    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    route_list = list(route_colors.keys())
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for band_name in band_list:
        band_df = plot_df[plot_df["Band"] == band_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max-x_min) / bin_width)

            for route_name, color in route_colors.items():
                group = band_df[band_df["route"] == route_name]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)

                counts, bin_edges = np.histogram(data, bins=bin_size, density=False)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                pdf = counts / counts.sum()

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
                    customdata = np.stack((counts, sinr_means, cqi_means), axis=-1)
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
                else:
                    bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                    tput_means = []

                    for bin_i in range(len(bin_edges) - 1):
                        in_bin = (bin_indices == bin_i)
                        if np.any(in_bin):
                            tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                        else:
                            tput_means.append(np.nan)

                    customdata = np.stack((counts, tput_means), axis=-1)

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=pdf,
                        mode="lines+markers",
                        name=f"{band_name} | {route_name}",
                        legendgroup=f"{route_name}_pdf",
                        line=dict(color=color, width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        ),
                        visible=(band_name == "n28"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                dtick=bin_width,
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="Probability Density Function (PDF)",
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )

    # Band 드롭 다운
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_site_pdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")
def dist_kpis_by_site_cdf(df, out_dir, rb_min):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    route_colors = {
        "Namsan": "#FF4500",
        "Huam345-5": "#FFC107",
        "Huam415-1": "#32CD32",
    }
    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    route_list = list(route_colors.keys())
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for band_name in band_list:
        band_df = plot_df[plot_df["Band"] == band_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max-x_min) / bin_width)

            for route_name, color in route_colors.items():
                group = band_df[band_df["route"] == route_name]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)
                counts, bin_edges = np.histogram(data, bins=bin_size, density=True)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)

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
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
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

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                # CDF
                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=cdf,
                        mode="lines+markers",
                        name=f"{band_name} | {route_name}",
                        legendgroup=f"{route_name}_cdf",
                        line=dict(color=color, width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        ),
                        visible=(band_name == "n28"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="CDF (Cumulative Distribution Function)",
                gridcolor="rgba(0,0,0,0.15)",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                tickformat=".2f",
                row=i, col=1,
                range=[-0.05, 1.05],
            )

    # Band 드롭 다운
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_site_cdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")

def dist_kpis_by_band_rsrp_bin(df, out_dir, rb_min, sample_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

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

            for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
                x_min, x_max = x_range[0], x_range[1]
                bin_size = 10

                for band in order:
                    color = band_colors[band]
                    group = route_df[route_df["Band"] == band]
                    data = group[metric].dropna().values
                    total_count = len(data)
                    if total_count < sample_min:
                        continue

                    counts, bin_edges = np.histogram(data, bins=bin_size, density=False)
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    pdf = counts / counts.sum()

                    if centers.min() < x_min:
                        x_min = centers.min()
                    if centers.max() > x_max:
                        x_max = centers.max()

                    if metric == "DL_Tput":
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        sinr_means, cqi_means = [], []
                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                                cqi_means.append(group.loc[in_bin, "CQI"].mean())
                            else:
                                sinr_means.append(np.nan)
                                cqi_means.append(np.nan)
                        customdata = np.stack((counts, sinr_means, cqi_means), axis=-1)
                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"SINR: %{{customdata[1]:.1f}}<br>"
                            f"CQI: %{{customdata[2]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )
                    else:
                        bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                        tput_means = []

                        for bin_i in range(len(bin_edges) - 1):
                            in_bin = (bin_indices == bin_i)
                            if np.any(in_bin):
                                tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                            else:
                                tput_means.append(np.nan)

                        customdata = np.stack((counts, tput_means), axis=-1)

                        hovertemplate = (
                            f"{metric}: %{{x:.1f}}<br>"
                            f"DL Tput: %{{customdata[1]:.1f}}<br>"
                            f"PDF: %{{y:.2f}}<br>"
                            f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                        )

                    # CDF
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=pdf,
                            mode="lines+markers",
                            name=f"{route_name} | {band}",
                            legendgroup=f"{band}_pdf",
                            line=dict(color=color, width=1.2),
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1
                    )

                fig.update_xaxes(
                    title_text=x_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="Probability Density Function (PDF)",
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
            autosize=True,
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
        out_path = os.path.join(out_dir, f"kpis_by_band")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"RSRP_{rsrp_min}_to_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")
def dist_kpis_by_band_pdf(df, out_dir, rb_min):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for route_name in route_list:
        route_df = plot_df if route_name == "All" else plot_df[plot_df["route"] == route_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max-x_min) / bin_width)

            for band in order:
                color = band_colors[band]
                group = route_df[route_df["Band"] == band]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)

                counts, bin_edges = np.histogram(data, bins=bin_size, density=False)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                pdf = counts / counts.sum()

                if centers.min() < x_min:
                    x_min = centers.min()
                if centers.max() > x_max:
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
                    customdata = np.stack((counts, sinr_means, cqi_means), axis=-1)
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
                else:
                    bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                    tput_means = []

                    for bin_i in range(len(bin_edges) - 1):
                        in_bin = (bin_indices == bin_i)
                        if np.any(in_bin):
                            tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                        else:
                            tput_means.append(np.nan)

                    customdata = np.stack((counts, tput_means), axis=-1)

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                # CDF
                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=pdf,
                        mode="lines+markers",
                        name=f"{route_name} | {band}",
                        legendgroup=f"{band}_pdf",
                        line=dict(color=color, width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        ),
                        visible=(route_name == "All"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                dtick=bin_width,
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="Probability Density Function (PDF)",
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_band_pdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")
def dist_kpis_by_band_cdf(df, out_dir, rb_min):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for route_name in route_list:
        route_df = plot_df if route_name == "All" else plot_df[plot_df["route"] == route_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max - x_min) / bin_width)

            for band in order:
                color = band_colors[band]
                group = route_df[route_df["Band"] == band]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)
                counts, bin_edges = np.histogram(data, bins=bin_size, density=True)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)
                cdf = np.cumsum(counts * np.diff(bin_edges))
                cdf = np.clip(cdf, 0, 1)

                if centers.min() < x_min:
                    x_min = centers.min()
                if centers.max() > x_max:
                    x_max = centers.max()

                if metric == "DL_Tput":
                    bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                    sinr_means, cqi_means = [], []
                    for bin_i in range(len(bin_edges) - 1):
                        in_bin = (bin_indices == bin_i)
                        if np.any(in_bin):
                            sinr_means.append(group.loc[in_bin, "SINR_SSB"].mean())
                            cqi_means.append(group.loc[in_bin, "CQI"].mean())
                        else:
                            sinr_means.append(np.nan)
                            cqi_means.append(np.nan)
                    customdata = np.stack((raw_counts, sinr_means, cqi_means), axis=-1)
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
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

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                # CDF
                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=cdf,
                        mode="lines+markers",
                        name=f"{route_name} | {band}",
                        legendgroup=f"{band}_cdf",
                        line=dict(color=band_colors[band], width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=band_colors[band]
                        ),
                        visible=(route_name == "All"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="CDF (Cumulative Distribution Function)",
                gridcolor="rgba(0,0,0,0.15)",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                tickformat=".2f",
                row=i, col=1,
                range=[-0.05, 1.05],
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_band_cdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")

def dist_kpis_by_uhd_each_band_pdf(df, out_dir, rb_min, grid_size):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    route_list = [
        "Namsan",
        "Huam345-5",
        "Huam415-1",
    ]

    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()
    plot_df = _common.assign_uhd_pwr_raw(plot_df, grid_size=grid_size)

    uhd_bins = [-float("inf"), -30, float("inf")]
    uhd_colors = {
        "UHD PWR < -30 dBm": "#0D9488",
        "UHD PWR ≥ -30 dBm": "#EA580C",
    }
    uhd_labels = list(uhd_colors.keys())
    plot_df["uhd_bin"] = pd.cut(plot_df["uhd_avg"], bins=uhd_bins, labels=uhd_labels)
    # display(plot_df[['lat_bin','lon_bin','uhd_avg','uhd_bin']])

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for band_name in band_list:
        band_df = plot_df[plot_df["Band"] == band_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max-x_min) / bin_width)

            for uhd_label, group in band_df.groupby(by="uhd_bin", observed=True):
                color = uhd_colors[uhd_label]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)

                counts, bin_edges = np.histogram(data, bins=bin_size, density=False)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                pdf = counts / counts.sum()

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
                    customdata = np.stack((counts, sinr_means, cqi_means), axis=-1)
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
                else:
                    bin_indices = np.digitize(group[metric].values, bin_edges) - 1
                    tput_means = []

                    for bin_i in range(len(bin_edges) - 1):
                        in_bin = (bin_indices == bin_i)
                        if np.any(in_bin):
                            tput_means.append(group.loc[in_bin, "DL_Tput"].mean())
                        else:
                            tput_means.append(np.nan)

                    customdata = np.stack((counts, tput_means), axis=-1)

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"PDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=pdf,
                        mode="lines+markers",
                        name=f"{band_name} | {uhd_label}",
                        legendgroup=f"{uhd_label}",
                        line=dict(color=color, width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        ),
                        visible=(band_name == "n28"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                dtick=bin_width,
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="Probability Density Function (PDF)",
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )

    # Band 드롭 다운
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_uhd_pdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")
def dist_kpis_by_uhd_each_band_cdf(df, out_dir, rb_min, grid_size):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("RSRP", "RSRP [dBm]", [-120, -50], 5),
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120], 10),
        ("SINR_SSB", "SINR [dB]", [-5, 45], 3),
        ("RSRQ", "RSRQ [dB]", [-20, -10], 0.5),
        ("CQI", "CQI Index", [-0.1, 15.1], 1),
        ("RI", "Rank Indicator", [0.9, 2.1], 0.05),
    ]

    route_list = [
        "Namsan",
        "Huam345-5",
        "Huam415-1",
    ]

    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min].copy()
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()
    plot_df = _common.assign_uhd_pwr_raw(plot_df, grid_size=grid_size)

    bins = [-float("inf"), -30, float("inf")]
    uhd_colors = {
        "UHD PWR < -30 dBm": "#0D9488",
        "UHD PWR ≥ -30 dBm": "#EA580C",
    }
    uhd_labels = list(uhd_colors.keys())
    plot_df["uhd_bin"] = pd.cut(plot_df["uhd_avg"], bins=bins, labels=uhd_labels)
    # display(plot_df[['lat_bin','lon_bin','uhd_avg','uhd_bin']])

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=VERTICAL_SPACING,
    )

    for band_name in band_list:
        band_df = plot_df[plot_df["Band"] == band_name]

        for i, (metric, x_title, x_range, bin_width) in enumerate(metrics, start=1):
            x_min, x_max = x_range
            bin_size = int((x_max-x_min) / bin_width)

            for uhd_label, group in band_df.groupby(by="uhd_bin", observed=True):
                color = uhd_colors[uhd_label]
                if len(group) < 5:
                    continue

                data = group[metric].dropna().values
                total_count = len(data)
                counts, bin_edges = np.histogram(data, bins=bin_size, density=True)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                raw_counts, _ = np.histogram(data, bins=bin_edges, density=False)

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
                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"SINR: %{{customdata[1]:.1f}}<br>"
                        f"CQI: %{{customdata[2]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )
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

                    hovertemplate = (
                        f"{metric}: %{{x:.1f}}<br>"
                        f"DL Tput: %{{customdata[1]:.1f}}<br>"
                        f"CDF: %{{y:.2f}}<br>"
                        f"Count: %{{customdata[0]}} / {total_count}<extra></extra>"
                    )

                # CDF
                fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=cdf,
                        mode="lines+markers",
                        name=f"{band_name} | {uhd_label}",
                        legendgroup=f"{uhd_label}",
                        line=dict(color=color, width=1.2),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color
                        ),
                        visible=(band_name == "n28"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1
                )

            fig.update_xaxes(
                title_text=x_title,
                gridcolor="rgba(0,0,0,0.15)",
                row=i, col=1,
            )
            fig.update_yaxes(
                title_text="CDF (Cumulative Distribution Function)",
                gridcolor="rgba(0,0,0,0.15)",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                tickformat=".2f",
                row=i, col=1,
                range=[-0.05, 1.05],
            )

    # Band 드롭 다운
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
        autosize=True,
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
    out_path = os.path.join(out_dir, f"kpis_by_uhd_cdf.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")
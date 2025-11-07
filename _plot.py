import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import _common
import numpy as np
import os

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

def plot_kpis_by_site(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SINR [dB]", [-5, 45]),
        # ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
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

    df = df[df["DL_RB"] > rb_min]
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()

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
                valid = band_df[band_df["route"] == route_name].copy()
                if valid.empty:
                    continue

                bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
                valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)

                stats = valid.groupby("RSRP_bin", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
                stats["RSRP_center"] = stats["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                stats["RSRP_left"] = stats["RSRP_bin"].apply(lambda x: x.left)
                stats["RSRP_range"] = stats["RSRP_bin"].apply(lambda x: f"{x.left:.0f} ~ {x.right:.0f}")
                stats["SE"] = stats["std"] / np.sqrt(stats["count"])
                stats["CI"] = 1.96 * stats["SE"]

                stats["hover_text"] = stats.apply(
                    lambda r: (
                        f"<b>Counts</b>: {int(r['count'])}<br>"
                        f"<b>RSRP</b>: {r['RSRP_range']}<br>"
                        f"<b>{metric.replace('_', ' ')}</b>: {r['mean']:.1f} (±{r['CI']:.2f})<br>"
                    ),
                    axis=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=stats["RSRP_center"],
                        y=stats["mean"],
                        mode="lines+markers",
                        name=f"{band_name} | {route_name} | Avg",
                        legendgroup=f"{route_name}",
                        showlegend=(i == 1),
                        line=dict(color=color, width=1.3),
                        marker=dict(size=5, color=color),
                        text=stats["hover_text"],
                        customdata=stats["RSRP_left"],
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color),
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
    out_path = os.path.join(out_dir, f"kpis_by_site_{rsrp_bin}dB.html")
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)

    js_script = f"""
    <script>
        const RSRP_BIN = {rsrp_bin};
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_click', function(data) {{
            var rsrp_left = data.points[0].customdata;
            var rsrp_right = rsrp_left + RSRP_BIN;

            var base_url = "https://joostone-ahn.github.io/nr-field-analysis/results/dist/kpis_by_site_" + RSRP_BIN + "dB/";
            var file_name = "RSRP_" + rsrp_left.toFixed(0) + "_to_" + rsrp_right.toFixed(0) + ".html";
            var full_url = base_url + file_name;

            console.log("Opening:", full_url);
            window.open(full_url, "_blank");
        }});
    </script>
    """

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(js_script)

    print(f"✅ Saved: {out_path}")

def plot_kpis_by_band(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SINR [dB]", [-5, 45]),
        # ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
    ]

    band_colors = {
        "n28": "#FF4500",
        "n26": "#1E90FF"
    }

    fixed_colors = {
        "n28": "#FF8C00",  # 주황색 계열 (Orange)
        "n26": "#32CD32",  # 초록색 계열 (LimeGreen)
    }

    order = ["n28", "n26"]
    route_list = [
        "All",
        "Namsan",
        "Huam345-5",
        "Huam415-1"
    ]

    df = df[df["DL_RB"] > rb_min]
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()
    fixed_routes = [''] + sorted(fixed_df["route"].unique().tolist())

    for fixed_route in fixed_routes:

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
                    color = band_colors[band] if 'Fixed' not in route_name else fixed_colors[band]
                    valid = group.copy()

                    bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
                    valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)

                    stats = valid.groupby("RSRP_bin", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
                    stats["RSRP_center"] = stats["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                    stats["RSRP_left"] = stats["RSRP_bin"].apply(lambda x: x.left)
                    stats["RSRP_range"] = stats["RSRP_bin"].apply(lambda x: f"{x.left:.0f} ~ {x.right:.0f}")
                    stats["SE"] = stats["std"] / np.sqrt(stats["count"])
                    stats["CI"] = 1.96 * stats["SE"]

                    stats["hover_text"] = stats.apply(
                        lambda r: (
                            f"<b>Counts</b>: {int(r['count'])}<br>"
                            f"<b>RSRP</b>: {r['RSRP_range']}<br>"
                            f"<b>{metric.replace('_', ' ')}</b>: {r['mean']:.1f} (±{r['CI']:.2f})<br>"
                        ),
                        axis=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=stats["RSRP_center"],
                            y=stats["mean"],
                            mode="lines+markers",
                            name=f"{route_name} | {band} | Avg",
                            legendgroup=f"{band}",
                            showlegend=(i == 1),
                            line=dict(color=color, width=1.3),
                            marker=dict(size=5, color=color),
                            text=stats["hover_text"],
                            customdata=stats["RSRP_left"],
                            hovertemplate="%{text}<extra></extra>",
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            )
                        ),
                        row=i, col=1
                    )

                    band_fixed = fixed_df[
                        (fixed_df["Band"] == band) &
                        (fixed_df["route"] == fixed_route)
                        ]

                    fig.add_trace(
                        go.Scatter(
                            x=band_fixed["RSRP"],
                            y=band_fixed[metric],
                            mode="markers",
                            name=f"{route_name} | {band} | {fixed_route}",
                            legendgroup=f"{band}_fixed",
                            showlegend=(i == 1),
                            marker=dict(size=5, color=color, opacity=0.2),
                            # hovertemplate=(
                            #     "<b>%{customdata[0]}</b><br>"
                            #     "RSRP: %{x:.1f} dBm<br>"
                            #     f"{metric}: %{{y:.2f}}<extra></extra>"
                            # ),
                            # hoverlabel=dict(
                            #     font=dict(size=11, color="white"),
                            #     bgcolor=color
                            # ),
                            # customdata=np.stack([band_fixed["test_no"]], axis=-1),
                            hoverinfo="skip",
                        ),
                        row=i, col=1
                    )

                    mean_rsrp = band_fixed["RSRP"].mean()
                    n_rsrp = band_fixed["RSRP"].count()
                    # std_rsrp = band_fixed["RSRP"].std()
                    # ci_rsrp = 1.96 * std_rsrp / np.sqrt(n_rsrp)

                    mean_metric = band_fixed[metric].mean()
                    # n_metric = band_fixed[metric].count()
                    # std_metric = band_fixed[metric].std()
                    # ci_metric = 1.96 * std_metric / np.sqrt(n_metric)

                    fig.add_trace(
                        go.Scatter(
                            x=[mean_rsrp],
                            y=[mean_metric],
                            mode="markers+text",
                            name=f"{route_name} | {band} | {fixed_route} | mean+ci",
                            legendgroup=f"{band}_fixed",
                            showlegend=False,
                            marker=dict(
                                size=15,
                                color=color,
                                opacity=1,
                                symbol="square",
                                line=dict(width=1.5, color="white")
                            ),
                            hovertemplate=(
                                f"<b>Samples:</b> {n_rsrp}<br>"
                                f"<b>RSRP:</b> {mean_rsrp:.1f}<br>"
                                f"<b>{metric.replace('_', ' ')}:</b> {mean_metric:.1f} <br><extra></extra>"
                            ),
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
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

        if fixed_route == '':
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"kpis_by_band_{rsrp_bin}dB.html")
            fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
        else:
            if rsrp_bin == 5:
                os.makedirs(os.path.join("results", "_fixed-point"), exist_ok=True)
                out_path = os.path.join('results', "_fixed-point", f"{fixed_route}.html")
                fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
            else:
                continue

        js_script = f"""
        <script>
            const RSRP_BIN = {rsrp_bin};
            var plot = document.getElementsByClassName('plotly-graph-div')[0];
            plot.on('plotly_click', function(data) {{
                var rsrp_left = data.points[0].customdata;
                var rsrp_right = rsrp_left + RSRP_BIN;

                var base_url = "https://joostone-ahn.github.io/nr-field-analysis/results/dist/kpis_by_band_" + RSRP_BIN + "dB/";
                var file_name = "RSRP_" + rsrp_left.toFixed(0) + "_to_" + rsrp_right.toFixed(0) + ".html";
                var full_url = base_url + file_name;

                console.log("Opening:", full_url);
                window.open(full_url, "_blank");
            }});
        </script>
        """

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(js_script)

        print(f"✅ Saved: {out_path}")

def plot_kpis_by_uhd(df, out_dir, rb_min, rsrp_bin, grid_size):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SINR [dB]", [-5, 45]),
        # ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
    ]

    route_list = [
        "Namsan",
        "Huam345-5",
        "Huam415-1",
    ]

    band_list = ["n28", "n26"]

    df = df[df["DL_RB"] > rb_min]
    df = df[(df["RSRP"] <= RSRP_HIGH) & (df["RSRP"] >= RSRP_LOW)]

    fixed_df, non_fixed_df = _common.separate_fixed_point(df)
    plot_df = non_fixed_df[non_fixed_df['route'].isin(route_list)].copy()
    plot_df = _common.assign_uhd_pwr_raw(plot_df, grid_size=grid_size)

    bins = [-float("inf"), -40, -30, float("inf")]
    uhd_colors = {
        "UHD < -40": "#32CD32",  # Green
        "-40 ≤ UHD < -30": "#FFD700",  # Yellow
        "UHD ≥ -30": "#FF4500",  # Red
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

        for i, (metric, y_title, y_range) in enumerate(metrics, start=1):
            for uhd_label, group in band_df.groupby(by="uhd_bin", observed=True):
                color = uhd_colors[uhd_label]
                if group.empty:
                    continue
                valid = group.copy()
                bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
                valid["RSRP_bin"] = pd.cut(valid["RSRP"], bins=bins)

                stats = valid.groupby("RSRP_bin", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
                stats["RSRP_center"] = stats["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
                stats["RSRP_left"] = stats["RSRP_bin"].apply(lambda x: x.left)
                stats["RSRP_range"] = stats["RSRP_bin"].apply(lambda x: f"{x.left:.0f} ~ {x.right:.0f}")
                stats["SE"] = stats["std"] / np.sqrt(stats["count"])
                stats["CI"] = 1.96 * stats["SE"]

                uhd_stats = valid.groupby("RSRP_bin", observed=True)["uhd_avg"].agg(["mean", "std", "count"]).reset_index()
                uhd_stats["SE"] = uhd_stats["std"] / np.sqrt(uhd_stats["count"])
                uhd_stats["CI"] = 1.96 * uhd_stats["SE"]

                stats = stats.merge(uhd_stats[["RSRP_bin", "mean", "CI"]], on="RSRP_bin", suffixes=("", "_uhd"))
                stats["hover_text"] = stats.apply(
                    lambda r: (
                        f"<b>Counts</b>: {int(r['count'])}<br>"
                        f"<b>RSRP</b>: {r['RSRP_range']}<br>"
                        f"<b>{metric.replace('_', ' ')}</b>: {r['mean']:.1f} (±{r['CI']:.2f})<br>"
                        f"<b>UHD PWR</b>: {r['mean_uhd']:.1f} (±{r['CI_uhd']:.2f})"
                    ),
                    axis=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=stats["RSRP_center"],
                        y=stats["mean"],
                        mode="lines+markers",
                        name=f"{band_name} | {uhd_label} | Avg",
                        legendgroup=f"{uhd_label}",
                        showlegend=(i == 1),
                        line=dict(color=color, width=1.2),
                        text=stats["hover_text"],
                        customdata=stats["RSRP_left"],
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            font=dict(size=11, color="white"),
                            bgcolor=color,
                        )
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
    out_path = os.path.join(out_dir, f"kpis_by_uhd_{rsrp_bin}dB.html")
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)

    js_script = f"""
    <script>
        const RSRP_BIN = {rsrp_bin};
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_click', function(data) {{
            var rsrp_left = data.points[0].customdata;
            var rsrp_right = rsrp_left + RSRP_BIN;

            var base_url = "https://joostone-ahn.github.io/nr-field-analysis/results/dist/kpis_by_site_" + RSRP_BIN + "dB/";
            var file_name = "RSRP_" + rsrp_left.toFixed(0) + "_to_" + rsrp_right.toFixed(0) + ".html";
            var full_url = base_url + file_name;

            console.log("Opening:", full_url);
            window.open(full_url, "_blank");
        }});
    </script>
    """

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(js_script)

    print(f"✅ Saved: {out_path}")

def plot_kpis_each_test(df, df_grid, out_dir, grid_size):

    lat_factor, lon_factor = 111320, 88000
    df["lat_bin"] = (df["Lat"] * lat_factor // grid_size).astype(int)
    df["lon_bin"] = (df["Lon"] * lon_factor // grid_size).astype(int)
    df = pd.merge(df, df_grid, on=["lat_bin", "lon_bin"], how="left")
    df[f"loc_id"] = df[f"loc_id"].astype("Int64")

    metrics = [
        "RSRP", "RSRQ",
        "SINR_SSB", "SINR_TRS",
        "DL_RB",
        "DL_Tput",
        "DL_Tput_per_RB",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
    ]

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
        save_dir = os.path.join(out_dir, date, route)
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
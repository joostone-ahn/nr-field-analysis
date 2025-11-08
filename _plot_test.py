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

def split_band_df(df):
    common_cols = [c for c in df.columns if not any(s in c for s in ["_n26", "_n28", "_diff"])]

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

    df_n26 = extract_band(df, "n26")
    df_n28 = extract_band(df, "n28")

    return pd.concat([df_n26, df_n28], axis=0)

def plot_kpis_by_uhd(df, out_dir, rb_min, sample_min, rsrp_bin, grid_size):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    HORIZONTAL_SPACING = 0.015
    TOP_MARGIN = 70
    LEGEND_Y = 1.02
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -120
    RSRP_HIGH = -50

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SSB SINR [dB]", [-5, 45]),
        ("SINR_TRS", "TRS SINR [dB]", [5, 40]),
        ("RSRQ", "RSRQ [dB]", [-20, -10]),
        ("CQI", "CQI Index", [-0.1, 15.1]),
        ("RI", "Rank Indicator", [0.9, 2.1]),
    ]

    route_list = [
        # "All",
        "Namsan",
        "Huam345-5",
        # "Huam415-1"
    ]

    band_list = ['n28','n26']

    plot_df = split_band_df(df)
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    rsrp_bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)
    plot_df["RSRP_bin"] = pd.cut(plot_df["RSRP"], bins=rsrp_bins)

    # uhd_bins = [-float("inf"), -30, float("inf")]
    # uhd_colors = {
    #     f"UHD<{uhd_bins[1]}": "#0D9488",  # 청록
    #     f"UHD≥{uhd_bins[1]}": "#EA580C",  # 주황
    # }
    uhd_bins = [-float("inf"), -30, -27, float("inf")]
    uhd_colors = {
        f"UHD<{uhd_bins[1]}": "#16A34A",
        f"{uhd_bins[1]}≤UHD<{uhd_bins[2]}": "#FFB347",
        f"UHD≥{uhd_bins[2]}": "#DC2626",
    }
    uhd_labels = list(uhd_colors.keys())
    plot_df["uhd_bin"] = pd.cut(plot_df["uhd_avg"], bins=uhd_bins, labels=uhd_labels)
    # display(plot_df[['lat_bin','lon_bin','uhd_avg','uhd_bin']])

    fig = make_subplots(
        rows=len(metrics),
        cols=2,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=VERTICAL_SPACING,
        horizontal_spacing=HORIZONTAL_SPACING,
    )

    for route_name in route_list:
        route_df = plot_df if route_name == "All" else plot_df[plot_df["route"] == route_name]

        for row, (metric, y_title, y_range) in enumerate(metrics, start=1):

            for col, band in enumerate(band_list, start=1):
                band_df = route_df[route_df["Band"] == band]

                for uhd_label, uhd_df in band_df.groupby(by="uhd_bin", observed=True):
                    color = uhd_colors[uhd_label]

                    fig.add_trace(
                        go.Scatter(
                            x=uhd_df["RSRP"],
                            y=uhd_df[metric],
                            mode="markers",
                            name=f"{route_name} | {uhd_label}",
                            legendgroup=uhd_label,
                            showlegend=(row == 1 and col == 1),
                            marker=dict(
                                size=8,
                                color=color,
                                opacity=1,
                                symbol="square",
                                line=dict(width=1, color="white")
                            ),
                            customdata=uhd_df[["loc_id"]],
                            hovertemplate=(
                                "<b>loc_id: </b>%{customdata[0]}<br>"
                                "<b>RSRP: </b>%{x:.1f}<br>"
                                f"<b>{metric.replace('_', ' ')}: </b>%{{y:.1f}}<extra></extra>"
                            ),
                            hoverlabel=dict(
                                font=dict(size=11, color="white"),
                                bgcolor=color
                            ),
                        ),
                        row=row, col=col
                    )

            for col, band in enumerate(band_list, start=1):
                fig.update_xaxes(
                    title=f"{band} RSRP [dBm]",
                    autorange="reversed",
                    dtick=5,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=row, col=col,
                    matches='x',
                )
                fig.update_yaxes(
                    title=y_title if col == 1 else "",
                    # range=y_range,
                    gridcolor="rgba(0,0,0,0.15)",
                    row=row, col=col,
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
    initial_visible = [("Namsan" in trace.name) for trace in fig.data]
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
    out_path = os.path.join(out_dir, f"kpis_by_uhd.html")
    fig.write_html(out_path)
    print(f"✅ Saved: {out_path}")

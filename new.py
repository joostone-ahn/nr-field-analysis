from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os


def plot_kpis_pdf(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 600
    VERTICAL_SPACING = 0.035
    TOP_MARGIN = 70
    LEGEND_Y = 1.03
    LEGEND_FONT_SIZE = 13
    RSRP_LOW = -115
    RSRP_HIGH = -65

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SSB SINR [dB]", [-10, 40]),
        ("SINR_TRS", "TRS SINR [dB]", [-10, 40]),
        ("RSRQ", "RSRQ [dB]", [-15, -5]),
    ]

    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]
    route_list = ["All", "Namsan", "Huam345-5", "Huam415-1"]

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)

    # ======= MAIN LOOP =======
    for b_idx, b in enumerate(bins[:-1]):
        rsrp_min, rsrp_max = b, b + rsrp_bin
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=VERTICAL_SPACING,
        )

        for route_name in route_list:
            route_df = plot_df if route_name == "All" else plot_df[plot_df["route"] == route_name]
            if route_df.empty:
                continue

            for i, (metric, y_title, y_range) in enumerate(metrics, start=1):
                x_min, x_max = None, None

                for band in order:
                    group = route_df[route_df["Band"] == band]
                    if len(group) < 5:
                        continue

                    kde = gaussian_kde(group[metric].dropna())
                    x_vals = np.linspace(group[metric].min(), group[metric].max(), 400)
                    y_vals = kde(x_vals)

                    # x 범위 추적
                    if x_min is None or group[metric].min() < x_min:
                        x_min = group[metric].min()
                    if x_max is None or group[metric].max() > x_max:
                        x_max = group[metric].max()

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=f"{route_name} | {band}",
                            legendgroup=f"{band}",  # 🔥 동일 band trace 동기화
                            line=dict(color=band_colors[band], width=2),
                            hoverinfo="skip",
                            visible=(route_name == "All"),
                            showlegend=(i == 1),
                        ),
                        row=i, col=1,
                    )

                # X축 tick 자동 계산 (15개, 소수점 1자리)
                if x_min is not None and x_max is not None:
                    target_ticks = 15
                    dtick = round((x_max - x_min) / target_ticks, 1)
                    dtick = max(dtick, 0.1)
                else:
                    dtick = None

                fig.update_xaxes(
                    title_text=y_title,
                    gridcolor="rgba(0,0,0,0.15)",
                    dtick=dtick,
                    row=i, col=1,
                )
                fig.update_yaxes(
                    title_text="Density",
                    gridcolor="rgba(0,0,0,0.15)",
                    row=i, col=1,
                )

        # ---- DROPDOWN MENU (route 선택) ----
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

        # ---- LAYOUT ----
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

        # ---- 파일 저장 ----
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pdf_kpis_{rsrp_min}_{rsrp_max}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")
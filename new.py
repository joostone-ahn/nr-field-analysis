from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

def plot_kpis_pdf(df, out_dir, rb_min, rsrp_bin):
    SUBPLOT_HEIGHT = 500
    VERTICAL_SPACING = 0.08
    TOP_MARGIN = 70
    LEGEND_FONT_SIZE = 12
    RSRP_LOW, RSRP_HIGH = -115, -65

    plot_df = df[df["DL_RB"] > rb_min].copy()
    plot_df = plot_df[(plot_df["RSRP"] <= RSRP_HIGH) & (plot_df["RSRP"] >= RSRP_LOW)]

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", [0, 120]),
        ("SINR_SSB", "SSB SINR [dB]", [-10, 40]),
        ("SINR_TRS", "TRS SINR [dB]", [-10, 40]),
        ("RSRQ", "RSRQ [dB]", [-15, -5]),
    ]
    band_colors = {"n28": "#FF4500", "n26": "#1E90FF"}
    order = ["n28", "n26"]

    bins = np.arange(RSRP_LOW, RSRP_HIGH + 1, rsrp_bin)

    # 출력 폴더
    pdf_dir = os.path.join(out_dir, f"pdf_kpis_{rsrp_bin}dB")
    os.makedirs(pdf_dir, exist_ok=True)

    for b_idx, b in enumerate(bins[:-1]):
        rsrp_min, rsrp_max = b, b + rsrp_bin
        subset = plot_df[(plot_df["RSRP"] >= rsrp_min) & (plot_df["RSRP"] < rsrp_max)]
        if subset.empty:
            continue

        # subplot 구조 생성
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=VERTICAL_SPACING,
        )

        for i, (metric, y_title, y_range) in enumerate(metrics, start=1):
            for band in order:
                group = subset[subset["Band"] == band]
                if len(group) < 5:
                    continue

                kde = gaussian_kde(group[metric].dropna())
                x_vals = np.linspace(group[metric].min(), group[metric].max(), 200)
                y_vals = kde(x_vals)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        name=f"{band}",
                        line=dict(color=band_colors[band], width=2),
                        hovertemplate=f"<b>{band}</b><br>{metric}: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                    ),
                    row=i, col=1
                )

            fig.update_yaxes(
                title_text="Density",
                gridcolor="rgba(0,0,0,0.1)",
                row=i, col=1
            )
            fig.update_xaxes(
                title_text=y_title,
                gridcolor="rgba(0,0,0,0.1)",
                row=i, col=1
            )

        fig.update_layout(
            title=f"PDF Distributions for RSRP {rsrp_min} ~ {rsrp_max} dBm",
            height=SUBPLOT_HEIGHT * len(metrics),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=LEGEND_FONT_SIZE)
            ),
            margin=dict(l=60, r=60, t=TOP_MARGIN, b=60),
        )

        out_path = os.path.join(pdf_dir, f"pdf_kpis_{rsrp_min}to{rsrp_max}dB.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

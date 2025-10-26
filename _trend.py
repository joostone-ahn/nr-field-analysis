import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import _common

metrics = [
    "RSRP", "RSRQ",
    "SINR", "SINR_TRS",
    "DL_RB", "DL_Tput",
    "CQI", "RI", "DL_MCS",
    "DL_BLER", "UL_BLER",
]

def kpi_by_test(df, out_dir):

    df_mean = (
        df.groupby(["test_no", "Band"])[metrics]
          .mean()
          .reset_index()
    )
    
    fig = plt.figure(figsize=(16, 4 * len(metrics)))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
    
        # Band별 필터링
        df_n26 = df_mean[df_mean["Band"] == "n26"]
        df_n28 = df_mean[df_mean["Band"] == "n28"]
    
        # --- 그래프 플로팅 ---
        plt.plot(
            df_n26["test_no"],
            df_n26[metric],
            marker="o",
            label=f"{metric}_n26",
            color="blue",
            alpha=0.7,
            linewidth=1.5,
        )
        plt.plot(
            df_n28["test_no"],
            df_n28[metric],
            marker="s",
            label=f"{metric}_n28",
            color="red",
            alpha=0.7,
            linewidth=1.5,
        )
    
        # --- 그래프 스타일 ---
        # plt.title(f"{metric} by Test (n26 vs n28)", fontsize=12, pad=5)
        plt.xlabel("Test No", fontsize=12)
        plt.xticks(rotation=90)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", fontsize=9)
        
    fig.suptitle(f"KPI by test (n26 vs n28)", fontsize=14, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "kpi_by_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    # plt.show()
    print(f"Saved: {out_path}")

def kpi_each_test(df, out_dir, grid_size, rb_min, sample_min):
    lat_factor, lon_factor = 111320, 88000
    df[f"lat_bin_{grid_size}m"] = (df["Lat"] * lat_factor // grid_size).astype(int)
    df[f"lon_bin_{grid_size}m"] = (df["Lon"] * lon_factor // grid_size).astype(int)

    df_grid = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
    df_grid = df_grid.rename(columns={
        "lat_bin": f"lat_bin_{grid_size}m",
        "lon_bin": f"lon_bin_{grid_size}m",
        "loc_id": f"loc_id_{grid_size}m"
    })
    df = df.merge(
        df_grid[[f"lat_bin_{grid_size}m", f"lon_bin_{grid_size}m", f"loc_id_{grid_size}m"]],
        on=[f"lat_bin_{grid_size}m", f"lon_bin_{grid_size}m"],
        how="left"
    )

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
                df_sub[["TIME", f"loc_id_{grid_size}m"]],
                on="TIME",
                how="left"
            )

            for m in metrics:
                df_pivot[f"{m}_delta"] = df_pivot[f"{m}_n28"] - df_pivot[f"{m}_n26"]

            hover_texts = []
            for _, row in df_pivot.iterrows():
                time_val = row["TIME"].strftime("%H:%M:%S")
                loc_id_val = int(row[f"loc_id_{grid_size}m"])
                f"<b>loc_id_{grid_size}m:</b> {loc_id_val} "

                lines = [
                    f"<b>time:</b> {time_val}<br>"
                    f"<b>loc_id_{grid_size}m:</b> {loc_id_val}<br>",
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
                y=df_pivot[f"loc_id_{grid_size}m"],
                mode="lines+markers",
                line=dict(color="gray", width=0.8),
                marker=dict(size=3),
                name=f"loc_id_{grid_size}m",
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
                title_text=f"loc_id_{grid_size}m",
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

        date, route = target_no.split("_")[0], target_no.split("_")[2]
        save_dir = os.path.join(out_dir, f"plot_kpi_each_test", date, route)
        os.makedirs(save_dir, exist_ok=True)
        out_path_html = os.path.join(save_dir, f"{target_no}.html")

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

            ymin = df_pivot[[ "n26", "n28" ]].min().min()
            ymax = df_pivot[[ "n26", "n28" ]].max().max()
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
        save_dir = os.path.join(out_dir, f"plot_RB_each_test")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{date}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        # plt.show()
        print(f"Saved: {out_path}")

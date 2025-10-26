import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import _common

metrics = [
    "RSRP", "RSRQ",
    "SINR", "SINR_TRS",
    "CQI", "RI", "DL_MCS",
    "DL_BLER", "UL_BLER",
    "DL_RB", "DL_Tput",
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

def kpi_each_test(df, out_dir, rb_min):
    lat_factor, lon_factor = 111320, 88000
    grid_list = [5, 25]

    for grid_size in grid_list:
        df[f"lat_bin_{grid_size}m"] = (df["Lat"] * lat_factor // grid_size).astype(int)
        df[f"lon_bin_{grid_size}m"] = (df["Lon"] * lon_factor // grid_size).astype(int)
        df_grid = _common.grid_kpi(df=df, grid_size=grid_size)
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
        df = df.dropna(subset=[f"loc_id_{grid_size}m"])
        df.drop(columns=[f"lat_bin_{grid_size}m", f"lon_bin_{grid_size}m"], inplace=True)
        df[f"loc_id_{grid_size}m"] = df[f"loc_id_{grid_size}m"].astype(int)

    test_list = sorted(df["test_no"].unique())

    for target_no in test_list:
        df_sub = df[df["test_no"] == target_no]
        fig, axes = plt.subplots(len(metrics) + 1, 1, figsize=(16, 4 * (len(metrics) + 1)), sharex=False)

        t_min = df_sub["TIME"].min().floor("10s")
        t_max = df_sub["TIME"].max().ceil("10s")
        tick_times_major = pd.date_range(start=t_min, end=t_max, freq="10s")
        tick_times_minor = pd.date_range(start=t_min, end=t_max, freq="1s")

        ax0 = axes[0]
        ax0.plot(
            df_sub["TIME"], df_sub["loc_id_25m"],
            color="black", linewidth=0.8, linestyle="-", alpha=0.7, marker="o", markersize=1, label="loc_id_25m"
        )
        ax0.set_ylabel("loc_id_25m", fontsize=12, color="black")
        ax0.tick_params(axis='y', labelcolor="black")
        ax1 = ax0.twinx()
        ax1.plot(
            df_sub["TIME"], df_sub["loc_id_5m"],
            color="gray", linewidth=0.8, linestyle="--", alpha=0.7, marker="o", markersize=1, label="loc_id_5m"
        )
        ax1.set_ylabel("loc_id_5m", fontsize=12, color="black")
        ax1.tick_params(axis='y', labelcolor="black")
        lines, labels = ax0.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax0.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")

        ax0.set_xticks(tick_times_major)
        ax0.set_xticks(tick_times_minor, minor=True)
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax0.tick_params(axis='x', rotation=90)
        ax0.grid(True, linestyle="--", alpha=0.5)
        ax0.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax0.minorticks_on()
        ax0.set_xlim(t_min, t_max)

        for i, metric in enumerate(metrics, start=1):
            ax = axes[i]
            df_pivot = (
                df_sub.pivot_table(index="TIME", columns="Band", values=metric)
                      .dropna()
                      .reset_index()
            )

            ymin = df_pivot[["n26", "n28"]].min().min()
            ymax = df_pivot[["n26", "n28"]].max().max()
            if metric == "DL_RB":
                ymax = 50
                ymin = rb_min

            ax.plot(df_pivot["TIME"], df_pivot["n26"], label="n26", color="blue", linewidth=0.8, alpha=0.7, marker="o", markersize=1)
            ax.plot(df_pivot["TIME"], df_pivot["n28"], label="n28", color="red", linewidth=0.8, alpha=0.7, marker="o", markersize=1)

            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel(metric, fontsize=12)

            ax.set_xticks(tick_times_major)
            ax.set_xticks(tick_times_minor, minor=True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.tick_params(axis='x', rotation=90)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", alpha=0.3)
            ax.minorticks_on()
            ax.set_xlim(t_min, t_max)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle(f"[{target_no}] KPI trends over time (n26 vs n28)", fontsize=14, y=0.995)

        date = target_no.split("_")[0]
        route = target_no.split("_")[1]
        os.makedirs(os.path.join(out_dir, "kpi_each_test", date, route), exist_ok=True)
        out_path = os.path.join(out_dir, "kpi_each_test", date, route, f"kpi_{target_no}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        # plt.show()
        plt.close(fig)
        print(f"Saved: {out_path}")

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
        os.makedirs(os.path.join(out_dir, "RB_each_test"), exist_ok=True)
        out_path = os.path.join(out_dir, "RB_each_test", f"{date}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        # plt.show()
        print(f"Saved: {out_path}")

import os
import matplotlib.pyplot as plt

metrics = [
    "RSRP",
    "RSRQ",
    "SINR",
    # "SINR_SSB",
    # "SINR_TRS",
    "DL_Tput",
    "DL_RB",
    "DL_BLER",
    "UL_BLER",
]

def kpi_by_test(df, out_dir):

    df_mean = (
        df.groupby(["test_no", "Band"])[metrics]
          .mean()
          .reset_index()
    )
    
    fig = plt.figure(figsize=(14, 4 * len(metrics)))
    
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
    test_list = sorted(df["test_no"].unique())
    for target_no in test_list:
        df_sub = df[df["test_no"] == target_no]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 16), sharex=False)

        for i, metric in enumerate(metrics):
            ymin = df_sub[metric].min()
            ymax = df_sub[metric].max()
            if metric == 'DL_RB':
                ymax = 50
                ymin = rb_min

            df_pivot = (
                df_sub.pivot_table(index="TIME", columns="Band", values=metric)
                      .dropna()
                      .reset_index()
            )
            df_pivot["idx"] = range(len(df_pivot))
            ax = axes[i]

            # n26 / n28 plot
            ax.plot(df_pivot["idx"], df_pivot["n26"], label="n26", color="blue", linewidth=0.8, alpha=0.7)
            ax.plot(df_pivot["idx"], df_pivot["n28"], label="n28", color="red", linewidth=0.8, alpha=0.7)

            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(metric, fontsize=11, pad=5)
            ax.set_xlabel("Time Index")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle(f"[{target_no}] KPI trends over time (n26 vs n28)", fontsize=14, y=0.995)

        date = target_no.split("_")[0]
        os.makedirs(os.path.join(out_dir, "kpi_each_test", date), exist_ok=True)
        out_path = os.path.join(out_dir, "kpi_each_test", date, f"kpi_{target_no}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"Saved: {out_path}")

def rb_each_test(df, out_dir, rb_min):
    metric = "DL_RB"
    test_list = sorted(df["test_no"].unique())
    fig, axes = plt.subplots(len(test_list), 1, figsize=(12, 3.5 * len(test_list)), sharex=False)
    
    if len(test_list) == 1:
        axes = [axes]
    
    for i, target_no in enumerate(test_list):
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
    fig.suptitle(f"DL RB num by Test (n26 vs n28)", fontsize=14, y=0.95)
    out_path = os.path.join(out_dir, "RB_each_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    # plt.show()
    print(f"Saved: {out_path}")

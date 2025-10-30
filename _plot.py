import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import re
import numpy as np
import pandas as pd
import _common

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

def scat_kpi_by_group(df, out_dir, grid_size, rb_min, sample_min, band, groupby):

    if band not in ["n26", "n28"]:
        ValueError("band must be n28 or n26")
    if groupby not in ["uhd_max", "route"]:
        ValueError("groupby must be 'uhd' or 'route'")

    metrics = [
        "SINR_TRS",
        # "DL_Tput",
        "DL_Tput_per_RB",
    ]

    for metric in metrics:
        df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
        df_n26, df_n28 = split_band_df(df_pair)
        plot_df = df_n28.copy() if band == "n28" else df_n26.copy()
        plot_df = plot_df[(plot_df["RSRP"] <= -60) & (plot_df["RSRP"] >= -120)]
        # print(plot_df.info())

        if groupby == "uhd":
            group_name = 'UHD Power'
            color_col = "uhd_max"
            valid_vals = plot_df[color_col].dropna()

            if len(valid_vals) > 0:
                q1, q2, q3 = -40, -35, -30

                def color_by_uhd(v):
                    if pd.isna(v):
                        return "gray", "null"
                    elif v >= q3:
                        return "#FF4500", f"PWR ≥ {q3:.0f}"  # 빨강 (높음)
                    elif v >= q2:
                        return "#FFD700", f"{q2:.0f} ≤ PWR < {q3:.0f}"  # 노랑
                    elif v >= q1:
                        return "#32CD32", f"{q1:.0f} ≤ PWR < {q2:.0f}"  # 초록
                    else:
                        return "#1E90FF", f"PWR < {q1:.0f}"  # 파랑 (낮음)

                plot_df[["color", "color_label"]] = plot_df[color_col].apply(lambda v: pd.Series(color_by_uhd(v)))
                order = [
                    f"PWR ≥ {q3:.0f}",
                    f"{q2:.0f} ≤ PWR < {q3:.0f}",
                    f"{q1:.0f} ≤ PWR < {q2:.0f}",
                    f"PWR < {q1:.0f}",
                    "null",
                ]
        elif groupby == "route":
            group_name = "Field Route"
            color_col = "route"
            route_colors = {
                "Namsan": "#FF4500",
                "Huam345-5": "#FFD700",
                "Huam415-1": "#32CD32",
            }

            def color_by_route(v):
                color = route_colors.get(v, "gray")
                label = v if v in route_colors else "Unknown"
                return color, label
            plot_df[["color", "color_label"]] = plot_df[color_col].apply(lambda v: pd.Series(color_by_route(v)))
            order = ["Namsan", "Huam345-5", "Huam415-1", "Unknown"]

        plot_df["color_label"] = pd.Categorical(plot_df["color_label"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("color_label")

        def make_hover_text(row):
            def ci95(std, n):
                if pd.isna(std) or pd.isna(n) or n <= 1:
                    return None
                return 1.96 * std / np.sqrt(n)

            ci = {
                "DL_Tput": ci95(row.get("DL_Tput_std"), row.get("count")),
                "DL_RB": ci95(row.get("DL_RB_std"), row.get("count")),
                "DL_Tput_per_RB": ci95(row.get("DL_Tput_per_RB_std"), row.get("count")),
                "RSRP": ci95(row.get("RSRP_std"), row.get("count")),
                "SINR_TRS": ci95(row.get("SINR_TRS_std"), row.get("count")),
            }

            def fmt(val, ci_val):
                if pd.isna(val):
                    return "null"
                if ci_val is None:
                    return f"{val:.1f}"
                return f"{val:.1f} <span style='color:#777;'>(±{ci_val:.2f})</span>"

            lines = [
                "────────────────────────",
                f"<b>UHD_PWR</b> : {row['uhd_max']:.1f}" if not pd.isna(row['uhd_max']) else "<b>UHD_PWR</b>: null",
                f"<b>route / loc_id</b> : {row['route']} / {row['loc_id']}",
                "────────────────────────",
                f"<b>samples</b> : {row['count']}",
                f"<b>RSRP</b> : {fmt(row['RSRP'], ci['RSRP'])}",
                f"<b>SINR_TRS</b> : {fmt(row['SINR_TRS'], ci['SINR_TRS'])}",
                f"<b>DL_Tput</b> : {fmt(row['DL_Tput'], ci['DL_Tput'])}",
                f"<b>DL_RB</b> : {fmt(row['DL_RB'], ci['DL_RB'])}",
                f"<b>DL_Tput_per_RB</b> : {fmt(row['DL_Tput_per_RB'], ci['DL_Tput_per_RB'])}",
                "────────────────────────",
            ]
            return "<br>".join(lines)

        plot_df["hover_text"] = plot_df.apply(make_hover_text, axis=1)

        fig = go.Figure()

        for label in order:
            group = plot_df[plot_df["color_label"] == label]
            if group.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group["RSRP"],
                    y=group[metric],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    marker=dict(size=7, color=group["color"].iloc[0]),
                    text=group["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        # 다항식 (개별 커브)
        if metric == 'SINR_TRS' and groupby == 'route':
            x_range = np.linspace(plot_df["RSRP"].min(), plot_df["RSRP"].max(), 200)

            for label, group in plot_df.groupby("color_label", observed=True):
                color = group["color"].iloc[0]
                valid = group[["RSRP", metric]].dropna().sort_values("RSRP")

                X = valid["RSRP"].values.reshape(-1, 1)
                y = valid[metric].values

                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode="lines",
                        line=dict(color=color, width=2, dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=label
                    )
                )
        else:
            # 다항식 (단일 커브)
            valid = plot_df.dropna(subset=["RSRP", metric])

            X = valid["RSRP"].values.reshape(-1, 1)
            y = valid[metric].values

            poly = PolynomialFeatures(degree=3)
            x_poly = poly.fit_transform(X)
            model = LinearRegression().fit(x_poly, y)

            x_range = np.linspace(valid["RSRP"].min(), valid["RSRP"].max(), 200)
            x_pred_poly = poly.transform(x_range.reshape(-1, 1))
            y_pred = model.predict(x_pred_poly)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    name="3rd-Order Poly",
                    line=dict(color="gray", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=True,
                    legendgroup="trend",
                )
            )

        # # 3차, CI 밴드 포함
        # valid = plot_df.dropna(subset=["RSRP", metric])
        # valid["RSRP_bin"] = valid["RSRP"].round(1)
        #
        # agg = valid.groupby("RSRP_bin").agg(
        #     y_mean=(metric, "mean"),
        #     y_std=(f"{metric}_std", lambda x: np.sqrt(np.mean(x ** 2))),  # RMS std
        #     n=("RSRP", "count")
        # ).reset_index()
        #
        # X = agg["RSRP_bin"].values.reshape(-1, 1)
        # y = agg["y_mean"].values
        # y_std = agg["y_std"].values
        #
        # poly = PolynomialFeatures(degree=3)
        # X_poly = poly.fit_transform(X)
        # model = LinearRegression().fit(X_poly, y)
        #
        # x_range = np.linspace(X.min(), X.max(), 200)
        # X_pred = poly.transform(x_range.reshape(-1, 1))
        # y_pred = model.predict(X_pred)
        #
        # residuals = y - model.predict(X_poly)
        # residual_var = np.var(residuals)
        #
        # meas_var = np.interp(x_range, X.flatten(), y_std ** 2)
        # total_std = np.sqrt(residual_var + meas_var)
        #
        # ci_upper = y_pred + 1.96 * total_std
        # ci_lower = y_pred - 1.96 * total_std
        #
        # fig.add_trace(go.Scatter(
        #     x=x_range, y=y_pred,
        #     mode="lines",
        #     name="3rd-Order Poly",
        #     line=dict(color="gray", dash="dot"),
        # ))
        #
        # fig.add_trace(go.Scatter(
        #     x=np.concatenate([x_range, x_range[::-1]]),
        #     y=np.concatenate([ci_upper, ci_lower[::-1]]),
        #     fill="toself",
        #     fillcolor="rgba(128,128,128,0.25)",
        #     line=dict(color="rgba(255,255,255,0)"),
        #     name="95 % CI Band",
        # ))

        # # RSRP bin 평균
        # bins = np.arange(plot_df["RSRP"].min(), plot_df["RSRP"].max() + 1, 1)
        # plot_df["RSRP_bin"] = pd.cut(plot_df["RSRP"], bins=bins, right=False)
        # plot_df["RSRP_bin_center"] = plot_df["RSRP_bin"].apply(lambda x: (x.left + x.right) / 2)
        #
        # avg_df = (
        #     plot_df.groupby("RSRP_bin_center", observed=True)[metric]
        #     .mean()
        #     .reset_index()
        #     .sort_values("RSRP_bin_center")
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=avg_df["RSRP_bin_center"],
        #         y=avg_df[metric],
        #         mode='lines',
        #         name="Average",
        #         line=dict(color="gray", width=2, dash="dot"),
        #         hoverinfo="skip",
        #         showlegend=True,
        #         legendgroup="trend"
        #     )
        # )

        ## Linear Regression (단일 커브)
        # valid = plot_df.dropna(subset=["RSRP", "DL_Tput"])

        # X = valid["RSRP"].values.reshape(-1, 1)
        # y = valid["DL_Tput"].values

        # model = LinearRegression().fit(X, y)
        # x_range = np.linspace(valid["RSRP"].min(), valid["RSRP"].max(), 200)
        # y_pred = model.predict(x_range.reshape(-1, 1))

        # fig.add_trace(
        #     go.Scatter(
        #         x=x_range,
        #         y=y_pred,
        #         mode="lines",
        #         name="Linear Regression",
        #         line=dict(color="black", width=2, dash="dot"),
        #         hoverinfo="skip",
        #         showlegend=True,
        #         legendgroup="trend",
        #     )
        # )

        ## Linear Regression (개별 커브)
        # valid_df = plot_df.dropna(subset=["RSRP", metric])
        # x_range = np.linspace(valid_df["RSRP"].min(), valid_df["RSRP"].max(), 200).reshape(-1, 1)
        #
        # for label, group in valid_df.groupby("color_label", observed=True):
        #     if len(group) < 2:
        #         continue  # 데이터 너무 적으면 스킵
        #
        #     X = group["RSRP"].values.reshape(-1, 1)
        #     y = group[metric].values
        #
        #     model = LinearRegression().fit(X, y)
        #     y_pred = model.predict(x_range)
        #
        #     color = group["color"].iloc[0]
        #
        #     fig.add_trace(
        #         go.Scatter(
        #             x=x_range.flatten(),
        #             y=y_pred,
        #             mode="lines",
        #             name=f"Trend ({label})",
        #             line=dict(color=color, width=1, dash="dot"),
        #             hoverinfo="skip",
        #             showlegend=False,
        #             legendgroup=label,
        #         )
        #     )


        if band == "n28":
            map_url = "https://joostone-ahn.github.io/nr-field-analysis/results/map_mobility/n28_DL_Tput.html"
            title_text = (
                f"{metric.replace("_"," ")} over RSRP group by {group_name} ({band}) "
                f"<a href='{map_url}' target='_blank' style='text-decoration:none; font-size:14px;'> [View Map]</a>"
            )
        else:
            title_text = f"{metric.replace("_", " ")} over RSRP group by {group_name} ({band})"

        fig.update_layout(
            title=title_text,
            template="plotly_white",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="gray",
                font=dict(size=10),
                align="left",
            ),
            legend=dict(
                title=dict(
                    text=f"<span><b>  {group_name}</b></span><br>",
                    font=dict(size=13),
                    side="top"
                ),
                font=dict(size=13),
                itemsizing="constant",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                tracegroupgap=8,
                yanchor="top",
                y=1.0,
                xanchor="left",
            )
        )
        fig.update_xaxes(
            title="RSRP [dBm]",
            autorange="reversed",
            dtick=5,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
            griddash="dot",
        )

        if metric == "SINR_TRS":
            y_title = "SINR TRS [dB]"
        elif metric == "DL_Tput":
            y_title = "DL Throughput [Mbps]"
        elif metric == "DL_Tput_per_RB":
            y_title = "DL Throughput per RB [Mbps]"
        fig.update_yaxes(
            title=y_title,
            # dtick=10,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.15)",
            griddash="dot",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{band}_{metric}_by_{groupby}.html")
        fig.write_html(out_path)
        print(f"✅ Saved: {out_path}")

def kpi_each_test(df, out_dir, grid_size, rb_min, sample_min):
    metrics = [
        "RSRP", "RSRQ",
        "SINR_SSB", "SINR_TRS",
        "DL_RB",
        "DL_Tput",
        "DL_Tput_per_RB",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
    ]

    lat_factor, lon_factor = 111320, 88000
    df[f"lat_bin"] = (df["Lat"] * lat_factor // grid_size).astype(int)
    df[f"lon_bin"] = (df["Lon"] * lon_factor // grid_size).astype(int)

    df_grid = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)
    df_grid = df_grid.rename(columns={
        "lat_bin": f"lat_bin",
        "lon_bin": f"lon_bin",
        "loc_id": f"loc_id"
    })
    df = df.merge(
        df_grid[[f"lat_bin", f"lon_bin", f"loc_id"]],
        on=[f"lat_bin", f"lon_bin"],
        how="left"
    )
    df[f"loc_id"] = df[f"loc_id"].astype("Int64")

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
        save_dir = os.path.join(out_dir, f"plot_kpis_each_test", date, route)
        os.makedirs(save_dir, exist_ok=True)
        out_path_html = os.path.join(save_dir, f"TEST_{test_num}.html")
        pio.write_html(fig, file=out_path_html, include_plotlyjs="cdn", full_html=True)
        print(f"Saved HTML: {out_path_html}")


def plot_kpi(df, grid_size, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, f"plot_{grid_size}m")
    os.makedirs(plot_dir, exist_ok=True)

    df_pair = _common.grid_kpi(df, grid_size=grid_size)

    rsrp_col = "RSRP_n28"
    rsrp_min = int(df_pair[rsrp_col].min())
    rsrp_max = int(df_pair[rsrp_col].max())
    bins = np.arange(rsrp_min, rsrp_max + 2, 1)  # 1 dB step
    df_pair["RSRP_bin"] = pd.cut(df_pair["RSRP_n28"], bins=bins, right=False)

    df_binned = (
        df_pair.groupby("RSRP_bin", observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    df_binned["RSRP_bin_tick"] = df_binned["RSRP_bin"].apply(lambda x: int(x.left))

    style_map = {
        "n26": {"marker": "o", "color": "blue"},
        "n28": {"marker": "s", "color": "red"},
    }

    # 공통 x축 설정 함수
    def apply_common_axis(ax, df, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.7)

        # min/max를 5dB 단위로 맞추기
        x_min = int(df["RSRP_bin_tick"].min())
        x_max = int(df["RSRP_bin_tick"].max())
        x_min_5 = (x_min // 5) * 5
        x_max_5 = (x_max // 5) * 5

        xticks = np.arange(x_min_5, x_max_5 + 1, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45)

    # Rx Quality (Absolute + Difference)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.32)

    # Absoulte
    metrics_group1 = [("RSRP", "RSRP [dBm]"),
                      ("RSRQ", "RSRQ [dB]"),
                      ("SINR", "SINR [dB]")]

    for ax, (metric, ylabel) in zip(axes[0], metrics_group1):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
                style = style_map[source]
                ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                        marker=style["marker"], markersize=3,
                        color=style["color"], label=source,
                        linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad=8)

    # n28 - n26 Difference
    metrics_group_diff = [
        ("RSRP", "RSRP Δ [dB]"),
        ("RSRQ", "RSRQ Δ [dB]"),
        ("SINR", "SINR Δ [dB]")
    ]

    for ax, (metric, ylabel) in zip(axes[1], metrics_group_diff):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax.plot(df_binned["RSRP_bin_tick"], diff,
                    marker="^", markersize=3,
                    color="green", label=f"{metric} (n28-n26)",
                    linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.set_ylim(-5, 5)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad=8)

    fig.suptitle(f"{title} Rx Quality (RSRP, RSRQ, SINR)", fontsize=14, y=0.95)
    out_path = os.path.join(plot_dir, f"plot_rx_quality.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Link Adaptation (CQI, MCS, RI)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.32)

    metrics_abs = [
        ("RI", "RI"),
        ("CQI", "CQI"),
        ("DL_MCS", "DL MCS"),
    ]
    metrics_diff = [
        ("RI", "RI Δ"),
        ("CQI", "CQI Δ"),
        ("DL_MCS", "DL MCS Δ"),
    ]

    # Absolute
    for ax, (metric, ylabel) in zip(axes[0], metrics_abs):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
                style = style_map[source]
                ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                        marker=style["marker"], markersize=3,
                        color=style["color"], label=source,
                        linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad=8)

    # delta Δ
    for ax, (metric, ylabel) in zip(axes[1], metrics_diff):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax.plot(df_binned["RSRP_bin_tick"], diff,
                    marker="^", markersize=3,
                    color="green", label=f"{metric} (n28−n26)",
                    linestyle="-", linewidth=1)
            apply_common_axis(ax, df_binned, "RSRP [dBm]", ylabel)

            if metric == "CQI":
                ax.set_ylim(-10, 10)
            elif metric == "DL_MCS":
                ax.set_ylim(-20, 20)
            elif metric == "RI":
                ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.legend()
            ax.set_title(ylabel, fontsize=12, pad=8)

    fig.suptitle(f"{title} Link Adaptation (RI, CQI, DL MCS)", fontsize=14, y=0.95)
    out_path = os.path.join(plot_dir, f"plot_link_adaptation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Throughput
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.25)

    metric = "DL_Tput"
    col_n26 = f"{metric}_n26"
    col_n28 = f"{metric}_n28"

    # Absolute Throughput
    ax = axes[0]
    for source, col in zip(["n26", "n28"], [col_n26, col_n28]):
        style = style_map[source]
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col],
                marker=style["marker"], markersize=3,
                color=style["color"], label=source,
                linestyle="-", linewidth=1)
    apply_common_axis(ax, df_binned, "RSRP [dBm]", "DL Tput [Mbps]")
    ax.legend()
    # ax.set_title("Absolute DL Tput", fontsize=10, pad=7)

    # Relative Throughput
    ax = axes[1]
    if col_n26 in df_binned.columns and col_n28 in df_binned.columns:
        rel_ratio = (df_binned[col_n28] / df_binned[col_n26]) * 100
        ax.plot(df_binned["RSRP_bin_tick"], rel_ratio,
                marker="s", markersize=3, color="green", label="n28/n26 [%]",
                linestyle="-", linewidth=1)
    apply_common_axis(ax, df_binned, "RSRP [dBm]", "DL Tput Ratio [%]")
    ax.legend()
    # ax.set_title("Relative DL Tput", fontsize=10, pad=7)

    fig.suptitle(f"{title} DL Throughput", fontsize=14, y=0.98)
    out_path = os.path.join(plot_dir, f"plot_{metric}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    # plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_tput_vs_sinr(df, grid_size, out_dir, title):
    df_pair = _common.grid_kpi(df, grid_size=grid_size)

    rsrp_col = "RSRP_n28"
    bins = np.arange(int(df_pair[rsrp_col].min()), int(df_pair[rsrp_col].max()) + 1, 1)
    df_pair["RSRP_bin"] = pd.cut(df_pair["RSRP_n28"], bins=bins, right=False)
    df_binned = df_pair.groupby("RSRP_bin", observed=True).mean(numeric_only=True).reset_index()
    df_binned["RSRP_bin_tick"] = df_binned["RSRP_bin"].apply(lambda x: int(x.left))

    style_map = {"n26": {"color": "blue", "label": "n26"},
                 "n28": {"color": "red", "label": "n28"},
                 "diff": {"color": "green", "label": "Δ / Ratio"}}

    metrics = [
        ("DL_Tput", "DL Throughput [Mbps]", "DL Tput Ratio (n28/n26) [%]", "ratio"),
        ("SINR", "SINR [dB]", "SINR Δ [dB]", "diff"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3)

    for ax, (metric, ylabel, ylabel_2, mode) in zip(axes, metrics):
        col_n26 = f"{metric}_n26"
        col_n28 = f"{metric}_n28"
        if col_n26 not in df_binned.columns or col_n28 not in df_binned.columns:
            continue

        # --- 왼쪽 y축: n26 / n28 ---
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col_n26], marker='o', markersize=3,
                color=style_map["n26"]["color"], label=style_map["n26"]["label"], linewidth=1)
        ax.plot(df_binned["RSRP_bin_tick"], df_binned[col_n28], marker='o', markersize=3,
                color=style_map["n28"]["color"], label=style_map["n28"]["label"], linewidth=1)
        ax.set_xlabel("RSRP [dBm]", fontsize=11)
        ax.set_ylabel(f"{ylabel}", fontsize=11)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        ax.invert_xaxis()

        # --- 오른쪽 y축: Δ or Ratio ---
        ax2 = ax.twinx()
        if mode == "diff":
            diff = df_binned[col_n28] - df_binned[col_n26]
            ax2.plot(df_binned["RSRP_bin_tick"], diff,
                     color=style_map["diff"]["color"], linestyle="--", linewidth=0.8,
                     label=f"{metric} Δ (n28−n26)")
            absmax = np.nanmax(np.abs(diff))
            absmax = np.ceil(absmax)
            ax2.set_ylim(-absmax, absmax)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        else:
            ratio = (df_binned[col_n28] / df_binned[col_n26]) * 100
            diff_from_100 = ratio - 100
            absmax = np.nanmax(np.abs(diff_from_100))
            absmax = np.ceil(absmax)
            ax2.set_ylim(- absmax, absmax)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax2.plot(df_binned["RSRP_bin_tick"], diff_from_100,
                     color=style_map["diff"]["color"], linestyle="--", linewidth=0.8,
                     label=f"{metric} Ratio [%]")

        # --- 오른쪽 축 스타일 ---
        ax2.set_ylabel(ylabel_2, color=style_map["diff"]["color"], fontsize=11)
        ax2.tick_params(axis='y', labelcolor=style_map["diff"]["color"])

        # --- 범례 병합 ---
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

        # ax.set_title(ylabel, fontsize=12, pad=8)

    fig.suptitle(f"{title} KPI Trend : DL Tput vs SINR", fontsize=14, y=0.93)
    out_path = os.path.join(out_dir, "plot_DL_Tput_vs_SINR.png")
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
        save_dir = os.path.join(out_dir, f"plot_RB_each_test")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{date}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        # plt.show()
        print(f"Saved: {out_path}")

import os
import pandas as pd
import numpy as np
import matplotlib
import folium
from branca.colormap import StepColormap
import matplotlib.colors as mcolors
import _common

uhd_th = -32

def make_step_cmap(vmin, vmax):

    base_colors = [
        "#FF0000",  # 1 빨강
        # "#FF4D00",  # 2 밝은 주황
        "#FF8000",  # 3 오렌지
        # "#FFB300",  # 4 연한 오렌지
        "#FFD700",  # 5 노랑
        # "#E6FF33",  # 6 연노랑-연두 사이
        "#ADFF2F",  # 7 연두
        "#008000",  # 6 진한 초록
        # "#7FFF00",  # 8 진연두
        # "#33FF99",  # 9 청록
        "#00FFFF",  # 10 하늘
        # "#33CCFF",  # 11 연하늘
        "#3399FF",  # 12 중간 파랑
        "#0066FF",  # 13 진파랑
        # "#4B00FF",  # 14 보라파랑
        # "#CC66FF",  # 16 연보라
        "#8B00FF",  # 15 보라
        "#FF66CC",  # 17 핑크보라 (마무리)
    ]

    # warm_start = "#FF0000"  # 빨강
    # warm_mid = "#FFD700"  # 노랑
    # warm_end = "#008000"  # 초록
    # colors_warm_1 = mcolors.LinearSegmentedColormap.from_list("warm_1", [warm_start, warm_mid])(np.linspace(0, 1, 3))
    # colors_warm_2 = mcolors.LinearSegmentedColormap.from_list("warm_2", [warm_mid, warm_end])(np.linspace(0, 1, 3))
    # warm = np.vstack((colors_warm_1[:-1], colors_warm_2))  # mid 중복 제거
    # warm = [mcolors.to_hex(c) for c in warm]
    #
    # cool_start = "#00FFFF"  # 하늘
    # cool_mid = "#3399FF"  # 중간 파랑
    # cool_end = "#8B00FF"  # 진파랑
    # colors_cool_1 = mcolors.LinearSegmentedColormap.from_list("cool_1", [cool_start, cool_mid])(np.linspace(0, 1, 3))
    # colors_cool_2 = mcolors.LinearSegmentedColormap.from_list("cool_2", [cool_mid, cool_end])(np.linspace(0, 1, 3))
    # cool = np.vstack((colors_cool_1[:-1], colors_cool_2))  # mid 중복 제거
    # cool = [mcolors.to_hex(c) for c in cool]
    #
    # base_colors = warm + cool

    extended_colors = []
    for c in base_colors:
        extended_colors.append(c)
    step = (vmax - vmin) / len(extended_colors)
    bins = np.linspace(vmin, vmax, len(extended_colors)+1)

    # # step = round(step, 1) if step < 1 else math.floor(step)
    # if step >= 1:
    #     step = math.floor(step)
    # else:
    #     step = math.floor(step * 10) / 10
    # neg = np.arange(vmin, -step, step)
    # rem = (vmax-vmin) - 2*(len(extended_colors)//2 * step)
    # half_rem = rem/2
    # mid_neg, mid_pos = -half_rem, half_rem
    # pos = np.arange(mid_pos, vmax+step, step)
    # bins = np.concatenate([neg, [mid_neg, mid_pos], pos])
    # bins = np.unique(np.round(bins, 6))

    # print("vmin","vmax","step", vmin, vmax, step)
    # # print("bins", bins)
    # print([f"{b:.1f}" for b in bins])
    # print("bins num", len(bins)-1)
    # print("colors num", len(extended_colors))

    cmap = StepColormap(
        colors=extended_colors,
        index=bins,
        vmin=vmin,
        vmax=vmax,
    )
    cmap.tick_labels = [f"{b:.1f}" for b in bins] # int 강제 변환되는 버그 있음
    # for i in range(len(cmap.tick_labels)):
    #     if i % 2 != 0:
    #         cmap.tick_labels[i] = ""

    return cmap
    
def add_basestation(map_name):
    site_list = [
        {"name": "Huam 415-1", "lat": 37.5472288, "lon": 126.9815217},
        {"name": "Huam 345-5", "lat": 37.549636, "lon": 126.981512},
        {"name": "NamsanTower", "lat": 37.552596, "lon": 126.987184},
    ]
    for site in site_list:
        folium.Marker(
            [site["lat"], site["lon"]],
            icon=folium.Icon(color="black", icon="signal"),
            popup=f"{site['name']}"
        ).add_to(map_name)

def render_step_map(df_pair, grid_size, lat, lon, values, metric, popup_func, cmap, out_file, caption):
    m = folium.Map(location=[np.mean(lat), np.mean(lon)], zoom_start=17, tiles="cartodbpositron")

    lat_factor, lon_factor = 111320, 88000
    dlat = grid_size / (2 * lat_factor)
    dlon = grid_size / (2 * lon_factor)
    
    for idx, val in enumerate(values):
        if pd.isna(val):
            continue

        color = cmap(val)
        popup = folium.Popup(popup_func(idx, val, df_pair, metric), max_width=300)

        lat_c = lat.iloc[idx]
        lon_c = lon.iloc[idx]

        border_weight = 0
        border_color = None
        border_dash = None
        if "uhd_max" in df_pair.columns:
            uhd = df_pair.iloc[idx]["uhd_max"]
            sinr_n26 = df_pair.iloc[idx]["SINR_n26"]
            sinr_n28 = df_pair.iloc[idx]["SINR_n28"]

            if pd.notna(uhd) and uhd > uhd_th:
                border_color = "red"
                border_weight = 2
                border_dash = "3,3"

                if metric == "DL_Tput":
                    if val < -5 and (sinr_n28-sinr_n26) < 0:
                        border_weight = 3
                        border_dash = None

        bounds = [
            [lat_c - dlat, lon_c - dlon],  # 남서(SW)
            [lat_c + dlat, lon_c + dlon],  # 북동(NE)
        ]
        folium.Rectangle(
            bounds=bounds,
            weight=border_weight,
            color=border_color,
            dash_array=border_dash,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=popup,
        ).add_to(m)
        
    cmap.caption = caption
    cmap.add_to(m)
    cmap._repr_html_ = lambda: cmap._repr_html_().replace(
        "background:", "opacity:0.3; background:"
    )
    add_basestation(m)
    m.save(out_file)
    print(f"Saved: {out_file} (rows={len(values)})")


def popup_table(idx, val, df_pair, metric):
    row = df_pair.iloc[idx]

    cell_padding = "padding:2px 6px;"
    align_left  = f"text-align:left; {cell_padding}"
    align_right = f"text-align:right; {cell_padding}"

    color = (
        "color:#0070C0;" if val > 0 else
        "color:#C00000;" if val < 0 else
        "color:#000000;"
    )

    if metric in ["DL_Tput", "SINR"]:
        title = f"{metric.replace('_',' ')} Δ"
        if metric == "DL_Tput":
            subtext = "(n28/n26-100)"
            unit = "%"
        elif metric == "SINR":
            subtext = "(n28−n26)"
            unit = "dB"
    
        header_html = f"""
        <div style="text-align:left; font-size:12px; margin-bottom:6px;">
            <span style="font-weight:bold;">{title}</span>
            <span style="font-weight:normal; font-size:11px;">{subtext}</span> :
            <span style="{color}">{val:+.1f} {unit}</span>
        </div>
        """
    else:
        header_html = ""

    # --- 표 본문 생성 ---
    table_html = f"""
    <table style="border-collapse:collapse; font-size:12px;">
    <tr style="background-color:#cfd8dc;">
        <th style="{align_left}">Metric</th>
        <th style="{align_right}">n26</th>
        <th style="{align_right}">n28</th>
        <th style="{align_right}">Δ<span style='font-size:8px;'>(n28−n26)</span></th>
    </tr>
    <tr style="background-color:#f2f2f2;">
        <td style="{align_left}">counts</td>
        <td style="{align_right}">{int(row.get("sample_count_n26", 0))}</td>
        <td style="{align_right}">{int(row.get("sample_count_n28", 0))}</td>
        <td style="{align_right}"></td>
    </tr>
    """

    table_items = [
        "DL_Tput", 
        "DL_RB",
        "RSRP", "SINR", "SINR_TRS", "RSRQ",
        "CQI", "RI",
        "DL_BLER", "UL_BLER",
        "uhd_cnt", "uhd_avg", "uhd_max", "uhd_min",
    ]
    
    for item in table_items:
        c26, c28 = f"{item}_n26", f"{item}_n28"
        if c26 not in df_pair.columns or c28 not in df_pair.columns:
            continue

        v26, v28 = row[c26], row[c28]
        if pd.isna(v26) or pd.isna(v28):
            continue

        diff = v28 - v26
        if diff > 0:
            diff_color = "color:#0070C0;"
            highlight = 'background-color:#d6eaff;' if item == metric else ''
        elif diff < 0:
            diff_color = "color:#C00000;"
            highlight = 'background-color:#ffe6e6;' if item == metric else ''
        else:
            diff_color = "color:#000000;"
            highlight = 'background-color:#f2f2f2;' if item == metric else ''

        table_html += f"""
        <tr style="{highlight}">
            <td style="{align_left}">{item}</td>
            <td style="{align_right}">{v26:.1f}</td>
            <td style="{align_right}">{v28:.1f}</td>
            <td style="{align_right} {diff_color}">{diff:+.1f}</td>
        </tr>
        """

    table_html += "</table>"

    # --- UHD Power 섹션 ---
    uhd_cnt = row.get("uhd_cnt", np.nan)
    uhd_avg = row.get("uhd_avg", np.nan)
    uhd_max = row.get("uhd_max", np.nan)
    uhd_min = row.get("uhd_min", np.nan)

    def colorize(val):
        if pd.isna(val):
            return f"{val}"
        return f'<span style="color:#C00000;">{val:.1f}</span>' if val > uhd_th else f"{val:.1f}"

    if not pd.isna(uhd_cnt):
        uhd_table = f"""
        <div style="margin-top:10px; font-size:12px;">
            <div style="margin-bottom:4px; text-align:left; font-size:12px;">
                <span style="font-weight:bold;">UHD Power</span>
                <span style="font-weight:normal; font-size:11px;"> [dBm/12MHz]</span>
            </div>
            <table style="border-collapse:collapse; font-size:12px; margin-top:2px;">
                <tr style="background-color:#cfd8dc;">
                    <th style="{align_right}">max</th>
                    <th style="{align_right}">min</th>
                    <th style="{align_right}">avg</th>
                    <th style="{align_right}">cnt</th>
                </tr>
                <tr style="background-color:#f2f2f2;">
                    <td style="{align_right}">{colorize(uhd_max)}</td>
                    <td style="{align_right}">{colorize(uhd_min)}</td>
                    <td style="{align_right}">{colorize(uhd_avg)}</td>
                    <td style="{align_right}">{int(uhd_cnt)}</td>
                </tr>
            </table>
        </div>
        """
        table_html += uhd_table

    return header_html + table_html

def map_pct(df, out_dir, grid_size, sample_min=0):

    df_pair = _common.grid_kpi(df, grid_size=grid_size)
    df_pair = df_pair[(df_pair["sample_count_n26"] >= sample_min) & (df_pair["sample_count_n28"] >= sample_min)].reset_index(drop=True)

    lat_factor, lon_factor = 111320, 88000
    lat = (df_pair["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df_pair["lon_bin"] + 0.5) * (grid_size / lon_factor)
    
    metrics_pct = [
        "DL_Tput",
        # "DL_RB",
        # "DL_Tput_per_RB",
        # "DL_Tput_full_RB",
    ]

    for metric_pct in metrics_pct:
        n26 = df_pair[f"{metric_pct}_n26"].astype(float)
        n28 = df_pair[f"{metric_pct}_n28"].astype(float)

        ratio = (n28 / n26.replace(0, np.nan)) * 100.0
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        ratio_diff = ratio - 100

        # vabs = int(np.ceil(np.nanmax(np.abs(ratio_diff))))

        mean = np.nanmean(ratio_diff)
        std = np.nanstd(ratio_diff)
        # vabs = max(abs(mean - 1.96 * std), abs(mean + 1.96 * std)) # 2.5% tail 제외
        # vabs = max(abs(mean - 1.645 * std), abs(mean + 1.645 * std)) # 5% tail 제외
        vabs = max(abs(mean - 1.28 * std), abs(mean + 1.28 * std)) # 10% tail 제외

        vmin, vmax = -vabs, vabs
        cmap = make_step_cmap(vmin, vmax)

        out_file = os.path.join(out_dir, f"{grid_size}m_{metric_pct}_ratio.html")
        caption = f"{metric_pct} Δ(n28/n26) [%-100]"
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=ratio_diff,
            metric=metric_pct,
            popup_func=popup_table,
            cmap=cmap,
            out_file=out_file,
            caption=caption
        )

def map_db(df, out_dir, grid_size, sample_min=0):

    df_pair = _common.grid_kpi(df, grid_size=grid_size)
    df_pair = df_pair[(df_pair["sample_count_n26"] >= sample_min) & (df_pair["sample_count_n28"] >= sample_min)].reset_index(drop=True)
    
    lat_factor, lon_factor = 111320, 88000
    lat = (df_pair["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df_pair["lon_bin"] + 0.5) * (grid_size / lon_factor)
    
    metrics_db = [
        "RSRP",
        "SINR", 
        # "SINR_TRS", 
        # "RSRQ",
    ]

    for metric_db in metrics_db:
        n26 = df_pair[f"{metric_db}_n26"].astype(float)
        n28 = df_pair[f"{metric_db}_n28"].astype(float)

        diff = n28 - n26

        # vabs = int(np.ceil(np.nanmax(np.abs(diff))))

        # mean = np.nanmean(diff)
        # std = np.nanstd(diff)
        # vabs = max(abs(mean - 1.96 * std), abs(mean + 1.96 * std)) # 2.5% tail 제외
        # vabs = max(abs(mean - 1.645 * std), abs(mean + 1.645 * std)) # 5% tail 제외
        # vabs = max(abs(mean - 1.28 * std), abs(mean + 1.28 * std)) # 10% tail 제외

        vabs = 10

        vmin, vmax = -vabs, vabs
        cmap = make_step_cmap(vmin, vmax)

        out_file = os.path.join(out_dir, f"{grid_size}m_{metric_db}_diff.html")
        caption = f"{metric_db} Δ(n28-n26) [dB]"
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=diff,
            metric=metric_db,
            popup_func=popup_table,
            cmap=cmap,
            out_file=out_file,
            caption=caption
        )

def map_coverage(df, out_dir, grid_size, sample_min=0):
    
    df_pair = _common.grid_kpi(df, grid_size=grid_size)
    df_pair = df_pair[(df_pair["sample_count_n26"] >= sample_min) & (df_pair["sample_count_n28"] >= sample_min)].reset_index(drop=True)

    lat_factor, lon_factor = 111320, 88000
    lat = (df_pair["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df_pair["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metrics = [
        "RSRP",
    ]

    for metric in metrics:
        n28 = df_pair[f"{metric}_n28"].astype(float)

        vmin = np.floor(n28.min())
        vmax = np.ceil(n28.max())
        cmap = make_step_cmap(vmin, vmax)


        caption = f"n28 {metric} [dBm]"
        out_file = os.path.join(out_dir, f"{grid_size}m_{metric}_n28.html")
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=n28,
            metric=metric,
            popup_func=popup_table,
            cmap=cmap,
            out_file=out_file,
            caption=caption,
        )

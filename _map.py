
import os
import pandas as pd
import numpy as np
import matplotlib
import folium
from branca.colormap import StepColormap
import _common
from collections import defaultdict

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

    extended_colors = []
    for c in base_colors:
        extended_colors.append(c)
    bins = np.linspace(vmin, vmax, len(extended_colors)+1)

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

def render_step_map(df_pair, grid_size, lat, lon, values, metric, popup_func, band, cmap, out_file, caption):
    m = folium.Map(location=[np.mean(lat), np.mean(lon)], zoom_start=17, tiles="cartodbpositron")

    lat_factor, lon_factor = 111320, 88000
    dlat = grid_size / (2 * lat_factor)
    dlon = grid_size / (2 * lon_factor)
    
    for idx, val in enumerate(values):
        if pd.isna(val):
            continue

        color = cmap(val)
        popup_html = popup_func(idx, val, df_pair, metric, out_file, band)
        popup = folium.Popup(popup_html, max_width=300)

        lat_c = lat.iloc[idx]
        lon_c = lon.iloc[idx]

        border_weight = 0
        border_color = None
        if "uhd_max" in df_pair.columns:
            uhd = df_pair.iloc[idx]["uhd_max"]

            if pd.notna(uhd) and uhd > uhd_th:
                border_color = "blue"
                border_weight = 2

        bounds = [
            [lat_c - dlat, lon_c - dlon],  # 남서(SW)
            [lat_c + dlat, lon_c + dlon],  # 북동(NE)
        ]
        rect = folium.Rectangle(
            bounds=bounds,
            weight=border_weight,
            color=border_color,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=popup,
        )
        rect.add_to(m)
        
    cmap.caption = caption
    cmap.add_to(m)
    cmap._repr_html_ = lambda: cmap._repr_html_().replace(
        "background:", "opacity:0.3; background:"
    )
    add_basestation(m)
    m.save(out_file)

    print(f"✅ Saved: {out_file} (rows={len(values)})")


def popup_table(idx, val, df_pair, metric, out_file, band):
    row = df_pair.iloc[idx]

    cell_padding = "padding:2px 6px;"
    align_left  = f"text-align:left; {cell_padding}"
    align_right = f"text-align:right; {cell_padding}"

    color = (
        "color:#0070C0;" if val > 0 else
        "color:#C00000;" if val < 0 else
        "color:#000000;"
    )

    table_items = [
        "RSRP", "RSRQ",
        "SINR_SSB", "SINR_TRS",
        "DL_RB",
        "DL_Tput",
        "DL_Tput_per_RB",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
    ]
    if not band:
        if metric == "DL_Tput":
            title = f"{metric.replace('_', ' ')} Δ"
            subtext = "(n28/n26-100)"
            unit = "%"

            header_html = f"""
            <div style="text-align:left; font-size:12px; margin-bottom:6px;">
                <span style="font-weight:bold;">{title}</span>
                <span style="font-weight:normal; font-size:11px;">{subtext}</span> :
                <span style="{color}">{val:+.1f} {unit}</span>
            </div>
            """
        else:
            header_html = ""

        n26 = int(row.get("sample_count_n26", 0))
        n28 = int(row.get("sample_count_n28", 0))
        n_diff = int(row.get("sample_count_diff", 0))

        table_html = f"""
        <table style="border-collapse:collapse; font-size:12px;">
        <tr style="background-color:#cfd8dc;">
            <th style="{align_left}">Metric</th>
            <th style="{align_right}">n26</th>
            <th style="{align_right}">n28</th>
            <th style="{align_right}; vertical-align:middle;">
                <div style="display:flex; flex-direction:column; align-items:flex-end; line-height:1.1;">
                    <span>Δ<span style='font-size:8px;'>(n28−n26)</span></span>
                    <span style='font-size:8px; color:#555;'>(±95% CI)</span>
                </div>
            </th>
        </tr>
        <tr style="background-color:#f2f2f2;">
            <td style="{align_left}">counts</td>
            <td style="{align_right}">{n26}</td>
            <td style="{align_right}">{n28}</td>
            <td style="{align_right}">{n_diff}</td>
        </tr>
        """

        for metric_name in table_items:
            c26_mean = f"{metric_name}_mean_n26"
            c28_mean = f"{metric_name}_mean_n28"
            c_diff_mean = f"{metric_name}_mean_diff"
            c_diff_std = f"{metric_name}_std_diff"

            if not all(c in df_pair.columns for c in [c26_mean, c28_mean, c_diff_mean, c_diff_std]):
                continue

            v26 = row[c26_mean]
            v28 = row[c28_mean]
            diff_mean = row[c_diff_mean]
            diff_std = row[c_diff_std]

            if any(pd.isna(x) for x in [v26, v28, diff_mean, diff_std]) or n_diff <= 1:
                continue

            # 95% CI 계산
            se_diff = diff_std / np.sqrt(n_diff)
            ci_delta = 1.96 * se_diff

            # 색상 처리
            if diff_mean > 0:
                diff_color = "color:#0070C0;"
                highlight = 'background-color:#d6eaff;' if metric_name == metric else ''
            elif diff_mean < 0:
                diff_color = "color:#C00000;"
                highlight = 'background-color:#ffe6e6;' if metric_name == metric else ''
            else:
                diff_color = "color:#000000;"
                highlight = 'background-color:#f2f2f2;' if metric_name == metric else ''

            table_html += f"""
            <tr style="{highlight}">
                <td style="{align_left}">{metric_name}</td>
                <td style="{align_right}">{v26:.1f}</td>
                <td style="{align_right}">{v28:.1f}</td>
                <td style="{align_right} {diff_color}">
                    {diff_mean:+.1f}
                    <span style="font-size:10px; color:#555;">(±{ci_delta:.1f})</span>
                </td>
            </tr>
            """
        table_html += "</table>"
        table_html = header_html + table_html
    else:
        n_count = int(row.get(f"sample_count_{band}", 0))
        table_html = f"""
        <div style="font-weight:bold; text-align:left; margin-bottom:4px; font-size:13px;">
            Metric Stats <span style="font-weight:normal; font-size:11px;">({n_count} samples)</span>
        </div>
        """
        table_html += f"""
        <table style="border-collapse:collapse; font-size:12px; white-space:nowrap;">
        <tr style="background-color:#e0e0e0;">
            <th style="{align_left};">Metric</th>
            <th style="{align_right};">Mean</th>
            <th style="{align_right};">
                95% CI <span style="font-weight:normal;">(±Δ)</span>
        </tr>
        """

        for metric_name in table_items:
            mean_col = f"{metric_name}_mean_{band}"
            std_col  = f"{metric_name}_std_{band}"

            if mean_col not in df_pair.columns or std_col not in df_pair.columns:
                continue

            mean_val = row[mean_col]
            std_val  = row[std_col]

            if pd.isna(mean_val) or pd.isna(std_val) or n_count <= 1:
                continue

            se = std_val / np.sqrt(n_count)
            ci_delta = 1.96 * se  # ±Δ

            if metric_name in ["DL_Tput", "SINR_TRS", "RSRP"]:
                highlight = "background-color:#e3f2fd;"
            else:
                highlight = ""

            table_html += f"""
            <tr style="{highlight}">
                <td style="{align_left}">{metric_name}</td>
                <td style="{align_right}">{mean_val:.2f}</td>
                <td style="{align_right}">{ci_delta:.2f}</td>
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

    test_list = row.get("test_list", [])
    loc_id = row.get("loc_id", np.nan)

    if isinstance(test_list, list) and len(test_list) > 0:

        test_by_date = defaultdict(list)
        for test in test_list:
            parts = test.split("_")
            date, num, site = parts[0], parts[1], parts[2]
            test_by_date[date].append((num, site))

        test_html = f"""
        <div style="margin-top:10px; font-size:12px;">
            <div style="font-weight:bold; color:#000; margin-bottom:2px;">
                View Test Results
                <span style="font-size:11px; font-weight:normal;">(loc_id: {loc_id})</span>
            </div>
            <details style="border:1px solid #ccc; border-radius:4px; padding:4px;">
                <summary style="cursor:pointer; font-weight:normal; font-size:11px; color:#777;">
                    click to expand
                </summary>
                <div style="margin-top:6px; padding-left:10px;">
        """
        out_dir = '/'.join(out_file.split("/")[0:2])
        base_url = f"https://joostone-ahn.github.io/nr-field-analysis/{out_dir}/plot/plot_kpis_each_test"

        for date, entries in sorted(test_by_date.items()):
            test_html += f"""
            <div style="margin:2px 0; line-height:1.4;">
                <div style="display:inline-block; width:60px; font-weight:bold; color:#333; text-align:right; vertical-align:top; white-space:nowrap;">
                    {date} :
                </div>
                <div style="display:inline-block; width:calc(100% - 70px); vertical-align:top;">
            """
            for num, site in sorted(entries, key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
                url = f"{base_url}/{date}/{site}/TEST_{num}.html"
                test_html += (
                    f'<a href="{url}" target="_blank" '
                    f'style="text-decoration:none; color:#0066cc; margin-right:6px;">{num}</a>'
                )
            test_html += "</div></div>\n"

        test_html += """
                </div>
            </details>
        </div>
        """

        table_html += test_html

    return table_html

def map_pct(df, out_dir, grid_size, rb_min, sample_min):

    df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)

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
        n26 = df_pair[f"{metric_pct}_mean_n26"].astype(float)
        n28 = df_pair[f"{metric_pct}_mean_n28"].astype(float)

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

        out_file = os.path.join(out_dir, f"cmpr_{metric_pct}.html")
        caption = f"{metric_pct} Δ(n28/n26) [%-100]"
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=ratio_diff,
            metric=metric_pct,
            popup_func=popup_table,
            band=None,
            cmap=cmap,
            out_file=out_file,
            caption=caption
        )

def map_db(df, out_dir, grid_size, rb_min, sample_min):

    df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)

    lat_factor, lon_factor = 111320, 88000
    lat = (df_pair["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df_pair["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metrics = [
        {"name": "RSRP",           "vmin": -10,   "vmax": 10,  "unit": "dB"},
        # {"name": "SINR_TRS",       "vmin": -10,   "vmax": 10,  "unit": "dB"},
    ]

    for m in metrics:
        metric = m['name']
        vmin, vmax = m['vmin'], m['vmax']
        unit = m['unit']

        n26 = df_pair[f"{metric}_mean_n26"].astype(float)
        n28 = df_pair[f"{metric}_mean_n28"].astype(float)
        diff = n28 - n26

        cmap = make_step_cmap(vmin, vmax)

        out_file = os.path.join(out_dir, f"cmpr_{metric}.html")
        caption = f"Δ{metric} (n28-n26) [{unit}]"
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=diff,
            metric=metric,
            popup_func=popup_table,
            band=None,
            cmap=cmap,
            out_file=out_file,
            caption=caption
        )

def map_coverage(df, out_dir, grid_size, rb_min, sample_min, band="n28"):

    df_pair = _common.grid_kpi(df, grid_size=grid_size, rb_min=rb_min, sample_min=sample_min)

    lat_factor, lon_factor = 111320, 88000
    lat = (df_pair["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df_pair["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metrics = [
        {"name": "RSRP",           "vmin": -120, "vmax": -60,  "unit": "dBm"},
        {"name": "SINR_TRS",       "vmin": 10,   "vmax": 40,   "unit": "dB"},
        {"name": "DL_Tput",        "vmin": 0,    "vmax": 100,  "unit": "Mbps"},
        {"name": "DL_Tput_per_RB", "vmin": 0,    "vmax": 2,    "unit": "Mbps"},
        {"name": "DL_RB",          "vmin": 0,    "vmax": 50,   "unit": ""},
    ]

    for m in metrics:
        metric = m['name']
        vmin, vmax = m['vmin'], m['vmax']
        unit = m['unit']

        n28 = df_pair[f"{metric}_mean_n28"].astype(float)
        cmap = make_step_cmap(vmin, vmax)
        caption = f"{band} {metric} [{unit}]" if unit != "" else f"{band} {metric}"
        out_file = os.path.join(out_dir, f"{band}_{metric}.html")
        render_step_map(
            df_pair=df_pair,
            grid_size=grid_size,
            lat=lat,
            lon=lon,
            values=n28,
            metric=metric,
            popup_func=popup_table,
            band=band,
            cmap=cmap,
            out_file=out_file,
            caption=caption,
        )


import os
import pandas as pd
import numpy as np
import matplotlib
import folium
from folium.plugins import BeautifyIcon
from branca.colormap import StepColormap
import ast
import _common
from collections import defaultdict

uhd_th = -30

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
        {"name": "Huam 415-1", "lat": 37.54721361, "lon": 126.98147333},
        {"name": "Huam 345-5", "lat": 37.54963306, "lon": 126.98153194},
        {"name": "NamsanTower", "lat": 37.5524474, "lon": 126.98724844},

    ]
    for site in site_list:
        folium.Marker(
            [site["lat"], site["lon"]],
            icon=folium.Icon(color="red", icon="broadcast-tower", prefix="fa"),
            popup=f"{site['name']}"
        ).add_to(map_name)

    uhd_lat, uhd_lon = 37.551130, 126.987443

    folium.Marker(
        [uhd_lat, uhd_lon],
        icon=folium.Icon(color="darkblue", icon="tower-cell", prefix='fa'),
        popup="UHD Broadcasting Tower"
    ).add_to(map_name)

    folium.Circle(
        location=[uhd_lat, uhd_lon],
        radius=1000,
        color="blue",
        weight=1,
        dash_array="3,3",
        fill=False,
        popup="1km"
    ).add_to(map_name)

    lte_b5_pci_list = [
        (3, 37.54800111, 126.98397667),
        (15, 37.54541233, 126.9854189),
        # (19, 37.54303361, 126.98501861),
        (25, 37.54935417, 126.97933694),
        # (37, 37.54443639, 126.98671278),
        (73, 37.5512825, 126.97917694),
        # (84, 37.5524474, 126.98724844),
        (136, 37.54406611, 126.98321917),
        # (155, 37.54721361, 126.98147333),
        # (156, 37.54211961, 126.98399833),
        # (159, 37.54696835, 126.97931584),
        # (175, 37.54614703, 126.97893946),
        (181, 37.5466775, 126.98423639),
        (190, 37.54764344, 126.98631522),
        (201, 37.551974, 126.980083),
        (204, 37.54511622, 126.98382172),
        # (207, 37.54313435, 126.98617105),
        # (212, 37.54603684, 126.97798664),
        (233, 37.55286576, 126.98172569),
        # (241, 37.54963306, 126.98153194),
        # (255, 37.5480225, 126.97774083),
        (265, 37.54479389, 126.9817375),
        # (270, 37.54997333, 126.9775925),
        (273, 37.54935417, 126.97933694),
        # (296, 37.543597, 126.984315),
        # (299, 37.54935417, 126.97933694),
        # (301, 37.54884417, 126.98626278),
        (303, 37.54596253, 126.98509033),
        (307, 37.55008833, 126.98204222),
        (308, 37.5518942, 126.97919868),
        # (354, 37.5524474, 126.98724844),
        (359, 37.54764344, 126.98631522),
        (364, 37.54764344, 126.98631522),
        # (365, 37.54764344, 126.98631522),
        # (366, 37.5448412, 126.9882567),
        (366, 37.54566688, 126.98288902),
        # (368, 37.5524474, 126.98724844),
        # (369, 37.5448267, 126.9882279),
        (390, 37.55118544, 126.9881064),
        (391, 37.55118544, 126.9881064),
        # (417, 37.54985218, 126.98487871),
        # (420, 37.5524474, 126.98724844),
        # (437, 37.546339, 126.976871),
        # (439, 37.546339, 126.976871),
        # (440, 37.546339, 126.976871),
        # (442, 37.546339, 126.976871),
        (443, 37.544483, 126.985776),
        (444, 37.544483, 126.985776),
        (446, 37.544483, 126.985776),
        (459, 37.55160578, 126.99030941),
    ]

    for pci, lat, lon in lte_b5_pci_list:
        icon = BeautifyIcon(
            icon_shape='circle',
            background_color='#9E9E9E',
            text_color='white',
            number='B5',
            border_width=0,
            inner_icon_style=(
                'font-size:9px;'
                'font-weight:bold;'
                'position: relative;'
                'top:50%;'
                'transform:translateY(-47%);'
                'text-align:center;'
            ),
            radius=4,
        )
        folium.Marker(
            [lat, lon],
            icon=icon,
            # popup=f"{pci}"
        ).add_to(map_name)

def render_step_map(df, grid_size, lat, lon, values, metric, popup_func, band, cmap, out_file, caption):
    uhd_lat, uhd_lon = 37.551130, 126.987443
    # m = folium.Map(location=[np.mean(lat), np.mean(lon)], zoom_start=16, tiles="cartodbpositron")
    m = folium.Map(location=[uhd_lat,uhd_lon], zoom_start=16, tiles="cartodbpositron")

    lat_factor, lon_factor = 111320, 88000
    dlat = grid_size / (2 * lat_factor)
    dlon = grid_size / (2 * lon_factor)
    
    for idx, val in enumerate(values):
        if pd.isna(val):
            continue

        color = cmap(val)
        popup_html = popup_func(idx, val, df, metric, band)
        popup = folium.Popup(popup_html, max_width=300)

        lat_c = lat.iloc[idx]
        lon_c = lon.iloc[idx]

        border_weight = 0
        border_color = None
        if "uhd_avg" in df.columns:
            uhd = df.iloc[idx]["uhd_avg"]
            if pd.notna(uhd) and uhd > uhd_th:
                border_color = "blue"
                if grid_size >= 15:
                    border_weight = 2
                if grid_size <= 5:
                    border_weight = 1

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

def popup_table(idx, val, df, metric, band):
    cell_padding = "padding:2px 6px;"
    align_left  = f"text-align:left; {cell_padding}"
    align_right = f"text-align:right; {cell_padding}"

    table_items = [
        "RSRP", "RSRQ",
        "SINR",
        "SINR_TRS",
        "DL_RB",
        "DL_Tput",
        "DL_Tput_per_RB",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
    ]

    row = df.iloc[idx]
    loc_id = row.get("loc_id", None)

    if not band:
        if metric == "DL_Tput":
            title = f"{metric.replace('_', ' ')} Δ"
            subtitle = "n28/n26"
            unit = "%"
        elif metric == "RSRP":
            title =  f"{metric} Δ"
            subtitle = "n28-n26"
            unit = "dB"

        color = (
            "color:#0070C0;" if val > 0 else
            "color:#C00000;" if val < 0 else
            "color:#000000;"
        )

        header_html = f"""
        <div style="
            display:flex;
            align-items:flex-end;
            font-size:12px;
            font-weight:bold;
            margin-bottom:6px;
        ">
            <span style="
                background-color:#424242;   
                color:#FFFFFF;              
                border-radius:3px;
                padding:1px 5px;
                margin-right:5px;
                font-size:11px;
                box-shadow:0 0 1px rgba(0,0,0,0.2);
            ">{loc_id}</span>
            <span>{title}</span>
            <span style="font-size:11px;font-weight:normal;margin-left:4px;">({subtitle})</span>
            <span style="font-weight:normal;margin-left:4px;">:</span>
            <span style="{color};margin-left:4px;">{val:+.1f} {unit}</span>
        </div>
        """

        n26 = int(row.get("sample_count_n26", 0))
        n28 = int(row.get("sample_count_n28", 0))
        n_diff = int(row.get("sample_count_diff", 0))

        table_html = f"""
        {header_html}
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

            if not all(c in df.columns for c in [c26_mean, c28_mean, c_diff_mean, c_diff_std]):
                continue

            v26 = row[c26_mean]
            v28 = row[c28_mean]
            diff_mean = row[c_diff_mean]
            diff_std = row[c_diff_std]

            if any(pd.isna(x) for x in [v26, v28, diff_mean, diff_std]) or n_diff <= 1:
                continue

            se_diff = diff_std / np.sqrt(n_diff)
            ci_delta = 1.96 * se_diff

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

    elif band in ['n26','n28']:
        n_count = int(row.get(f"sample_count_{band}", 0))

        header_html = f"""
        <div style="
            display:flex;
            align-items:flex-end;
            font-size:12px;
            font-weight:bold;
            margin-bottom:6px;
        ">
            <span style="
                background-color:#424242;   
                color:#FFFFFF;              
                border-radius:3px;
                padding:1px 5px;
                margin-right:5px;
                font-size:11px;
                box-shadow:0 0 1px rgba(0,0,0,0.2);
            ">{loc_id}</span>
            Statistic
            <span style="font-size:11px; font-weight:normal; color:#333; margin-left:4px;">
                ({n_count} samples)
            </span>
        </div>
        """

        table_html = f"""
        {header_html}
        <table style="border-collapse:collapse; font-size:12px; white-space:nowrap;">
        <tr style="background-color:#e0e0e0;">
            <th style="{align_left};">Metric</th>
            <th style="{align_right};">Avg</th>
            <th style="{align_right};">
                CI<span style="font-weight:normal; font-size:9px;">(95%)</span>
            </th>
        </tr>
        """

        for metric_name in table_items:
            mean_col = f"{metric_name}_mean_{band}"
            std_col  = f"{metric_name}_std_{band}"

            if mean_col not in df.columns or std_col not in df.columns:
                continue

            mean_val = row[mean_col]
            std_val  = row[std_col]

            # cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else np.nan
            se = std_val / np.sqrt(n_count)
            ci_delta = 1.96 * se  # ±Δ

            if metric_name in ["DL_Tput", "SINR", "RSRP"]:
                highlight = "background-color:#e3f2fd;"
            else:
                highlight = ""

            table_html += f"""
            <tr style="{highlight}">
                <td style="{align_left}">{metric_name}</td>
                <td style="{align_right}">{mean_val:.1f}</td>
                <td style="{align_right}">±{ci_delta:.2f}</td>
            </tr>
            """
        table_html += "</table>"

    uhd_cnt = row.get("uhd_cnt", np.nan)
    uhd_avg = row.get("uhd_avg", np.nan)
    uhd_max = row.get("uhd_max", np.nan)
    uhd_min = row.get("uhd_min", np.nan)
    uhd_ci95 = row.get("uhd_ci95", np.nan)

    def colorize(val):
        if pd.isna(val):
            return f"{val}"
        return f'<span style="color:#C00000;">{val:.1f}</span>' if val > uhd_th else f"{val:.1f}"

    uhd_html = f"""
    {table_html}
    <div style="margin-top:10px; font-size:12px;">
        <div style="margin-bottom:4px; text-align:left; font-size:12px;">
            <span style="font-weight:bold;">UHD Power</span>
            <span style="font-weight:normal; font-size:11px;"> [dBm/12MHz]</span>
        </div>
        <table style="border-collapse:collapse; font-size:12px; margin-top:2px;">
            <tr style="background-color:#cfd8dc;">
                <th style="{align_right}">cnt</th>
                <th style="{align_right}">max</th>
                <th style="{align_right}">min</th>  
                <th style="{align_right}">avg</th>
                <th style="{align_right};">
                    CI<span style="font-weight:normal; font-size:9px;">(95%)</span>
                </th>               
            </tr>
            <tr style="background-color:#f2f2f2;">
                <td style="{align_right}">{int(uhd_cnt)}</td>
                <td style="{align_right}">{round(uhd_max, 1)}</td>
                <td style="{align_right}">{round(uhd_min, 1)}</td>
                <td style="{align_right}">{colorize(round(uhd_avg, 1))}</td>
                <td style="{align_right}">{round(uhd_ci95, 2)}</td>
            </tr>
        </table>
    </div>
    """

    test_list = row.get("test_list", [])
    if isinstance(test_list, str):
        test_list = ast.literal_eval(test_list)

    test_html = f"""
    {uhd_html}
    """
    if isinstance(test_list, list) and len(test_list) > 0:
        test_by_date = defaultdict(list)

        for test in test_list:
            parts = test.split("_")
            date, num, site = parts[0], parts[1], parts[2]
            test_by_date[date].append((num, site))

        test_html += f"""
        <div style="margin-top:10px; font-size:12px;">
            <div style="font-weight:bold; color:#000; margin-bottom:2px;">
                View Test Results
            </div>
            <details style="border:1px solid #ccc; border-radius:4px; padding:4px;">
                <summary style="cursor:pointer; font-weight:normal; font-size:11px; color:#777;">
                    click to expand
                </summary>
                <div style="margin-top:6px; padding-left:10px;">
        """
        base_url = f"https://joostone-ahn.github.io/nr-field-analysis/results/plot_each_test"

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
    return test_html

def map_pct(df, out_dir, grid_size):

    lat_factor, lon_factor = 111320, 88000
    lat = (df["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df["lon_bin"] + 0.5) * (grid_size / lon_factor)
    
    metrics_pct = [
        "DL_Tput",
        # "DL_RB",
        # "DL_Tput_per_RB",
        # "DL_Tput_full_RB",
    ]

    for metric_pct in metrics_pct:
        n26 = df[f"{metric_pct}_mean_n26"].astype(float)
        n28 = df[f"{metric_pct}_mean_n28"].astype(float)

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

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"cmpr_{metric_pct}.html")
        caption = f"{metric_pct} Δ(n28/n26) [%-100]"
        render_step_map(
            df=df,
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

def map_db(df, out_dir, grid_size):

    lat_factor, lon_factor = 111320, 88000
    lat = (df["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metrics = [
        {"name": "RSRP",           "vmin": -10,   "vmax": 10,  "unit": "dB"},
        # {"name": "SINR_TRS",       "vmin": -10,   "vmax": 10,  "unit": "dB"},
    ]

    for m in metrics:
        metric = m['name']
        vmin, vmax = m['vmin'], m['vmax']
        unit = m['unit']

        n26 = df[f"{metric}_mean_n26"].astype(float)
        n28 = df[f"{metric}_mean_n28"].astype(float)
        diff = n28 - n26

        cmap = make_step_cmap(vmin, vmax)

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"cmpr_{metric}.html")
        caption = f"Δ{metric} (n28-n26) [{unit}]"
        render_step_map(
            df=df,
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

def map_coverage(df, out_dir, grid_size, band):

    lat_factor, lon_factor = 111320, 88000
    lat = (df["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metrics = [
        {"name": "RSRP",           "vmin": -120, "vmax": -50,  "unit": "dBm"},
        {"name": "SINR",           "vmin": -5,    "vmax": 45,   "unit": "dB"},
        {"name": "DL_Tput",        "vmin": 0,    "vmax": 120,  "unit": "Mbps"},
        # {"name": "DL_Tput_per_RB", "vmin": 0,    "vmax": 2,    "unit": "Mbps"},
        # {"name": "DL_RB",          "vmin": 0,    "vmax": 50,   "unit": ""},
    ]

    for m in metrics:
        metric = m['name']
        vmin, vmax = m['vmin'], m['vmax']
        unit = m['unit']

        n28 = df[f"{metric}_mean_n28"].astype(float)
        cmap = make_step_cmap(vmin, vmax)
        caption = f"{band} {metric} [{unit}]" if unit != "" else f"{band} {metric}"

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{band}_{metric}.html")
        render_step_map(
            df=df,
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

def map_uhd_pwr(df, out_dir, grid_size):

    lat_factor, lon_factor = 111320, 88000
    lat = (df["lat_bin"] + 0.5) * (grid_size / lat_factor)
    lon = (df["lon_bin"] + 0.5) * (grid_size / lon_factor)

    metric = 'uhd_avg'
    unit = 'dBm/12MHz'

    vmin = df[metric].min()
    vmax = df[metric].max()

    uhd_avg = df[metric].astype(float)
    cmap = make_step_cmap(vmin, vmax)
    caption = f"UHD Power Avg [{unit}]"

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"n28_UHD_pwr_avg.html")
    render_step_map(
        df=df,
        grid_size=grid_size,
        lat=lat,
        lon=lon,
        values=uhd_avg,
        metric=metric,
        popup_func=popup_table,
        band='n28',
        cmap=cmap,
        out_file=out_file,
        caption=caption,
    )
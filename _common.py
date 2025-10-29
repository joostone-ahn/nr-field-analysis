import os
import pandas as pd
import numpy as np

band_map = {
    868.85: "n26",
    798.15: "n28", # upper
}

def read_UHD(uhd_dir):
    csv_files = [f for f in os.listdir(uhd_dir) if f.endswith('.csv')]
    df_list = []
    for f in csv_files:
        file_path = os.path.join(uhd_dir, f)
        df_temp = pd.read_csv(file_path, index_col=0)
        df_list.append(df_temp)
    df_uhd = pd.concat(df_list, ignore_index=True)
    
    return df_uhd

def grid_uhd(df_uhd, grid_size):
    lat_factor, lon_factor = 111320, 88000  

    df_grid = df_uhd.copy()

    df_grid["lat_bin"] = (df_grid["lat"] * lat_factor // grid_size).astype(int)
    df_grid["lon_bin"] = (df_grid["lon"] * lon_factor // grid_size).astype(int)

    df_agg = (
        df_grid.groupby(["lat_bin", "lon_bin"])
        .agg(
            uhd_min=("value", "min"),
            uhd_max=("value", "max"),
            uhd_avg=("value", "mean"),
            uhd_cnt=("value", "count"),
        )
        .reset_index()
    )

    # --- 컬럼 순서 정리 ---
    df_agg = df_agg[
        ["lat_bin", "lon_bin", "uhd_cnt", "uhd_avg", "uhd_max", "uhd_min"]
    ]

    df_agg = df_agg.round(2)
    
    return df_agg
    
def read_logs():
    log_dir = "logs"
    device_data = {}  
    route_list = []
    for route in os.listdir(log_dir):
        route_path = os.path.join(log_dir, route)
        if not os.path.isdir(route_path):
            continue  # 폴더가 아니면 스킵
        route_list.append(route)
        # route 폴더 안의 csv 파일만 처리
        for fname in os.listdir(route_path):
            if fname.endswith(".xlsx"):
                fpath = os.path.join(route_path, fname)

                parts = fname.replace(".xlsx", "").split("_")
                if len(parts) < 3:
                    print(f"⚠️ Skipped (unexpected filename): {fname}")
                    continue

                date    = parts[0]
                device  = parts[1]
                test_no = parts[2]
                if '-' in test_no:
                    test_no = test_no.split('-')[0]

                df = pd.read_excel(fpath)
                print(f"Reading {fpath}")

                df["date"]     = date
                df["device"]   = device
                df["test_no"]  = test_no
                df["route"]    = route

                if device not in device_data:
                    device_data[device] = []

                device_data[device].append(df)

    for device, df_list in device_data.items():
        merged = pd.concat(df_list, ignore_index=True)
        merge_path = os.path.join(log_dir, f"{device}_All.xlsx")
        merged.to_excel(merge_path, index=False)
        print(f"✅ Saved: {merge_path}")

def analyze_kpi(fname, date_list):
    print(f"Reading {fname}")
    df= pd.read_excel(fname)
    # print(df.info())
    # display(df)
    print(f"✅ Read Complete : {len(df)} lines")

    unique_values = df["5G KPI PCell Chip Type"].dropna().drop_duplicates().tolist()
    if len(unique_values) > 1:
        print(unique_values)
    
    col_map = {
        "TIME_STAMP": "TIME",
        "5G KPI PCell RF Serving PCI": "PCI",
        # "5G KPI PCell RF Band": "Band",
        "5G KPI PCell RF Frequency [MHz]": "Freq",
        "5G KPI PCell RF Serving SS-RSRP [dBm]": "RSRP",
        "5G KPI PCell RF Serving SS-RSRQ [dB]": "RSRQ",
        "5G KPI PCell RF Serving SS-SINR [dB]": "SINR",
        "Qualcomm 5G-NR LL1 Serving Freq Tracking Loop Result PCell FTL SNR_SSB [dB]": "SINR_SSB",     
        "Qualcomm 5G-NR LL1 Serving Freq Tracking Loop Result PCell FTL SNR_TRS [dB]": "SINR_TRS",
        "5G KPI PCell RF RI": "RI",
        "5G KPI PCell RF CQI": "CQI",
        "5G KPI PCell Layer1 DL BLER [%]": "DL_BLER",
        "5G KPI PCell Layer1 UL BLER [%]": "UL_BLER",
        "5G KPI PCell Layer1 DL MCS (Avg)": "DL_MCS",
        "5G KPI PCell Layer1 DL RB Num (Including 0)": "DL_RB",
        "5G KPI PCell Layer1 PDSCH Throughput [Mbps]": "DL_Tput",
        # "5G KPI PCell Layer2 MAC DL Throughput [Mbps]": "MAC DL Throughput",
        "GPS Lon": "Lon",
        "GPS Lat": "Lat",
        "date": "date",
        "test_no": "test_no",
        "device": "device",
        "route": "route",
    }
    df = df[list(col_map.keys())].rename(columns=col_map)
    # print(len(df))

    if date_list:
        df["date"] = df["date"].astype(str)
        df = df[df["date"].isin(date_list)].reset_index(drop=True)
        # display(df)
    # print(len(df))

    df["test_no"] = df["date"].astype(str) + "_" + df["test_no"].astype(str).str.replace("TEST","") + "_" + df['route'].astype(str)

    df["Band"] = df["Freq"].map(band_map)
    df.drop(columns=["Freq"], inplace=True)

    df['DL_Tput_per_RB'] = df['DL_Tput']/df['DL_RB']
    df['DL_Tput_full_RB'] = df['DL_Tput_per_RB'] * 52

    df = df.sort_values(by="TIME", ascending=True)
    df.reset_index(drop=True, inplace=True)
    # print(len(df))

    time_counts = df["TIME"].value_counts()
    valid_times = time_counts[time_counts == 2].index
    df = df[df["TIME"].isin(valid_times)]
    # print(len(df))

    valid_pairs = [(1, 2), (11, 12), (21, 22)]
    time_pairs = (
        df.groupby("TIME")["PCI"]
        .apply(lambda s: tuple(sorted(s.tolist())))
        .reset_index()
    )
    valid_times = time_pairs[
        time_pairs["PCI"].isin(valid_pairs)
    ]["TIME"]
    df = df[df["TIME"].isin(valid_times)].drop(columns=["PCI"])
    # print(len(df))

    uhd_lat, uhd_lon = 37.551179, 126.987671
    R = 6371000  # 지구 반지름 (m)
    lat1 = np.radians(uhd_lat)
    lon1 = np.radians(uhd_lon)
    lat2 = np.radians(df["Lat"])
    lon2 = np.radians(df["Lon"])
    df["Distance"] = R * 2 * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2 +
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    # print("dropna", len(df))

    new_order = [
        "TIME",
        "date",
        "route",
        "test_no",
        "Lon", "Lat",
        "Distance",
        "Band",
        "RSRP", "RSRQ",
        "SINR",
        "SINR_SSB",
        "SINR_TRS",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
        "DL_RB", "DL_Tput",
        "DL_Tput_per_RB",
        "DL_Tput_full_RB",
    ]
    df = df[new_order]

    return df

def grid_kpi(df, grid_size, rb_min, sample_min):

    kpi_cols = [
        "RSRP", "RSRQ",
        "SINR", "SINR_TRS",
        "CQI", "RI", "DL_MCS",
        "DL_BLER", "UL_BLER",
        "DL_RB", "DL_Tput",
        "DL_Tput_per_RB",
        # "DL_Tput_full_RB",
    ]
    df = df[df["DL_RB"]>rb_min].copy()
    df = df.dropna(subset=["Lat", "Lon", "Band"])

    lat_factor, lon_factor = 111320, 88000
    df["lat_bin"] = (df["Lat"] * lat_factor // grid_size).astype(int)
    df["lon_bin"] = (df["Lon"] * lon_factor // grid_size).astype(int)
    df = df.drop(columns=["Lat", "Lon"])

    df_n26 = df[df["Band"] == "n26"].copy()
    df_n28 = df[df["Band"] == "n28"].copy()

    df_diff = pd.merge(
        df_n26,
        df_n28,
        on=["TIME", "lat_bin", "lon_bin"],
        suffixes=("_n26", "_n28"),
        how="inner",
    )
    diff_records = []
    for _, row in df_diff.iterrows():
        record = {
            "TIME": row["TIME"],
            "lat_bin": row["lat_bin"],
            "lon_bin": row["lon_bin"],
        }
        for kpi in kpi_cols:
            c26, c28 = f"{kpi}_n26", f"{kpi}_n28"
            if (
                    c26 in row
                    and c28 in row
                    and not pd.isna(row[c26])
                    and not pd.isna(row[c28])
            ):
                record[kpi] = row[c28] - row[c26]
        diff_records.append(record)
    df_diff = pd.DataFrame(diff_records)

    def grid_stats(df, suffix):
        df_stat = (
            df.groupby(["lat_bin", "lon_bin"])[kpi_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_stat.columns = [
            f"{col[0]}_{col[1]}_{suffix}" if col[1] else col[0]
            for col in df_stat.columns.to_flat_index()
        ]
        df_count = (
            df.groupby(["lat_bin", "lon_bin"])["TIME"]
            .count()
            .reset_index()
            .rename(columns={"TIME": f"sample_count_{suffix}"})
        )
        df_stat = pd.merge(df_stat, df_count, on=["lat_bin", "lon_bin"], how="left")
        if "test_no" in df.columns:
            df_tests = (
                df.groupby(["lat_bin", "lon_bin"])
                .agg({"test_no": lambda x: list(x.unique())})
                .reset_index()
                .rename(columns={"test_no": f"test_list_{suffix}"})
            )
            df_stat = pd.merge(df_stat, df_tests, on=["lat_bin", "lon_bin"], how="left")
        return df_stat

    df_n26_stat = grid_stats(df_n26, "n26")
    df_n28_stat = grid_stats(df_n28, "n28")
    df_diff_stat = grid_stats(df_diff, "diff")

    df_pair = df_n26_stat.merge(df_n28_stat, on=["lat_bin", "lon_bin"], how="outer")
    df_pair = df_pair.merge(df_diff_stat, on=["lat_bin", "lon_bin"], how="outer")

    count_cols = [c for c in df_pair.columns if c.startswith("sample_count_")]
    df_pair[count_cols] = df_pair[count_cols].fillna(0).astype(int)

    def merge_test_lists(row):
        list1 = row.get("test_list_n26", [])
        if not isinstance(list1, list):
            list1 = []
        list2 = row.get("test_list_n28", [])
        if not isinstance(list2, list):
            list2 = []
        merged = sorted(set(list1) | set(list2))
        return merged
    df_pair["test_list"] = df_pair.apply(merge_test_lists, axis=1)
    df_pair = df_pair.drop(columns=["test_list_n26", "test_list_n28"], errors="ignore")

    df_uhd = read_UHD(uhd_dir='UHD_power')
    df_uhd_grid = grid_uhd(df_uhd, grid_size=grid_size)
    df_pair = pd.merge(df_pair, df_uhd_grid, on=["lat_bin", "lon_bin"], how="left")

    df_pair = df_pair[
        (df_pair["sample_count_n26"] >= sample_min)
        & (df_pair["sample_count_n28"] >= sample_min)
    ].reset_index(drop=True)

    df_pair = df_pair.sort_values(["lat_bin", "lon_bin"], ascending=[True, True])
    df_pair = df_pair.reset_index().rename(columns={"index": "loc_id"})

    common_cols = ["loc_id", "lat_bin", "lon_bin", "test_list"]
    sample_cols = [c for c in ["sample_count_n26", "sample_count_n28", "sample_count_diff"] if c in df_pair.columns]
    metric_cols = []
    for kpi in kpi_cols:
        for stat in ["mean", "std"]:
            for band in ["n26", "n28", "diff"]:
                col = f"{kpi}_{stat}_{band}"
                if col in df_pair.columns:
                    metric_cols.append(col)
    uhd_cols = [c for c in ["uhd_cnt", "uhd_avg", "uhd_max", "uhd_min"] if c in df_pair.columns]
    ordered_cols = common_cols + sample_cols + metric_cols + uhd_cols
    df_pair = df_pair[[c for c in ordered_cols if c in df_pair.columns]]

    # print(len(df_pair))
    # print(df_pair.info())
    # display(df_pair)

    return df_pair

# def grid_kpi(df, grid_size, rb_min, sample_min):
#
#     lat_factor, lon_factor = 111320, 88000
#
#     df_grid = df[df["DL_RB"]>rb_min].copy()
#
#     df_grid["lat_bin"] = (df_grid["Lat"] * lat_factor // grid_size).astype(int)
#     df_grid["lon_bin"] = (df_grid["Lon"] * lon_factor // grid_size).astype(int)
#     df_grid = df_grid.drop(columns=["Lat", "Lon"])
#
#     kpi_cols = [
#         "RSRP", "RSRQ",
#         "SINR", "SINR_TRS",
#         "CQI", "RI", "DL_MCS",
#         "DL_BLER", "UL_BLER",
#         "DL_RB", "DL_Tput",
#         "DL_Tput_per_RB",
#         "DL_Tput_full_RB",
#     ]
#
#     df_stats = (
#         df_grid.groupby(["lat_bin", "lon_bin", "Band"])[kpi_cols]
#         .agg(["mean", "std"])
#         .reset_index()
#     )
#     df_stats.columns = [
#         f"{col}_{stat}" if stat else col
#         for col, stat in df_stats.columns.to_flat_index()
#     ]
#
#     df_count = (
#         df_grid.groupby(["lat_bin", "lon_bin", "Band"])
#           .size()
#           .reset_index(name="sample_count")
#     )
#     df_tests = (
#         df_grid.groupby(["lat_bin", "lon_bin", "Band"])
#         .agg({"test_no": lambda x: list(x.unique())})
#         .reset_index()
#         .rename(columns={"test_no": "test_list"})
#     )
#
#     df_grid = (
#         df_stats
#         .merge(df_count, on=["lat_bin", "lon_bin", "Band"], how="left")
#         .merge(df_tests, on=["lat_bin", "lon_bin", "Band"], how="left")
#     )
#
#     df_pair = (
#         df_grid.pivot(
#             index=["lat_bin", "lon_bin"],
#             columns="Band",
#             values=df_grid.columns.drop(["lat_bin", "lon_bin", "Band"])
#         ).reset_index()
#     )
#
#     df_pair.columns = [
#         f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
#         for col in df_pair.columns.to_flat_index()
#     ]
#     # print(len(df_pair))
#     # display(df_pair[df_pair.isna().any(axis=1)])
#     df_pair = df_pair.dropna()
#     # print(len(df_pair))
#
#     df_pair["test_list"] = df_pair["test_list_n26"]
#     df_pair = df_pair.drop(columns=["test_list_n26", "test_list_n28"], errors="ignore")
#
#     df_uhd = read_UHD(uhd_dir='UHD_power')
#     df_uhd_grid = grid_uhd(df_uhd, grid_size=grid_size)
#     df_pair = pd.merge(df_pair, df_uhd_grid, on=["lat_bin", "lon_bin"], how="left")
#
#     df_pair = df_pair[
#         (df_pair["sample_count_n26"] >= sample_min)
#         & (df_pair["sample_count_n28"] >= sample_min)
#     ].reset_index(drop=True)
#     df_pair = df_pair.sort_values(["lat_bin", "lon_bin"], ascending=[True, True])
#     df_pair = df_pair.reset_index().rename(columns={"index": "loc_id"})
#
#     # display(df_pair)
#     # print(len(df_pair))
#     return df_pair

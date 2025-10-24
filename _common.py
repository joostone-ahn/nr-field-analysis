import os
import pandas as pd

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

        for route in route_list:
            filtered = merged[merged["route"] == route]
            filtered_path = os.path.join(log_dir, f"{device}_{route}.xlsx")
            filtered.to_excel(filtered_path, index=False)
            print(f"✅ Saved: {filtered_path}")

def analyze_kpi(fname, date_list, rb_min):
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

    df["test_no"] = df["date"].astype(str) + "_" + df['route'].astype(str) + "_" + df["test_no"].astype(str).str.replace("TEST","")
    df = df.drop(columns=["date"])

    df["Band"] = df["Freq"].map(band_map)
    df.drop(columns=["Freq"], inplace=True)

    df['DL_Tput_per_RB'] = df['DL_Tput']/df['DL_RB']
    df['DL_Tput_full_RB'] = df['DL_Tput_per_RB'] * 52
    
    new_order = [
        "TIME",
        "test_no",
        "Lon", "Lat",
        "Band", "PCI", 
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

    # rb_check = (
    #     df.groupby("TIME")["DL_RB"]
    #       .apply(lambda s: (s >= rb_min).all())
    # )
    # df = df[df["TIME"].isin(rb_check[rb_check].index)]
    # print(len(df))

    df = df[df['DL_RB'] >= rb_min]
    # print(len(df))
        
    df = df.dropna()
    # print(len(df))
    
    df.reset_index(drop=True, inplace=True)

    return df

def grid_kpi(df, grid_size):

    lat_factor, lon_factor = 111320, 88000
    
    # # --- ① 기준 offset 설정 (동쪽으로 10m 이동) ---
    # lon_offset_m = 10  # 10m 이동
    # lon_offset_deg = lon_offset_m / lon_factor  # 약 0.0001136도
    
    df_grid = df.copy()

    df_grid["lat_bin"] = (df_grid["Lat"] * lat_factor // grid_size).astype(int)
    df_grid["lon_bin"] = (df_grid["Lon"] * lon_factor // grid_size).astype(int)
    # df_grid["lon_bin"] = ((df_grid["Lon"] + lon_offset_deg) * lon_factor // grid_size).astype(int)
    
    df_grid = df_grid.drop(columns=["Lat", "Lon"])

    df_mean = (
        df_grid.groupby(["lat_bin", "lon_bin", "Band"])
          .mean(numeric_only=True)
          .reset_index()
    )
    df_count = (
        df_grid.groupby(["lat_bin", "lon_bin", "Band"])
          .size()
          .reset_index(name="sample_count")
    )
    
    df_grid = pd.merge(df_mean, df_count, on=["lat_bin", "lon_bin", "Band"])

    df_grid["loc_id"] = df_grid.groupby(["lat_bin", "lon_bin"]).ngroup()
    df_grid = df_grid[df_grid.groupby("loc_id")["loc_id"].transform("count") == 2]
    
    cols = ["loc_id", "lat_bin", "lon_bin", "Band"]
    others = [c for c in df_grid.columns if c not in cols]
    df_grid = df_grid[cols+others]

    df_grid = df_grid.reset_index(drop=True)
    # display(df_grid)

    kpi_cols = [
        "RSRP", "RSRQ", 
        "SINR", "SINR_TRS",
        "CQI", "RI", "DL_MCS", 
        "DL_BLER", "UL_BLER", 
        "DL_RB", "DL_Tput", 
        "DL_Tput_per_RB", 
        "DL_Tput_full_RB",
    ]
    
    df_pair = (
        df_grid.pivot(index=["loc_id", "lat_bin", "lon_bin"], columns="Band", values=[*kpi_cols, "sample_count"])
        .reset_index()
    )
    df_pair.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
        for col in df_pair.columns.to_flat_index()
    ]
    df_pair = df_pair.reset_index(drop=True)
    # display(df_pair)

    df_uhd = read_UHD(uhd_dir='UHD_power')
    df_uhd_grid = grid_uhd(df_uhd, grid_size=grid_size)
    df_pair = pd.merge(df_pair, df_uhd_grid, on=["lat_bin", "lon_bin"], how="left")

    # display(df_pair)
    return df_pair

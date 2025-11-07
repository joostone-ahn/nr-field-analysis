from datetime import datetime
import os

updated_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html_header = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NR Field Statistical Analysis (n26/n28)</title>
<style>
body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 28px;
    background-color: #fafafa;
}}
.header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}}
.header h1 {{
    color: #1a1a1a;
    font-size: 1.9em;
    font-weight: 600;
    margin: 0;
}}
.header span.updated {{
    font-size: 0.9em;
    color: #777;
}}
ul {{ list-style-type: none; padding-left: 20px; }}
li {{ margin: 6px 0; }}
a {{ text-decoration: none; color: #0056b3; }}
a:hover {{ text-decoration: underline; color: #003366; }}
summary {{ font-weight: bold; color: #222; cursor: pointer; margin-top: 8px; }}
.folder {{ margin-top: 6px; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 22px 0; }}
.section-title {{
    font-size: 1.05em;
    color: #1a3b7a;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 6px;
}}
table.comparison {{
    border-collapse: collapse;
    width: 80%;
    background-color: #f9fbff;
    border: 1px solid #d0e2ff;
    border-radius: 6px;
    font-size: 0.85em;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
table.comparison th {{
    background-color: #e3eeff;
    color: #1a3b7a;
    text-align: left;
    padding: 5px 10px;
    font-weight: 600;
}}
table.comparison td {{
    padding: 5px 10px;
    vertical-align: top;
    color: #222;
    line-height: 1.3;
}}
table.comparison tr:nth-child(even) td {{
    background-color: #f4f7ff;
}}
</style>
</head>
<body>

<div class="header">
  <h1>NR Field Statistical Analysis</h1>
  <span class="updated">Updated: {updated_str}</span>
</div>
<hr>
"""

html_footer = """
<div class="section-title">📊 Comparison by Map Grid Resolution</div>
<table class="comparison">
  <tr>
    <th>항목</th>
    <th>Map Grid Size = 30m</th>
    <th>Map Grid Size = 5m</th>
  </tr>
  <tr>
    <td><b>분석 목적</b></td>
    <td>GPS 오차 보정 및 통계적 안정성 확보, 전체 커버리지 비교 분석에 적합</td>
    <td>세부 위치별 커버리지·간섭 경향 분석에 적합</td>
  </tr>
  <tr>
    <td><b>이동 조건</b></td>
    <td>이동 속도 약 36 km/h (≈10 m/s) 기준, 30m 구간당 약 3개 샘플 수집</td>
    <td>이동 속도 약 36 km/h (≈10 m/s) 기준, 5m 구간당 약 0.5개 샘플 수집</td>
  </tr>
  <tr>
    <td><b>반복 주행 시 샘플 확보량</b></td>
    <td>동일 루트를 10회 이상 반복 시 grid별 약 30개 샘플 확보 가능</td>
    <td>동일 루트를 10회 이상 반복 시 grid별 약 5개 샘플 확보 가능</td>
  </tr>
  <tr>
    <td><b>통계적 해석</b></td>
    <td>샘플 수 30개 이상 확보 시 중심극한정리(CLT)에 따른 정규분포 근사 가능</td>
    <td>샘플 수가 적어 통계적 안정성은 낮으나 지역적 패턴 탐지에 유용</td>
  </tr>
  <tr>
    <td><b>적용 지표</b></td>
    <td>SINR, Tput 등 고분산 지표의 신뢰구간(CI) 추정에 적합</td>
    <td>RSRP, RSRQ 등 세밀한 위치 기반 비교에 유용</td>
  </tr>
</table>
</body></html>
"""

def generate_list_html(root_dir, depth=0):
    """'map', 'map/grid_30m', 'plot', 'plot/raw', 'plot/raw/rsrp_1dB' 폴더만 기본 펼침"""
    items = sorted(os.listdir(root_dir))
    html = "<ul>"

    for item in items:
        if item == "index.html":
            continue

        path = os.path.join(root_dir, item)
        rel_path = os.path.relpath(path, "results").replace("\\", "/")

        if os.path.isdir(path):
            is_open = rel_path in (
                "map",
                "plot",
                "dist",
            )

            html += (
                f'<li class="folder"><details {"open" if is_open else ""}><summary>{item}/</summary>'
            )
            html += generate_list_html(path, depth + 1)
            html += "</details></li>"

        elif item.endswith((".html", ".png")):
            href = f"results/{rel_path}"
            html += f'<li><a href="{href}" target="_blank">{item}</a></li>'

    html += "</ul>"
    return html

# 실행부
result_dir = "results"
index_file = "index.html"

with open(index_file, "w", encoding="utf-8") as f:
    f.write(html_header)
    f.write(generate_list_html(result_dir))
    # f.write(html_footer)

print(f"✅ index.html generated at: {index_file}")
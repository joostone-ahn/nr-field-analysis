import os

base_dir = "results"
index_path = os.path.join(base_dir, "index.html")

html_header = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NR Field Analysis Results</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }
h2 { color: #333; }
ul { list-style-type: none; padding-left: 20px; }
li { margin: 6px 0; }
a { text-decoration: none; color: #0066cc; }
a:hover { text-decoration: underline; color: #003366; }
summary { font-weight: bold; color: #222; cursor: pointer; margin-top: 8px; }
.folder { margin-top: 6px; }
</style>
</head>
<body>
<h1>NR Field Analysis Results (n26/n28)</h1>
<hr>
"""

html_footer = "</body></html>"

def generate_list_html(root_dir, parent_dir=None):
    items = sorted(os.listdir(root_dir))
    html = "<ul>"

    for item in items:
        path = os.path.join(root_dir, item)
        rel_path = os.path.relpath(path, base_dir).replace("\\", "/")

        if os.path.isdir(path):
            # ✅ 접힘 조건: 상위 폴더 이름이 kpi_each_test 또는 map_으로 시작할 경우
            parent_name = os.path.basename(parent_dir or "")
            if parent_name == "kpi_each_test" or os.path.basename(root_dir).startswith("map_"):
                fold_state = ""  # 접힘
            else:
                fold_state = " open"  # 펼침

            html += f'<li class="folder"><details{fold_state}><summary>{item}/</summary>'
            html += generate_list_html(path, parent_dir=path)  # <-- 수정된 부분
            html += "</details></li>"

        elif item.endswith(".html") or item.endswith(".png"):
            html += f'<li><a href="{rel_path}" target="_blank">{os.path.basename(rel_path)}</a></li>'

    html += "</ul>"
    return html


os.makedirs(base_dir, exist_ok=True)

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html_header)
    f.write(generate_list_html(base_dir))
    f.write(html_footer)

print(f"✅ index.html generated (map/date folders collapsed) at: {index_path}")
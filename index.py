import os
from datetime import datetime

base_dir = "results"
index_path = os.path.join(base_dir, "index.html")

# 현재 날짜/시간 포맷
updated_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html_header = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NR Field Analysis Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
h1 {{ color: #333; }}
h1 span.updated {{ font-size: 0.6em; color: #666; margin-left: 10px; }}
h2 {{ color: #333; }}
ul {{ list-style-type: none; padding-left: 20px; }}
li {{ margin: 6px 0; }}
a {{ text-decoration: none; color: #0066cc; }}
a:hover {{ text-decoration: underline; color: #003366; }}
summary {{ font-weight: bold; color: #222; cursor: pointer; margin-top: 8px; }}
.folder {{ margin-top: 6px; }}
</style>
</head>
<body>
<h1>NR Field Analysis Results (n26/n28)<span class="updated">Updated: {updated_str}</span></h1>
<hr>
"""

html_footer = "</body></html>"


def generate_list_html(root_dir, depth=0):
    items = sorted(os.listdir(root_dir))
    html = "<ul>"

    for item in items:
        if item == "index.html":
            continue

        path = os.path.join(root_dir, item)
        rel_path = os.path.relpath(path, base_dir).replace("\\", "/")

        if os.path.isdir(path):
            fold_state = ""
            if depth == 0:
                fold_state = " open"
            html += f'<li class="folder"><details{fold_state}><summary>{item}/</summary>'
            html += generate_list_html(path, depth + 1)
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

print(f"✅ index.html generated at: {index_path}")

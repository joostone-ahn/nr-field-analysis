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
li { margin: 4px 0; }
a { text-decoration: none; color: #0066cc; }
a:hover { text-decoration: underline; color: #003366; }
.folder { font-weight: bold; color: #333; }
</style>
</head>
<body>
<h1>NR Field Analysis Results</h1>
<hr>
"""

html_footer = "</body></html>"

def generate_list_html(root_dir, base_url=""):
    html = "<ul>"
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        rel_path = os.path.relpath(path, base_dir).replace("\\", "/")
        if os.path.isdir(path):
            html += f'<li class="folder">{item}/</li>'
            html += generate_list_html(path, base_url + item + "/")
        elif item.endswith(".html"):
            html += f'<li><a href="{rel_path}" target="_blank">{rel_path}</a></li>'
    html += "</ul>"
    return html

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html_header)
    f.write(generate_list_html(base_dir))
    f.write(html_footer)

print(f"✅ index.html generated at: {index_path}")

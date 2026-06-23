#!/usr/bin/env python3
"""Convert markdown to PDF using Playwright."""

import markdown
import sys
import os
from playwright.sync_api import sync_playwright

md_file = sys.argv[1] if len(sys.argv) > 1 else "experiment-report.md"
out_file = md_file.replace(".md", ".pdf")

with open(md_file, "r", encoding="utf-8") as f:
    md_content = f.read()

html_content = markdown.markdown(
    md_content,
    extensions=["fenced_code", "tables", "toc", "nl2br"],
    extension_configs={"toc": {"permalink": True, "title": "目录"}}
)

full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>基于 OpenFace 3.0 的面部动作编码系统模式识别实验报告</title>
<style>
  @page {{ size: A4; margin: 25mm 20mm; }}
  body {{
    font-family: "PingFang SC", "Noto Sans SC", "Source Han Sans SC", sans-serif;
    font-size: 11pt; line-height: 1.7; color: #333; max-width: 100%;
  }}
  h1 {{
    font-family: "Noto Serif SC", "Songti SC", serif; font-size: 22pt;
    font-weight: 700; color: #1A1A2E; text-align: center;
    margin: 30pt 0 10pt 0; page-break-before: always;
  }}
  h1:first-of-type {{ page-break-before: avoid; margin-top: 0; }}
  h2 {{
    font-family: "Noto Serif SC", serif; font-size: 16pt; font-weight: 700;
    color: #2D4A7A; margin-top: 24pt; margin-bottom: 10pt;
    padding-bottom: 6pt; border-bottom: 1pt solid #E8E6DF; page-break-after: avoid;
  }}
  h3 {{
    font-family: "Noto Serif SC", serif; font-size: 13pt; font-weight: 600;
    color: #1A1A2E; margin-top: 16pt; margin-bottom: 6pt; page-break-after: avoid;
  }}
  h4 {{ font-size: 11pt; font-weight: 600; color: #444; margin-top: 12pt; margin-bottom: 4pt; }}
  p {{ margin-bottom: 8pt; text-align: justify; }}
  ul, ol {{ margin: 8pt 0; padding-left: 20pt; }}
  li {{ margin-bottom: 4pt; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12pt 0; font-size: 10pt; }}
  th {{ background: #2D4A7A; color: #fff; padding: 8pt 10pt; text-align: left; font-weight: 600; }}
  td {{ padding: 6pt 10pt; border-bottom: 1pt solid #E8E6DF; }}
  tr:nth-child(even) td {{ background: #F8F7F3; }}
  code {{ font-family: "SFMono-Regular", monospace; font-size: 9.5pt; background: #F2F1EC; padding: 1pt 5pt; border-radius: 2pt; }}
  pre {{ background: #F8F7F3; border-left: 3pt solid #D4A843; padding: 12pt 16pt; overflow-x: auto; margin: 12pt 0; page-break-inside: avoid; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 3pt solid #D4A843; margin: 12pt 0; padding: 8pt 16pt; background: #F5F0E0; color: #555; }}
  hr {{ border: none; border-top: 1pt solid #E0DDD5; margin: 20pt 0; }}
</style>
</head>
<body>{html_content}</body>
</html>"""

html_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(md_file)), "_temp_report.html"))
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(f"file://{html_path}")
    page.pdf(path=out_file, format="A4", margin={"top": "25mm", "right": "20mm", "bottom": "25mm", "left": "20mm"})
    browser.close()

os.remove(html_path)
print(f"PDF 已生成: {out_file}")

# Puts the numbers from data.json into ../index.html (replaces the `const D=...;` line).
# Run after build_data.py:  python update_page.py
import re, pathlib
DIR = pathlib.Path(__file__).resolve().parent
data = (DIR / "data.json").read_text(encoding="utf-8")
page = DIR.parent / "index.html"
html = page.read_text(encoding="utf-8")
new, n = re.subn(r"const D=\{.*?\};\n(?=const TC)", lambda m: "const D=" + data + ";\n", html, count=1, flags=re.S)
assert n == 1, "Could not find the const D line in index.html"
page.write_text(new, encoding="utf-8")
print("index.html updated")

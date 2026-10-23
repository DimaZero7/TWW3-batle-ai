"""Copy design/general/tasks.json into index.html (the page opens from file://, where fetch is blocked).

The copy sits between the markers <!-- tasks-data:begin --> and <!-- tasks-data:end -->.
Run after every edit of tasks.json:  python design/general/sync_tasks.py
--check: exit 1 if the page copy differs from tasks.json, change nothing.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BEGIN = "<!-- tasks-data:begin -->"
END = "<!-- tasks-data:end -->"


def block(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2).replace("</", "<\\/")
    return f'{BEGIN}\n<script type="application/json" id="tasks-data">\n{text}\n</script>\n{END}'


def main(argv: list[str]) -> int:
    data = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    for key in ("in_progress", "planned", "closed", "parked"):
        if not isinstance(data.get(key), list):
            raise SystemExit(f"tasks.json: list '{key}' missing")
    ids = [t["id"] for key in ("in_progress", "planned", "closed", "parked") for t in data[key]]
    if len(ids) != len(set(ids)):
        raise SystemExit("tasks.json: duplicate id")
    data["planned"].sort(key=lambda t: t.get("order", 1e9))
    page = HERE / "index.html"
    html = page.read_bytes().decode("utf-8")
    a, b = html.find(BEGIN), html.find(END)
    if a < 0 or b < a:
        raise SystemExit("index.html: markers not found")
    new = html[:a] + block(data) + html[b + len(END):]
    if "--check" in argv:
        print("in sync" if new == html else "OUT OF SYNC")
        return 0 if new == html else 1
    if new != html:
        page.write_bytes(new.encode("utf-8"))
        print("index.html updated")
    else:
        print("index.html already in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

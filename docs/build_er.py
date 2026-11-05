"""Рисует ER-диаграмму docs/er_diagram.png напрямую из sql/schema.sql: схема и картинка не расходятся.

Запуск: python docs/build_er.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "sql" / "schema.sql"
OUTPUT = ROOT / "docs" / "er_diagram.png"

INK, MUTED, HEADER, BODY, EDGE = "#14342b", "#5b6b65", "#14342b", "#f3f7f5", "#9db3aa"
ROW, TITLE_ROW, WIDTH = 0.34, 0.44, 3.5

# Положение левого верхнего угла каждой таблицы (x, y)
LAYOUT = {
    "dim_grade": (0.2, 17.0), "dim_purpose": (4.3, 17.0), "dim_state": (8.4, 17.0),
    "loans": (4.3, 15.3),
    "borrowers": (0.2, 10.0), "dim_home": (4.3, 8.0), "outcomes": (8.4, 10.0),
}
# (родитель, потомок, подпись, сторона родителя, сторона потомка)
RELATIONS = [
    ("dim_grade", "loans", "1 : N  sub_grade", "bottom", "left"),
    ("dim_purpose", "loans", "1 : N  purpose_id", "bottom", "top"),
    ("dim_state", "loans", "1 : N  state_code", "bottom", "right"),
    ("loans", "borrowers", "1 : 1  loan_id", "bottom", "top"),
    ("loans", "outcomes", "1 : 1  loan_id", "bottom", "top"),
    ("dim_home", "borrowers", "1 : N  home_id", "left", "right"),
]


def parse_schema(text: str) -> dict:
    """{таблица: [(колонка, тип, 'PK'/'FK'/'PK, FK'/'')]} из DDL."""
    tables = {}
    for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\n\);", text, flags=re.S):
        columns = []
        for line in body.splitlines():
            line = line.split("--")[0].strip().rstrip(",")
            if not line:
                continue
            parts = line.split()
            keys = [k for k, marker in (("PK", "PRIMARY KEY"), ("FK", "REFERENCES")) if marker in line]
            columns.append((parts[0], parts[1], ", ".join(keys)))
        tables[name] = columns
    return tables


def table_height(columns: list) -> float:
    return TITLE_ROW + ROW * len(columns)


def draw_table(ax: plt.Axes, name: str, columns: list) -> None:
    x, top = LAYOUT[name]
    height = table_height(columns)
    ax.add_patch(FancyBboxPatch((x, top - height), WIDTH, height, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=BODY, ec=EDGE, lw=1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, top - TITLE_ROW), WIDTH, TITLE_ROW, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=HEADER, ec=HEADER, lw=1, zorder=3))
    ax.text(x + WIDTH / 2, top - TITLE_ROW / 2, name, ha="center", va="center", color="white", weight="bold",
            fontsize=11, zorder=4)
    for i, (col, kind, key) in enumerate(columns):
        y = top - TITLE_ROW - ROW * (i + 0.5)
        weight = "bold" if key else "normal"
        ax.text(x + 0.12, y, col, va="center", color=INK, fontsize=9.5, weight=weight, zorder=4)
        ax.text(x + WIDTH - 0.12, y, f"{key}  {kind}".strip(), va="center", ha="right", color=MUTED, fontsize=8.2,
                zorder=4)


def anchor(name: str, columns: list, side: str) -> tuple:
    x, top = LAYOUT[name]
    height = table_height(columns)
    return {"bottom": (x + WIDTH / 2, top - height), "top": (x + WIDTH / 2, top),
            "left": (x, top - height / 2), "right": (x + WIDTH, top - height / 2)}[side]


def draw_relation(ax: plt.Axes, tables: dict, relation: tuple) -> None:
    parent, child, label, parent_side, child_side = relation
    start, end = anchor(parent, tables[parent], parent_side), anchor(child, tables[child], child_side)
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.2, "shrinkA": 0, "shrinkB": 0})
    ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, label, ha="center", va="center", fontsize=8.5,
            color=INK, zorder=5, bbox={"fc": "white", "ec": "none", "pad": 1.5})


def build() -> None:
    tables = parse_schema(SCHEMA.read_text(encoding="utf-8"))
    missing = set(tables) ^ set(LAYOUT)
    if missing:
        raise ValueError(f"Схема и раскладка расходятся: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(11, 12.2))
    ax.set(xlim=(0, 12.1), ylim=(3.9, 17.2))
    ax.axis("off")
    for relation in RELATIONS:
        draw_relation(ax, tables, relation)
    for name, columns in tables.items():
        draw_table(ax, name, columns)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Готово: {OUTPUT}")


if __name__ == "__main__":
    build()

"""Собирает docs/guide.docx — пояснительную записку к ноутбуку project_walkthrough.ipynb.

Запуск из корня проекта после выполнения ноутбука:  python docs/build_guide.py
                                                   python docs/build_guide.py --with-defense
Второй вариант добавляет приложение с вопросами к защите из docs/defense_prep.md (личная шпаргалка автора, в
репозиторий не входит) и пишет docs/guide_full.docx.
Документ повторяет ноутбук по порядку: текст ячеек, код, подробное объяснение (docs/guide_text.py) и результаты —
таблицы и графики из сохранённых выводов ноутбука. В конце: справочник модулей, вопросы к защите, глоссарий.
"""
import ast
import base64
import io
import re
import sys
from pathlib import Path

import nbformat
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import Report, add_field, footer_page_numbers, rgb, set_font, setup_styles  # noqa: E402
from guide_text import ARCHITECTURE, CAPTIONS, EXPLAIN, GLOSSARY, INTRO, OVERVIEW  # noqa: E402
from results import AUTHORS, COURSE, GROUP, placeholders_left  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "project_walkthrough.ipynb"
DEFENSE = ROOT / "docs" / "defense_prep.md"
OUTPUT = ROOT / "docs" / "guide.docx"
OUTPUT_FULL = ROOT / "docs" / "guide_full.docx"
CODE_FONT = "Consolas"
MAX_OUTPUT_LINES = 28
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


# --------------------------------------------------------------------------------------------- оформление
def shade_paragraph(paragraph, color: str) -> None:
    """Заливка фона абзаца: так блок кода отделяется от текста."""
    props = paragraph._p.get_or_add_pPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def add_inline(paragraph, text: str) -> None:
    """Добавляет текст с разметкой Markdown: **жирный**, `код`, [ссылка](адрес)."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_font(run, CODE_FONT)
            run.font.size = Pt(9.5)
        elif part.startswith("["):
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part).groups()
            paragraph.add_run(f"{label} ({url})")
        else:
            paragraph.add_run(part)


def monospace_block(doc: Document, text: str, size: float, fill: str) -> None:
    """Блок моноширинного текста: код или текстовый вывод ячейки."""
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(6)
    para.paragraph_format.line_spacing = 1.0
    shade_paragraph(para, fill)
    lines = text.rstrip("\n").split("\n")
    for i, line in enumerate(lines):
        run = para.add_run(line)
        set_font(run, CODE_FONT)
        run.font.size = Pt(size)
        if i < len(lines) - 1:
            run.add_break()


def label(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.keep_with_next = True
    run = para.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = rgb("green")


# --------------------------------------------------------------------------------------------- markdown
def markdown_table(rep: Report, lines: list) -> None:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    header, body = rows[0], [r for r in rows[2:]]
    clean = [[re.sub(r"\*\*|`", "", c) for c in row] for row in body]
    widths = [16.0 / len(header)] * len(header)
    if len(header) == 2:
        widths = [5.5, 10.5]
    elif len(header) == 3:
        widths = [3.5, 6.0, 6.5]
    rep.table([re.sub(r"\*\*|`", "", h) for h in header], clean, "Таблица из ноутбука", widths)


def render_markdown(rep: Report, text: str) -> None:
    """Простая конвертация Markdown ячейки ноутбука в абзацы, заголовки, списки и таблицы docx."""
    doc = rep.doc
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            markdown_table(rep, block)
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            rep.figure(ROOT / image.group(2), image.group(1), 14.0)
        elif line.startswith("# "):
            pass  # заголовок ноутбука уже на титульном листе
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif re.match(r"^\d+\. ", line):
            add_inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
        elif line.startswith("- "):
            add_inline(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            paragraph = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(\||#|- |\d+\. |!\[)", lines[i]):
                paragraph.append(lines[i].strip())
                i += 1
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(para, " ".join(paragraph))
            continue
        i += 1


# --------------------------------------------------------------------------------------------- ячейки кода
def render_outputs(rep: Report, cell, key: str) -> None:
    images, texts = [], []
    for output in cell.outputs:
        data = output.get("data", {})
        if "image/png" in data:
            images.append(base64.b64decode(data["image/png"]))
        elif output.output_type == "stream" and output.name == "stdout":
            texts.append(output.text)
        elif "text/plain" in data and not data["text/plain"].startswith("<IPython"):
            texts.append(data["text/plain"])  # таблицы DataFrame и словари
    text = "\n".join(t.rstrip("\n") for t in texts if t.strip())
    if text and not key.startswith("src_"):
        lines = text.split("\n")
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[:MAX_OUTPUT_LINES] + [f"... (ещё {len(lines) - MAX_OUTPUT_LINES} строк)"]
        lines = [line if len(line) <= 150 else line[:147] + "..." for line in lines]
        label(rep.doc, "Результат")
        monospace_block(rep.doc, "\n".join(lines), 7.0, "F7F7F5")
    for picture in images:
        para = rep.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(io.BytesIO(picture), width=Cm(15.5))
        rep.figure_no += 1
        caption = rep.doc.add_paragraph(f"Рисунок {rep.figure_no}. {CAPTIONS.get(key, 'Результат ячейки')}",
                                        style="Caption")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER


def render_code_cell(rep: Report, cell) -> None:
    key = cell.metadata.get("guide", "")
    label(rep.doc, "Код")
    monospace_block(rep.doc, cell.source, 8.0, "EEF3F0")
    explanation = EXPLAIN.get(key, "").strip()
    if explanation:
        label(rep.doc, "Что делает этот код")
        for paragraph in explanation.split("\n\n"):
            para = rep.doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(para, " ".join(paragraph.split()))
    if key.startswith("src_"):  # исходный код функции из src/ выведен блоком Markdown ```python ... ```
        text = "\n".join(o.get("data", {}).get("text/markdown", "") for o in cell.outputs)
        source = re.sub(r"^```python\n|```\s*$", "", text.strip()).strip("\n")
        if source:
            label(rep.doc, "Исходный код функции")
            monospace_block(rep.doc, source, 7.5, "F7F7F5")
    render_outputs(rep, cell, key)


# --------------------------------------------------------------------------------------------- справочник модулей
def module_reference(rep: Report) -> None:
    rep.doc.add_heading("Приложение А. Справочник модулей и функций", level=1)
    rep.p("Ниже перечислены все модули папки src/ и функции в них. Описание взято из строки документации функции "
          "(docstring), поэтому совпадает с кодом.")
    for path in sorted((ROOT / "src").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rep.doc.add_heading(f"src/{path.name}", level=2)
        module_doc = ast.get_docstring(tree)
        if module_doc:
            rep.p(" ".join(module_doc.split()))
        rows = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                doc = (ast.get_docstring(node) or "").split("\n\n")[0]
                kind = "класс" if isinstance(node, ast.ClassDef) else "функция"
                rows.append([node.name, kind, " ".join(doc.split())])
        if rows:
            rep.table(["Имя", "Вид", "Что делает"], rows, f"Функции модуля src/{path.name}", [4.2, 1.8, 10.0])


# --------------------------------------------------------------------------------------------- вопросы к защите
def defense_questions(rep: Report) -> None:
    text = DEFENSE.read_text(encoding="utf-8")
    section = text.split("## 4. Вопросы, которые стоит ожидать", 1)[1].split("## 5.", 1)[0]
    rep.doc.add_heading("Приложение Б. Вопросы на защите и ответы", level=1)
    rep.p("Вопросы, которые вероятнее всего зададут, и ответы с цифрами проекта.")
    blocks = [b.strip() for b in section.strip().split("\n\n") if b.strip()]
    for block in blocks:
        if block.startswith("**") and block.rstrip().endswith("**") and "\n" not in block:
            heading = rep.doc.add_paragraph()
            heading.paragraph_format.space_before = Pt(8)
            heading.paragraph_format.keep_with_next = True
            run = heading.add_run(block.strip("*"))
            run.bold = True
            run.font.color.rgb = rgb("pine")
            continue
        if block.startswith("**") and "\n" in block:
            question, answer = block.split("\n", 1)
            heading = rep.doc.add_paragraph()
            heading.paragraph_format.space_before = Pt(8)
            heading.paragraph_format.keep_with_next = True
            run = heading.add_run(question.strip("*"))
            run.bold = True
            run.font.color.rgb = rgb("pine")
            block = answer
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                add_inline(rep.doc.add_paragraph(style="List Bullet"), line[2:])
            else:
                para = rep.doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                add_inline(para, re.sub(r"^\*(.+?)\*", r"**\1**", line))


# --------------------------------------------------------------------------------------------- сборка
def title(doc: Document) -> None:
    for text, size, bold, before in [
        (COURSE, 12, False, 90), ("Итоговый проект, тема №33", 12, False, 6),
        ("Прогноз дефолта по займам LendingClub", 22, True, 60),
        ("Пояснительная записка к ноутбуку project_walkthrough.ipynb", 14, False, 12),
        ("Подробное объяснение данных, кода, графиков и результатов", 12, False, 6),
        (f"{AUTHORS}, {GROUP}", 12, False, 150),
    ]:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(before)
        run = para.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        if size > 13:
            run.font.color.rgb = rgb("pine")
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build(with_defense: bool = False) -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    missing = [c.metadata.get("guide") for c in notebook.cells
               if c.cell_type == "code" and c.metadata.get("guide") not in EXPLAIN]
    if missing:
        raise ValueError(f"Нет объяснения для ячеек: {missing}")
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin = section.bottom_margin = Cm(2)
    setup_styles(doc)
    doc.core_properties.title = "Пояснительная записка к ноутбуку"
    doc.core_properties.author = AUTHORS
    placeholders_left()
    footer_page_numbers(doc)
    title(doc)
    head = doc.add_paragraph()
    run = head.add_run("Содержание")
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = RGBColor(0x14, 0x34, 0x2B)
    add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    rep = Report(doc)
    doc.add_heading("Об этом документе", level=1)
    for paragraph in INTRO.strip().split("\n\n"):
        rep.p(" ".join(paragraph.split()))
    doc.add_heading("Суть проекта простыми словами", level=1)
    for heading, text in OVERVIEW:
        doc.add_heading(heading, level=2)
        rep.p(" ".join(text.split()))
    doc.add_heading("Из чего состоит проект", level=1)
    rep.table(["Файл", "Назначение"], [list(row) for row in ARCHITECTURE], "Структура проекта", [5.5, 10.5])
    rep.p("Порядок работы конвейера: CSV.gz → Parquet (чтение частями) → аудит → база SQLite и SQL-запросы → "
          "разведочный анализ → признаки и разбиение по времени → подбор параметров и обучение девяти моделей → "
          "калибровка, порог и выбор модели → анализ результатов → сравнение способов обработки данных.")

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            render_markdown(rep, cell.source)
        else:
            render_code_cell(rep, cell)

    rep.page_break()
    module_reference(rep)
    if with_defense:
        rep.page_break()
        defense_questions(rep)
    doc.add_heading(f"Приложение {'В' if with_defense else 'Б'}. Глоссарий", level=1)
    rep.table(["Термин", "Значение"], [list(row) for row in GLOSSARY], "Термины проекта", [4.0, 12.0])
    output = OUTPUT_FULL if with_defense else OUTPUT
    doc.save(output)
    print(f"Готово: {output}")
    return output


if __name__ == "__main__":
    build(with_defense="--with-defense" in sys.argv)

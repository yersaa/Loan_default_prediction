"""Собирает презентацию docs/presentation.pptx. Все числа читаются из reports/ через docs/results.py.

Запуск: python docs/build_deck.py
Графики нативные: их можно редактировать в PowerPoint.
"""
import re

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from results import AUTHOR, BRAND, COURSE, GROUP, ROOT, TITLE, Results, load, ru, thousands

OUTPUT = ROOT / "docs" / "presentation.pptx"
NUMERIC = re.compile(r"[\d\s ,.%—-]+")
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
GBM, LR, GRADE, GBM_B = "LightGBM", "Logistic Regression", "Grade only (LogReg)", "LightGBM (borrower only)"
THEME_COLORS = {"dk1": "ink", "lt1": "white", "dk2": "pine", "lt2": "tint", "accent1": "green", "accent2": "amber",
                "accent3": "rust", "accent4": "pine", "accent5": "muted", "accent6": "mint", "hlink": "green",
                "folHlink": "muted"}


def color(name: str) -> RGBColor:
    return RGBColor.from_string(BRAND[name])


def apply_theme(prs: Presentation) -> None:
    """Цвета и шрифты задаются в теме, поэтому оформление меняется в одном месте."""
    part = prs.slide_master.part.part_related_by(RT.THEME)
    xml = part.blob.decode("utf-8")
    for slot, brand in THEME_COLORS.items():
        xml = re.sub(rf"(<a:{slot}>).*?(</a:{slot}>)", rf'\g<1><a:srgbClr val="{BRAND[brand]}"/>\g<2>', xml, flags=re.S)
    xml = re.sub(r'(<a:majorFont>\s*<a:latin typeface=")[^"]*', rf"\g<1>{HEAD_FONT}", xml)
    xml = re.sub(r'(<a:minorFont>\s*<a:latin typeface=")[^"]*', rf"\g<1>{BODY_FONT}", xml)
    xml = re.sub(r'(<a:clrScheme name=")[^"]*', r"\g<1>LendingClub", xml)
    part._blob = xml.encode("utf-8")


def set_layout_placeholders(prs: Presentation) -> tuple:
    """Заголовки стоят на одном месте на всех слайдах: геометрия и шрифт заданы в макетах."""
    title_slide, title_only = prs.slide_layouts[0], prs.slide_layouts[5]
    for ph in title_only.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.left, ph.top, ph.width, ph.height = MARGIN, Inches(0.4), SLIDE_W - 2 * MARGIN, Inches(1.0)
            style_placeholder(ph, 30, "pine", MSO_ANCHOR.TOP)
    for ph in list(title_only.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    for ph in title_slide.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.left, ph.top, ph.width, ph.height = MARGIN, Inches(2.0), Inches(10.5), Inches(1.9)
            style_placeholder(ph, 46, "white", MSO_ANCHOR.BOTTOM)
        elif ph.placeholder_format.idx == 1:
            ph.left, ph.top, ph.width, ph.height = MARGIN, Inches(4.1), Inches(10.5), Inches(1.4)
            style_placeholder(ph, 18, "mint", MSO_ANCHOR.TOP)
        else:
            ph._element.getparent().remove(ph._element)
    add_slide_number(title_only)
    remove_unused_layouts(prs, keep=(title_slide, title_only))
    return title_slide, title_only


def style_placeholder(ph, size: int, tint: str, anchor) -> None:
    frame = ph.text_frame
    frame.vertical_anchor = anchor
    frame.word_wrap = True
    body = frame._txBody
    lst = body.find(qn("a:lstStyle"))
    for child in list(lst):
        lst.remove(child)
    level = etree.SubElement(lst, qn("a:lvl1pPr"), algn="l", marL="0", indent="0")
    etree.SubElement(level, qn("a:buNone"))
    run = etree.SubElement(level, qn("a:defRPr"), sz=str(size * 100), b="1" if size > 20 else "0")
    fill = etree.SubElement(run, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr"), val=BRAND[tint])


def add_slide_number(layout) -> None:
    """У макета нет add_textbox в python-pptx, поэтому поле номера слайда добавляется в XML макета."""
    left, top, width, height = SLIDE_W - Inches(1.2), SLIDE_H - Inches(0.5), Inches(0.7), Inches(0.3)
    xml = f"""<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:nvSpPr><p:cNvPr id="99" name="Номер слайда"/><p:cNvSpPr txBox="1"/><p:nvPr userDrawn="1"/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{left}" y="{top}"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/>
        <a:p><a:pPr algn="r"/><a:fld id="{{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}}" type="slidenum">
          <a:rPr lang="ru-RU" sz="1100"><a:solidFill><a:srgbClr val="{BRAND['muted']}"/></a:solidFill></a:rPr>
          <a:t>‹#›</a:t></a:fld></a:p></p:txBody></p:sp>"""
    layout.shapes._spTree.append(etree.fromstring(xml))


def remove_unused_layouts(prs: Presentation, keep: tuple) -> None:
    layouts = prs.slide_master.slide_layouts
    for layout in list(layouts):
        if layout not in keep:
            layouts.remove(layout)


class Deck:
    def __init__(self, prs: Presentation, layouts: tuple, r: Results):
        self.prs, self.title_layout, self.content_layout, self.r = prs, layouts[0], layouts[1], r

    def slide(self, title: str, notes: str):
        s = self.prs.slides.add_slide(self.content_layout)
        s.shapes.title.text = title
        s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def text(s, left, top, width, height, runs: list, size: int = 16, tint: str = "ink", bold: bool = False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name: str = "Текст"):
        """runs: абзацы; абзац — строка или список пар (текст, жирный)."""
        box = s.shapes.add_textbox(left, top, width, height)
        box.name = name
        frame = box.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = anchor
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        for i, paragraph in enumerate(runs):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.alignment = align
            para.space_after = Pt(8)
            for text, strong in ([(paragraph, bold)] if isinstance(paragraph, str) else paragraph):
                run = para.add_run()
                run.text = text
                run.font.size = Pt(size)
                run.font.bold = strong
                run.font.color.rgb = color(tint)
        return box

    @staticmethod
    def card(s, left, top, width, height, fill: str = "tint", name: str = "Карточка"):
        shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.name = name
        shape.adjustments[0] = 0.06
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(fill)
        shape.line.fill.background()
        shape.shadow.inherit = False
        return shape

    def stat(self, s, left, top, width, height, value: str, label: str, fill: str = "tint", ink: str = "pine",
             sub: str = "ink", value_size: int = 44, label_size: int = 15) -> None:
        self.card(s, left, top, width, height, fill, name=f"Показатель {value}")
        pad = Inches(0.25)
        self.text(s, left + pad, top + Inches(0.2), width - 2 * pad, Inches(0.95), [value], size=value_size, tint=ink,
                  bold=True, anchor=MSO_ANCHOR.MIDDLE)
        self.text(s, left + pad, top + Inches(1.2), width - 2 * pad, height - Inches(1.3), [label], size=label_size, tint=sub)

    @staticmethod
    def table(s, left, top, width, rows: list, col_widths: list, highlight: int | None = None, size: int = 14,
              row_height: float = 0.42):
        shape = s.shapes.add_table(len(rows), len(rows[0]), left, top, width, Inches(row_height * len(rows)))
        shape.name = "Таблица"
        table = shape.table
        tbl_pr = table._tbl.tblPr
        style = tbl_pr.find(qn("a:tableStyleId"))
        if style is not None:
            style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # без заливки, с сеткой
        for j, w in enumerate(col_widths):
            table.columns[j].width = Inches(w)
        for i, row in enumerate(rows):
            table.rows[i].height = Inches(row_height)
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.02)
                para = cell.text_frame.paragraphs[0]
                numeric = NUMERIC.fullmatch(str(value)) is not None
                para.alignment = PP_ALIGN.RIGHT if numeric and j > 0 else PP_ALIGN.LEFT
                run = para.add_run()
                run.text = str(value)
                run.font.size = Pt(size)
                run.font.bold = i == 0 or i == highlight
                run.font.color.rgb = color("white" if i == 0 else "ink")
                cell.fill.solid()
                cell.fill.fore_color.rgb = color("pine" if i == 0 else "mint" if i == highlight else "white")
        return shape


def style_chart(chart, title: str | None = None, legend: bool = False, number_format: str = "0") -> None:
    chart.font.size = Pt(13)
    chart.font.name = BODY_FONT
    chart.font.color.rgb = color("muted")
    chart.has_title = bool(title)
    if title:
        chart.chart_title.text_frame.text = title
        run = chart.chart_title.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = color("ink")
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(13)
    value_axis = chart.value_axis
    value_axis.has_major_gridlines = True
    value_axis.major_gridlines.format.line.color.rgb = color("line")
    value_axis.format.line.fill.background()
    value_axis.tick_labels.number_format = number_format
    value_axis.tick_labels.number_format_is_linked = False
    chart.category_axis.format.line.color.rgb = color("line")
    chart.category_axis.has_major_gridlines = False


def color_series(series, tint: str, line: bool = False) -> None:
    if line:
        series.format.line.color.rgb = color(tint)
        series.format.line.width = Pt(2.5)
        series.smooth = False
        series.marker.style = XL_MARKER_STYLE.CIRCLE
        series.marker.size = 8
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = color(tint)
        series.marker.format.line.color.rgb = color(tint)
    else:
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = color(tint)


def add_bar(s, left, top, width, height, categories, values, tint="green", title=None, number_format="0.0",
            name="Диаграмма"):
    data = CategoryChartData()
    data.categories = categories
    data.add_series(title or "Значение", values)
    frame = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, left, top, width, height, data)
    frame.name = name
    chart = frame.chart
    style_chart(chart, title, number_format=number_format)
    plot = chart.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.number_format = number_format
    labels.number_format_is_linked = False
    labels.font.size = Pt(13)
    labels.font.color.rgb = color("ink")
    color_series(plot.series[0], tint)
    return chart


def add_lines(s, left, top, width, height, categories, series: dict, title=None, number_format="0", name="Диаграмма"):
    data = CategoryChartData()
    data.categories = categories
    for label, (values, _) in series.items():
        data.add_series(label, values)
    frame = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, left, top, width, height, data)
    frame.name = name
    chart = frame.chart
    style_chart(chart, title, legend=True, number_format=number_format)
    for plot_series, (_, tint) in zip(chart.plots[0].series, series.values(), strict=True):
        color_series(plot_series, tint, line=True)
    return chart


def s01_title(d: Deck) -> None:
    s = d.prs.slides.add_slide(d.title_layout)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = color("pine")
    s.shapes.title.text = TITLE
    s.placeholders[1].text = f"Итоговый проект: {COURSE}\n{AUTHOR}, {GROUP}. Проект индивидуальный"
    s.notes_slide.notes_text_frame.text = (
        "Здравствуйте. Моя тема №33: прогноз дефолта по данным LendingClub. Задача двойная: предсказать вероятность "
        "дефолта по данным до выдачи займа и проверить, насколько грейды LendingClub отражают реальный риск.")


def s02_summary(d: Deck) -> None:
    r = d.r
    s = d.slide("Модель полезна, но главное открытие в проверке метода",
                "Сразу итог. Выбранная логистическая регрессия даёт AUC 0,689 на займах, которые не участвовали в "
                "обучении. Первоначальная версия показывала 0,736, но эта цифра была завышена: об этом отдельный слайд. "
                "И неожиданный результат: один грейд сильнее всех данных заёмщика вместе взятых.")
    w, h, top = Inches(3.85), Inches(3.3), Inches(1.7)
    gap = (SLIDE_W - 2 * MARGIN - 3 * w) // 2
    d.stat(s, MARGIN, top, w, h, ru(r.m(LR, "roc_auc")),
           f"ROC-AUC выбранной логистической регрессии на {thousands(r.metadata['test_rows'])} займах 2015 г.", "pine",
           "white", "mint", label_size=18)
    d.stat(s, MARGIN + w + gap, top, w, h,
           f"−{ru(r.censored('все', GBM, 'test_auc') - r.censored('созревшие', GBM, 'test_auc'), 2)}",
           "столько AUC оказалось завышением из-за незавершённых 60-месячных займов", label_size=18)
    d.stat(s, MARGIN + 2 * (w + gap), top, w, h, f"{ru(r.m(GRADE, 'roc_auc'))} > {ru(r.m(GBM_B, 'roc_auc'))}",
           "один подгрейд LendingClub сильнее модели на всех полях заёмщика", value_size=34, label_size=18)
    d.text(s, MARGIN, Inches(5.45), SLIDE_W - 2 * MARGIN, Inches(1.2),
           [f"При одобрении 80% заявок модель снижает дефолтность с "
            f"{ru(r.approve(LR, 1.0, 'default_rate_pct'), 1)}% до {ru(r.approve(LR, 0.8, 'default_rate_pct'), 1)}%."],
           size=20, tint="ink")


def s03_data(d: Deck) -> None:
    r = d.r
    raw = r.audit["raw_columns"]
    s = d.slide("Данные: 2,26 млн займов и всего 33 полезные колонки",
                "Данные с Kaggle. Из 151 колонки 57 пропущены более чем наполовину, ещё 18 известны только после выдачи: "
                "использовать их значит подсмотреть ответ. Аудит нашёл 33 служебные строки, некорректные доходы и DTI, "
                "дубликатов нет. Значения заменяются пропуском, строки не удаляются.")
    w = Inches(2.85)
    gap = (SLIDE_W - 2 * MARGIN - 4 * w) // 3
    for i, (value, label) in enumerate([
        (f"{ru(r.ingest['rows'] / 1e6, 2)} млн", "займов, 2007–2018"),
        (str(raw["n_columns"]), "колонка в файле"),
        (str(raw["columns_over_50pct_missing"]), "колонок пропущены более чем на 50%"),
        (str(len(raw["known_leakage_columns_excluded"])), "колонок известны только после выдачи"),
    ]):
        d.stat(s, MARGIN + i * (w + gap), Inches(1.6), w, Inches(2.4), value, label)
    invalid = r.audit["invalid_values"]
    keys = list(invalid)
    d.table(s, MARGIN, Inches(4.35), Inches(12.1), [
        ["Аудит качества", "Строк", "Решение"],
        ["Служебные строки-итоги", r.audit["summary_lines_removed"], "удалены"],
        ["Дубликаты id", r.audit["duplicate_ids"], "нет"],
        [f"{keys[0]};  {keys[1]};  {keys[2]}", f"{thousands(invalid[keys[0]] + invalid[keys[1]] + invalid[keys[2]])}",
         "заменены на пропуск"],
        ["Пропуски: emp_length, mort_acc", f"{ru(r.audit['missing_pct']['emp_length'], 1)}%, "
         f"{ru(r.audit['missing_pct']['mort_acc'], 1)}%", "медиана или мода по train"],
    ], [6.6, 2.0, 3.5], row_height=0.44)


def s04_volume(d: Deck) -> None:
    r = d.r
    b = r.benchmarks
    mem = b["memory"]
    est = mem["full_dataset_estimate_mb"]
    csv, pq = b["csv_vs_parquet"]["csv_gz"], b["csv_vs_parquet"]["parquet"]
    pd_run, sql = b["pandas_vs_sql"]["pandas_parquet"], b["pandas_vs_sql"]["sqlite"]
    s = d.slide("Отбор колонок и типов сократил память в 12 раз",
                "Чтение чанками, запись в Parquet, типы category и float32. Первая цифра оценена масштабированием с 200 "
                "тысяч строк, последняя измерена. Сравнение Pandas и SQLite: по времени выигрывает Pandas, по памяти "
                "SQLite. В ранней версии я считал SQLite быстрее, но это был артефакт tracemalloc. Каждый замер теперь "
                "идёт в отдельном процессе, берётся медиана трёх.")
    add_bar(s, MARGIN, Inches(1.5), Inches(5.6), Inches(5.3),
            ["Все 151 колонка", "33 колонки", "33 колонки + типы"],
            [est["all_cols"], est["selected_cols"], mem["full_parquet_loaded_optimized_mb"]],
            tint="green", title="Память для всего набора, МБ", number_format="#,##0", name="Память по вариантам")
    d.table(s, Inches(6.6), Inches(1.7), Inches(6.1), [
        ["Задача", "Время, с", "Память, МБ"],
        ["CSV.gz: чтение 6 колонок", ru(csv["seconds"], 1), ru(csv["peak_mb"], 0)],
        ["Parquet: чтение 6 колонок", ru(pq["seconds"], 1), ru(pq["peak_mb"], 0)],
        ["Pandas: агрегация", ru(pd_run["seconds"], 1), ru(pd_run["peak_mb"], 0)],
        ["SQLite: агрегация", ru(sql["seconds"], 1), ru(sql["peak_mb"], 0)],
    ], [3.5, 1.2, 1.4], row_height=0.5)
    d.text(s, Inches(6.6), Inches(4.5), Inches(6.1), Inches(2.1),
           [[("Выбор: ", True), ("Parquet как рабочий формат, Pandas для аналитики и обучения, "
                                 "SQLite для нормализованного хранения и запросов.", False)],
            f"Размер файла: {ru(csv['file_mb'], 0)} МБ в CSV.gz против {ru(pq['file_mb'], 0)} МБ в Parquet."],
           size=16)


def s05_sql(d: Deck) -> None:
    s = d.slide("Нормализованная база и 10 запросов SQL",
                "Семь таблиц: четыре справочника, условия займа, профиль заёмщика и исход. Поля исхода вынесены отдельно, "
                "и это ещё одна защита от утечки. Запросов десять, среди них JOIN, HAVING, CTE, RANK, LAG, NTILE и "
                "скользящее среднее. Результаты загружаются в Pandas и идут в графики.")
    image = s.shapes.add_picture(str(ROOT / "docs" / "er_diagram.png"), MARGIN, Inches(1.35), height=Inches(5.75))
    image.name = "ER-диаграмма базы данных"
    left = image.left + image.width + Inches(0.5)
    width = SLIDE_W - left - MARGIN
    d.table(s, left, Inches(1.6), width, [
        ["Приём", "Запрос"],
        ["JOIN, GROUP BY", "дефолт по грейдам"],
        ["HAVING", "цели от 5000 займов"],
        ["CTE, RANK()", "рейтинг штатов"],
        ["AVG() OVER", "скользящее среднее"],
        ["LAG()", "прирост год к году"],
        ["NTILE()", "децили DTI"],
        ["Подзапрос", "доход и подгрейд"],
        ["CASE", "FICO и тип жилья"],
    ], [Emu(int(width * 0.42)) / 914400, Emu(int(width * 0.58)) / 914400], row_height=0.46, size=15)


def s06_eda(d: Deck) -> None:
    r = d.r
    g = r.default_by_grade
    dti = r.dti.set_index("dti_decile")
    income = r.income.set_index(["grade", "income_group"])
    s = d.slide("Грейд хорошо упорядочивает риск, но не исчерпывает его",
                "Дефолтность монотонно растёт от грейда A к G. Но внутри грейда доход всё ещё различает заёмщиков: в грейде "
                "C разница около 4 процентных пунктов. Значит, в данных есть сигнал, который грейд не учёл, поэтому "
                "модель имеет смысл.")
    add_bar(s, MARGIN, Inches(1.5), Inches(7.6), Inches(5.3), list(g["grade"]), list(g["default_rate_pct"]),
            title="Дефолтность по грейду, % (созревшие займы)", name="Дефолтность по грейдам")
    x, w = Inches(8.6), Inches(4.1)
    d.stat(s, x, Inches(1.6), w, Inches(2.45),
           f"{ru(income.loc[('C', 'above_avg'), 'default_rate_pct'], 1)} и {ru(income.loc[('C', 'below_avg'), 'default_rate_pct'], 1)}%",
           "дефолт в грейде C при доходе выше и ниже среднего по подгрейду")
    d.stat(s, x, Inches(4.25), w, Inches(2.45),
           f"{ru(dti.loc[1, 'default_rate_pct'], 1)} и {ru(dti.loc[10, 'default_rate_pct'], 1)}%",
           "дефолт в нижнем и верхнем дециле долговой нагрузки DTI")


def s07_censoring(d: Deck) -> None:
    r = d.r
    v = r.vintage[r.vintage["issue_year"].between(2011, 2017)]
    t36, t60 = v[v["term_months"] == 36], v[v["term_months"] == 60]
    s = d.slide("Ловушка: незавершённые займы завышали результат",
                "Главная находка. Займ на 60 месяцев, выданный в 2015 году, к концу данных ещё не истёк. Завершены только "
                "досрочно погашенные и уже списанные, поэтому дефолтность среди них 36%, а у полностью созревшей когорты 2013 "
                "года 25%. Истинное значение для 2015 года неизвестно. Модель "
                "выучила срок как искусственно сильный признак. Исправление: берём только займы, срок которых истёк к "
                "декабрю 2018. AUC падает с 0,736 до 0,697, зато эта цифра честная. Цена: тест содержит только 36 месяцев.")
    add_lines(s, MARGIN, Inches(1.5), Inches(7.6), Inches(5.3), [str(y) for y in t36["issue_year"]],
              {"36 месяцев": (list(t36["default_rate_pct"]), "green"), "60 месяцев": (list(t60["default_rate_pct"]), "rust")},
              title="Дефолтность среди завершённых займов, %", name="Смещение дефолтности по годам")
    x, w = Inches(8.6), Inches(4.1)
    d.table(s, x, Inches(1.65), w, [
        ["Когорта LightGBM", "AUC"],
        ["Все завершённые", ru(r.censored("все", GBM, "test_auc"))],
        ["Они же, только 36 мес.", ru(r.censored("все", GBM, "auc_36m"))],
        ["Созревшие (итог)", ru(r.censored("созревшие", GBM, "test_auc"))],
    ], [3.1, 1.0], highlight=3, row_height=0.5, size=15)
    d.text(s, x, Inches(4.0), w, Inches(2.8),
           [[("Правило: ", True), ("срок займа истёк к декабрю 2018 г.", False)],
            [("Цена: ", True), ("тест содержит только займы на 36 месяцев.", False)]], size=18)


def s08_method(d: Deck) -> None:
    r = d.r
    s = d.slide("Методика: разбиение по времени и без утечки",
                "Разбиение только по времени. Порог и калибровка подбираются на 2014 годе на модели, обученной до 2013. "
                "Правило выбора задано заранее и применяется к валидации. Тест используется один раз. Поля, известные "
                "только после выдачи, исключены, и это проверяет автоматический тест. Поля, назначенные LendingClub, "
                "вынесены в отдельный вариант модели, потому что тема требует полей на момент заявки.")
    windows = [("fit", "2007–2013", "обучение для подбора", "mint", "ink"),
               ("valid", "2014, 36 мес.", "порог, калибровка, выбор", "green", "white"),
               ("test", "2015, 36 мес.", f"{thousands(r.metadata['test_rows'])} займов, только оценка", "pine", "white")]
    w = Inches(3.85)
    gap = (SLIDE_W - 2 * MARGIN - 3 * w) // 2
    for i, (name, years, role, fill, ink) in enumerate(windows):
        left = MARGIN + i * (w + gap)
        d.card(s, left, Inches(1.55), w, Inches(1.55), fill, name=f"Окно {name}")
        d.text(s, left + Inches(0.25), Inches(1.68), w - Inches(0.5), Inches(0.55), [f"{name}: {years}"], size=20, tint=ink,
               bold=True)
        d.text(s, left + Inches(0.25), Inches(2.3), w - Inches(0.5), Inches(0.7), [role], size=16, tint=ink)
    d.text(s, MARGIN, Inches(3.2), SLIDE_W - 2 * MARGIN, Inches(0.4),
           [f"Финальная модель обучена на 2007–2014: {thousands(r.metadata['train_rows'])} займов. "
            "Подбор гиперпараметров: TimeSeriesSplit внутри этого окна."], size=16, tint="muted")
    half = (SLIDE_W - 2 * MARGIN - Inches(0.4)) // 2
    for i, (head, items) in enumerate([
        ("Без утечки", ["только поля, известные до выдачи", "total_pymnt только для оценки портфеля",
                        "предобработка обучается внутри Pipeline"]),
        ("Два набора признаков", ["все поля, 28 признаков", "только поля заёмщика: без grade, sub_grade, int_rate, "
                                  "installment, verification_status и др."]),
    ]):
        left = MARGIN + i * (half + Inches(0.4))
        d.card(s, left, Inches(3.85), half, Inches(2.95), name=head)
        d.text(s, left + Inches(0.3), Inches(4.05), half - Inches(0.6), Inches(0.5), [head], size=20, tint="pine", bold=True)
        d.text(s, left + Inches(0.3), Inches(4.65), half - Inches(0.6), Inches(2.0), items, size=18)


def s09_models(d: Deck) -> None:
    r = d.r
    labels = {"Dummy (prior)": "Dummy", GRADE: "Только подгрейд", LR: "Logistic Regression, выбрана", GBM: "LightGBM",
              "Logistic Regression (borrower only)": "LogReg, поля заёмщика", GBM_B: "LightGBM, поля заёмщика"}
    order = list(labels)
    rows = [["Модель", "Valid AUC", "Test AUC", "Gini", "KS", "Brier", "ECE", "Обучение, с", "Размер, МБ"]]
    for m in order:
        rows.append([labels[m], ru(r.m(m, "valid_auc")), ru(r.m(m, "roc_auc")), ru(r.m(m, "gini")), ru(r.m(m, "ks")),
                     ru(r.m(m, "brier"), 4), ru(r.m(m, "ece"), 4), ru(r.m(m, "fit_seconds"), 1), ru(r.m(m, "model_mb"), 2)])
    gap = r.m(GBM, "valid_auc") - r.m(LR, "valid_auc")
    s = d.slide("LightGBM лучше, но правило выбрало простую модель",
                "Сравнение с двумя базовыми вариантами на одном тесте. Правило задано заранее: лучшая AUC на валидации, но "
                f"логистическая регрессия предпочитается при отставании менее 0,005. Отставание {ru(gap)}, поэтому она и "
                f"выбрана. На тесте LightGBM лучше на {ru(r.m(GBM, 'roc_auc') - r.m(LR, 'roc_auc'))}, но менять выбор по тесту "
                "нельзя: это подгонка. Если важнее качество, чем интерпретируемость, LightGBM является обоснованной "
                "альтернативой. Калибровка снижает ECE почти вдвое и не меняет порядок заёмщиков.")
    d.table(s, MARGIN, Inches(1.55), Inches(12.1), rows, [3.9, 1.0, 1.0, 0.9, 0.9, 1.05, 1.05, 1.2, 1.1], highlight=3,
            size=14, row_height=0.5)
    d.text(s, MARGIN, Inches(5.3), Inches(12.1), Inches(1.5),
           [[("Правило, заданное заранее: ", True),
             (f"лучшая AUC на валидации, но логистическая регрессия при отставании менее 0,005. Отставание {ru(gap)}.", False)],
            [("Калибровка Платта на 2014 годе: ", True),
             (f"ECE логистической регрессии {ru(r.m(LR, 'ece_raw'), 4)} → {ru(r.m(LR, 'ece'), 4)}, AUC не меняется.", False)]],
           size=17)


def s10_grade(d: Deck) -> None:
    r = d.r
    rec = r.recovery
    s = d.slide("Грейд сильнее всех опубликованных данных заёмщика",
                "Ответ на вопрос темы об оценке риска по грейду. Один подгрейд даёт больше, чем LightGBM на всех полях "
                "заёмщика. Вероятно, LendingClub использует сведения кредитного бюро, которых нет в наборе; это "
                "предположение, а не проверенный факт. Модель без грейда сжимает риск: она завышает PD в грейде A и сильно "
                "занижает в F и G. Но и грейд не всё: данные заёмщика добавляют к нему 0,018 AUC.")
    data = CategoryChartData()
    data.categories = list(rec["grade"])
    data.add_series("Факт", list(rec["actual_default_pct"]))
    data.add_series("PD, поля заёмщика", list(rec["borrower_pd_pct"]))
    data.add_series("PD, все поля", list(rec["full_pd_pct"]))
    frame = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN, Inches(1.5), Inches(8.0), Inches(5.3), data)
    frame.name = "Фактический дефолт и прогноз по грейдам"
    chart = frame.chart
    style_chart(chart, "Дефолтность и средняя PD на тесте, %", legend=True)
    chart.plots[0].gap_width = 60
    chart.plots[0].overlap = -5
    for series, tint in zip(chart.plots[0].series, ["pine", "amber", "green"], strict=True):
        color_series(series, tint)
    x, w = Inches(9.0), Inches(3.75)
    d.table(s, x, Inches(1.65), w, [
        ["Test AUC", ""],
        ["Только подгрейд", ru(r.m(GRADE, "roc_auc"))],
        ["LightGBM, поля заёмщика", ru(r.m(GBM_B, "roc_auc"))],
        ["LightGBM, все поля", ru(r.m(GBM, "roc_auc"))],
    ], [2.85, 0.9], row_height=0.5, size=15)
    d.text(s, x, Inches(4.0), w, Inches(2.8),
           [f"Поля LendingClub добавляют {ru(r.m(GBM, 'roc_auc') - r.m(GBM_B, 'roc_auc'))} AUC.",
            f"Данные заёмщика добавляют к подгрейду {ru(r.m(GBM, 'roc_auc') - r.m(GRADE, 'roc_auc'))} AUC.",
            f"Главные признаки без грейда: {', '.join(r.shap_borrower['feature'].head(3))}."], size=17)


def s11_policy(d: Deck) -> None:
    r = d.r
    rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    errors = r.errors.set_index("grade")
    s = d.slide("Для решений нужна таблица одобрения",
                f"Единый порог F1 работает как граница по грейду: в грейде A пропускается "
                f"{ru(errors.loc['A', 'missed_defaults_pct'], 0)}% дефолтов, в F ложно отклоняется "
                f"{ru(errors.loc['F', 'false_alarms_pct'], 0)}% хороших заёмщиков. Поэтому показываю другое: если одобрить "
                "заданную долю заявок с наименьшей PD, какая будет дефолтность. Доходность здесь номинальная, за весь срок, "
                "без стоимости фондирования, так что это сравнение моделей, а не прогноз прибыли.")
    chart = add_lines(s, MARGIN, Inches(1.5), Inches(7.6), Inches(5.3), [f"{int(x * 100)}%" for x in rates],
                      {"Только подгрейд": ([r.approve(GRADE, x, "default_rate_pct") for x in rates], "amber"),
                       "Logistic Regression": ([r.approve(LR, x, "default_rate_pct") for x in rates], "pine"),
                       "LightGBM": ([r.approve(GBM, x, "default_rate_pct") for x in rates], "rust")},
                      title="Дефолтность одобренного портфеля, %, по доле одобрения",
                      name="Дефолтность по доле одобрения")
    chart.value_axis.minimum_scale = 6
    chart.value_axis.maximum_scale = 16
    x, w = Inches(8.6), Inches(4.1)
    d.table(s, x, Inches(1.65), w, [
        ["Одобрено 80%", "Дефолт", "Возврат"],
        ["Только подгрейд", f"{ru(r.approve(GRADE, 0.8, 'default_rate_pct'), 1)}%", f"{ru(r.approve(GRADE, 0.8, 'net_return_pct'), 1)}%"],
        ["Logistic Regression", f"{ru(r.approve(LR, 0.8, 'default_rate_pct'), 1)}%", f"{ru(r.approve(LR, 0.8, 'net_return_pct'), 1)}%"],
        ["LightGBM", f"{ru(r.approve(GBM, 0.8, 'default_rate_pct'), 1)}%", f"{ru(r.approve(GBM, 0.8, 'net_return_pct'), 1)}%"],
        ["Одобрить всех", f"{ru(r.approve(LR, 1.0, 'default_rate_pct'), 1)}%", f"{ru(r.approve(LR, 1.0, 'net_return_pct'), 1)}%"],
    ], [2.2, 0.95, 0.95], highlight=2, row_height=0.5, size=14)
    d.text(s, x, Inches(4.5), w, Inches(2.3),
           ["Возврат номинальный: за весь срок, без стоимости фондирования.",
            f"Единый порог F1: в грейде A пропускает {ru(errors.loc['A', 'missed_defaults_pct'], 0)}% дефолтов."],
           size=16)


def s12_close(d: Deck) -> None:
    s = d.prs.slides.add_slide(d.content_layout)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = color("pine")
    s.shapes.title.text = "Выводы и ограничения"
    s.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = color("white")
    s.notes_slide.notes_text_frame.text = (
        "Итоги. Проверка метода дала больше, чем усложнение модели. Ограничения называю сам: тест только на 36 месяцев, "
        "обучающие метки 2013–2014 годов на начало 2015 ещё не были полностью известны, нет отклонённых заявок. Проект "
        "запускается из README, демо-выборка работает без загрузки данных, 98 тестов, покрытие 93%. Использовал ИИ-ассистента "
        "Claude Code, это указано в отчёте. Покажу дашборд и готов ответить на вопросы.")
    half = (SLIDE_W - 2 * MARGIN - Inches(0.4)) // 2
    d.text(s, MARGIN, Inches(1.5), half, Inches(0.5), ["Выводы"], size=22, tint="amber", bold=True)
    d.text(s, MARGIN, Inches(2.1), half, Inches(4.2), [
        "Грейды хорошо отражают риск, но не исчерпывают его.",
        "Проверка смещения дала больше, чем усложнение модели.",
        "Прирост LightGBM над логистической регрессией мал.",
        "Parquet и типы данных позволяют работать с полным набором на обычном компьютере.",
    ], size=18, tint="white")
    x = MARGIN + half + Inches(0.4)
    d.text(s, x, Inches(1.5), half, Inches(0.5), ["Ограничения"], size=22, tint="amber", bold=True)
    d.text(s, x, Inches(2.1), half, Inches(4.2), [
        "Тест содержит только займы на 36 месяцев.",
        "Обучающие метки 2013–2014 гг. на начало 2015 г. ещё не были полностью известны.",
        "Нет отклонённых заявок, reject inference не выполнялся.",
        "Риск недооценивается и после калибровки: нужен регулярный пересчёт.",
    ], size=18, tint="white")
    d.text(s, MARGIN, Inches(6.45), Inches(11.2), Inches(0.5),
           ["98 тестов, покрытие 93%. Запуск без загрузки данных описан в README. При подготовке использован ИИ-ассистент "
            "Claude Code, см. отчёт."], size=13, tint="mint")


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    apply_theme(prs)
    layouts = set_layout_placeholders(prs)
    prs.core_properties.title = TITLE
    prs.core_properties.author = AUTHOR
    deck = Deck(prs, layouts, load())
    for build_slide in (s01_title, s02_summary, s03_data, s04_volume, s05_sql, s06_eda, s07_censoring, s08_method,
                        s09_models, s10_grade, s11_policy, s12_close):
        build_slide(deck)
    prs.save(OUTPUT)
    print(f"Готово: {OUTPUT}")


if __name__ == "__main__":
    build()

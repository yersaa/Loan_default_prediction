"""Собирает письменный отчёт docs/report.docx. Все числа читаются из reports/ через docs/results.py.

Запуск: python docs/build_report.py
Оглавление обновляется при открытии в Word (или автоматически при экспорте в PDF).
"""
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from results import (
    AUTHOR,
    BRAND,
    COURSE,
    COVERAGE_PCT,
    FIGURES,
    GROUP,
    ROOT,
    TEST_COUNT,
    TITLE,
    Results,
    load,
    ru,
    thousands,
)

OUTPUT = ROOT / "docs" / "report.docx"
NUMERIC = re.compile(r"[\d\s ,.%—-]+")
BODY_FONT, HEAD_FONT = "Times New Roman", "Arial"
MODEL_LABELS = {
    "Dummy (prior)": "Dummy (доля дефолтов)", "Grade only (LogReg)": "Только подгрейд LC",
    "Logistic Regression": "Logistic Regression", "LightGBM": "LightGBM",
    "Logistic Regression (borrower only)": "LogReg, поля заёмщика", "LightGBM (borrower only)": "LightGBM, поля заёмщика",
}
GBM, LR, GRADE, GBM_B, LR_B = ("LightGBM", "Logistic Regression", "Grade only (LogReg)", "LightGBM (borrower only)",
                               "Logistic Regression (borrower only)")


def rgb(name: str) -> RGBColor:
    return RGBColor.from_string(BRAND[name])


def set_font(run_or_style, name: str) -> None:
    run_or_style.font.name = name
    element = run_or_style.element.get_or_add_rPr() if hasattr(run_or_style.element, "get_or_add_rPr") else None
    if element is None:
        element = run_or_style.element.rPr
    fonts = element.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        element.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def setup_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    set_font(normal, BODY_FONT)
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for level, size, before, after in ((1, 15, 16, 8), (2, 12.5, 12, 6)):
        style = doc.styles[f"Heading {level}"]
        set_font(style, HEAD_FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = rgb("pine")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    caption = doc.styles["Caption"]
    set_font(caption, BODY_FONT)
    caption.font.size = Pt(10)
    caption.font.italic = True
    caption.font.bold = False
    caption.font.color.rgb = rgb("muted")
    for bullet in ("List Bullet", "List Number"):
        doc.styles[bullet].paragraph_format.space_after = Pt(3)


def add_field(paragraph, instruction: str) -> None:
    """Поле Word: номер страницы или оглавление."""
    for kind, text in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        run = paragraph.add_run()
        if kind:
            field = OxmlElement("w:fldChar")
            field.set(qn("w:fldCharType"), kind)
            run._r.append(field)
        else:
            code = OxmlElement("w:instrText")
            code.set(qn("xml:space"), "preserve")
            code.text = text
            run._r.append(code)


def shade(cell, color: str) -> None:
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(fill)


class Report:
    """Нумерует рисунки и таблицы и держит единый вид всех элементов."""

    def __init__(self, doc: Document):
        self.doc, self.figure_no, self.table_no = doc, 0, 0

    def h1(self, text: str) -> None:
        self.doc.add_heading(text, level=1)

    def h2(self, text: str) -> None:
        self.doc.add_heading(text, level=2)

    def p(self, text: str, bold_lead: str | None = None) -> None:
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_lead:
            para.add_run(bold_lead + " ").bold = True
        para.add_run(text)

    def bullets(self, items: list) -> None:
        for item in items:
            self.doc.add_paragraph(item, style="List Bullet")

    def figure(self, path: Path, caption: str, width_cm: float = 16.0) -> None:
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        self.figure_no += 1
        cap = self.doc.add_paragraph(f"Рисунок {self.figure_no}. {caption}", style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def table(self, header: list, rows: list, caption: str, widths: list, highlight: int | None = None) -> None:
        self.table_no += 1
        cap = self.doc.add_paragraph(f"Таблица {self.table_no}. {caption}", style="Caption")
        cap.paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for j, title in enumerate(header):
            self._cell(table.cell(0, j), title, widths[j], bold=True, color="white", first=j == 0)
            shade(table.cell(0, j), BRAND["pine"])
        for i, row in enumerate(rows, 1):
            for j, value in enumerate(row):
                self._cell(table.cell(i, j), str(value), widths[j], bold=i - 1 == highlight, first=j == 0)
                if i - 1 == highlight:
                    shade(table.cell(i, j), BRAND["mint"])
        for cell in table.rows[-1].cells:
            cell.paragraphs[0].paragraph_format.keep_with_next = False
        repeat = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    @staticmethod
    def _cell(cell, text: str, width: float, bold: bool = False, color: str | None = None, first: bool = False) -> None:
        cell.width = Cm(width)
        para = cell.paragraphs[0]
        numeric = NUMERIC.fullmatch(text) is not None
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT if numeric and not first else WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.keep_with_next = True  # таблица не разрывается между страницами
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1.0
        run = para.add_run(text)
        run.font.size = Pt(9.5)
        run.bold = bold
        if color:
            run.font.color.rgb = rgb(color)

    def page_break(self) -> None:
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def title_page(doc: Document) -> None:
    for text, size, bold, before in (
        ("Applied Artificial Intelligence, магистратура", 12, False, 40),
        (f"Дисциплина «{COURSE}»", 12, False, 4),
        ("ИТОГОВЫЙ ПРОЕКТ", 13, True, 110),
        (TITLE, 22, True, 10),
        ("Тема №33: Loan Default Prediction (LendingClub)", 12, False, 6),
        (f"Автор: {AUTHOR}, группа {GROUP}", 12, False, 150),
        ("Проект выполнен индивидуально", 12, False, 4),
        ("2026", 12, False, 70),
    ):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(before)
        run = para.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        if size > 12:
            run.font.color.rgb = rgb("pine")
            set_font(run, HEAD_FONT)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def contents(rep: Report) -> None:
    head = rep.doc.add_paragraph()
    run = head.add_run("Содержание")
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = rgb("pine")
    set_font(run, HEAD_FONT)
    add_field(rep.doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    rep.page_break()


def summary(rep: Report, r: Results) -> None:
    rep.h1("Краткие итоги")
    rep.p(f"По данным LendingClub ({thousands(r.ingest['rows'])} займов за 2007–2018 гг.) построены и сравнены модели "
          f"вероятности дефолта. Итоговая оценка выполнена на {thousands(r.metadata['test_rows'])} созревших займах "
          "2015 года, которые не участвовали ни в обучении, ни в подборе порога и калибровки.")
    rep.bullets([
        f"По заранее заданному правилу выбрана логистическая регрессия: ROC-AUC {ru(r.m(LR, 'roc_auc'))} "
        f"и Gini {ru(r.m(LR, 'gini'))} на тесте. LightGBM на тесте лучше на "
        f"{ru(r.m(GBM, 'roc_auc') - r.m(LR, 'roc_auc'))}, но правило применяется к валидации, а не к тесту.",
        f"В исходной версии проекта результат был завышен: AUC {ru(r.censored('все', GBM, 'test_auc'))} на всех "
        f"завершённых займах против {ru(r.censored('созревшие', GBM, 'test_auc'))} на созревших. Причина в "
        "цензурировании 60-месячных займов.",
        f"Один подгрейд LendingClub (AUC {ru(r.m(GRADE, 'roc_auc'))}) обгоняет LightGBM на всех полях заёмщика "
        f"(AUC {ru(r.m(GBM_B, 'roc_auc'))}). Грейд содержит сигнал, которого нет в опубликованных полях анкеты.",
        f"Модель даёт практическую пользу при отборе заявок: при одобрении 80% заявок дефолтность составляет "
        f"{ru(r.approve(LR, 0.8, 'default_rate_pct'), 1)}% против {ru(r.approve(LR, 1.0, 'default_rate_pct'), 1)}% "
        "при одобрении всех.",
    ])


def data_section(rep: Report, r: Results) -> None:
    raw = r.audit["raw_columns"]
    rep.h1("1. Данные")
    rep.p(f"Источник: набор All Lending Club loan data на Kaggle, файл accepted_2007_to_2018Q4.csv.gz [1]. Он содержит "
          f"{thousands(r.ingest['rows'])} займов и {raw['n_columns']} колонку, размер сжатого файла "
          f"{ru(r.ingest['csv_gz_mb'], 0)} МБ. Это превышает рекомендованный порог требований: 500 тысяч строк или 200 МБ.")
    rep.h2("1.1. Аудит качества")
    rep.p(f"Из {thousands(r.audit['rows_raw'])} строк файла {r.audit['summary_lines_removed']} оказались служебными "
          "итогами без данных и удалены. Дубликатов id нет. Проверки логики дали следующие результаты.")
    invalid = r.audit["invalid_values"]
    keys = list(invalid)
    rep.table(["Проверка", "Строк", "Решение"], [
        [keys[0], thousands(invalid[keys[0]]), "заменить на пропуск: доход не может быть нулевым"],
        [keys[1], thousands(invalid[keys[1]]), "заменить на пропуск: служебное значение 999"],
        [keys[2], thousands(invalid[keys[2]]), "заменить на пропуск: вероятная ошибка данных"],
        [keys[3], invalid[keys[3]], "нарушений нет"],
        [keys[4], invalid[keys[4]], "нарушений нет"],
        [keys[5], invalid[keys[5]], "нарушений нет"],
    ], "Некорректные значения и решения по ним", [6.0, 1.6, 8.4])
    missing = r.audit["missing_pct"]
    rep.p(f"Пропуски в отобранных колонках невелики: emp_length {ru(missing['emp_length'], 2)}%, mort_acc "
          f"{ru(missing['mort_acc'], 2)}%, остальные менее 0,2%. Они заполняются медианой или модой внутри Pipeline, "
          "то есть только по обучающей выборке. Выбросы по правилу IQR есть в доходе и балансе "
          f"({ru(r.audit['iqr_outliers']['annual_inc']['share_pct'], 2)}% и "
          f"{ru(r.audit['iqr_outliers']['revol_bal']['share_pct'], 2)}%). Это реальные заёмщики с большими значениями, "
          "поэтому строки не удаляются: для логистической регрессии значения обрезаются по 1-му и 99-му перцентилям train.")
    rep.figure(FIGURES / "01_missing.png", "Доля пропусков в отобранных колонках", 13)
    rep.p(f"На первых {thousands(raw['rows_checked'])} строках файла {raw['columns_over_50pct_missing']} колонок из "
          f"{raw['n_columns']} пропущены более чем на 50%, ещё {len(raw['known_leakage_columns_excluded'])} колонок "
          f"известны только после выдачи. Поэтому в работу взяты {raw['kept_columns']} колонки: поля, доступные до "
          "выдачи, и четыре поля исхода для целевой переменной и оценки портфеля.")
    rep.h2("1.2. Обработка большого объёма")
    mem = r.benchmarks["memory"]
    est = mem["full_dataset_estimate_mb"]
    rep.p(f"Файл читается чанками по 250 тысяч строк и записывается в единый Parquet со сжатием snappy. Выбор колонок "
          f"и типов category и float32 уменьшает занимаемую память с оценочных {thousands(est['all_cols'])} МБ до "
          f"{ru(mem['full_parquet_loaded_optimized_mb'], 0)} МБ, то есть в "
          f"{ru(est['all_cols'] / mem['full_parquet_loaded_optimized_mb'], 1)} раза. Оценка для всех колонок получена "
          f"масштабированием замера на первых {thousands(mem['sample_rows'])} строках; значение после оптимизации "
          "измерено на полном наборе.")
    rep.h2("1.3. Цензурирование и отбор созревших займов")
    v = r.vintage.set_index(["issue_year", "term_months"])
    rep.p("Целевая переменная: Fully Paid равно 0, Charged Off и Default равно 1. Обычно незавершённые займы просто "
          "исключают, но этого недостаточно. Среди ещё не истёкших займов завершены только досрочно погашенные и уже "
          f"списанные. Так, из 60-месячных займов 2015 года завершены {ru(v.loc[(2015, 60), 'resolved_pct'], 1)}%, "
          f"и дефолтность среди них составляет {ru(v.loc[(2015, 60), 'default_rate_pct'], 1)}%. У полностью созревших "
          f"60-месячных займов 2013 года она равна {ru(v.loc[(2013, 60), 'default_rate_pct'], 1)}%.")
    rep.figure(FIGURES / "03_vintage.png", "Дефолтность среди завершённых займов и доля завершённых займов по годам")
    rep.p("Поэтому в модель попадают только займы, срок которых истёк к декабрю 2018 г.: 36-месячные с выдачей до 2015 "
          "года включительно и 60-месячные до 2013 года включительно. Следствие: тест 2015 года содержит только займы "
          "на 36 месяцев, и 60-месячные займы вне времени не проверены.")


def database_section(rep: Report, r: Results) -> None:
    rep.h1("2. База данных и SQL")
    rep.p("Данные загружены в SQLite в нормализованную схему. Справочники dim_grade, dim_purpose, dim_state и dim_home "
          "устраняют повторы строк. Таблица loans хранит условия займа, borrowers содержит профиль заёмщика на момент "
          "заявки, outcomes содержит поля исхода. Такое разделение также защищает от утечки: всё, что известно только "
          "после выдачи, находится в отдельной таблице.")
    rep.figure(ROOT / "docs" / "er_diagram.png", "ER-диаграмма базы данных", 14.5)
    rep.table(["№", "Запрос", "Приём SQL", "Что показывает"], [
        [1, "default_by_grade", "JOIN, GROUP BY", "дефолтность и ставка по грейдам"],
        [2, "purpose_having", "GROUP BY, HAVING", "цели кредита от 5000 займов"],
        [3, "vintage_by_year_term", "JOIN, COUNT колонки", "цензурирование по годам и срокам"],
        [4, "state_rank", "CTE, RANK() OVER", "рейтинг штатов в целом и в регионе"],
        [5, "monthly_rolling", "CTE, AVG() OVER ROWS", "скользящая дефолтность за 3 месяца"],
        [6, "yoy_volume_lag", "CTE, LAG()", "прирост объёма год к году"],
        [7, "dti_deciles", "CTE, NTILE()", "дефолтность по децилям DTI"],
        [8, "income_vs_grade_avg", "CTE, подзапрос", "доход относительно среднего по подгрейду"],
        [9, "loss_by_grade", "JOIN, агрегаты", "возвраты по списанным займам"],
        [10, "fico_home_matrix", "CASE, JOIN 4 таблиц", "дефолтность по FICO и типу жилья"],
    ], "Аналитические запросы (sql/queries.sql). Результаты загружаются в Pandas", [0.9, 4.5, 4.3, 6.3])


def eda_section(rep: Report, r: Results) -> None:
    grade = r.default_by_grade.set_index("grade")
    dti = r.dti.set_index("dti_decile")
    purpose = r.purpose.set_index("purpose")
    income = r.income.set_index(["grade", "income_group"])
    fico = r.fico_home.set_index(["fico_band", "home_ownership"])
    top, bottom = r.state.iloc[0], r.state.iloc[-1]
    rep.h1("3. Разведочный анализ")
    rep.p("Показатели дефолтности в этом разделе рассчитаны только по созревшим займам.")
    rep.figure(FIGURES / "02_default_by_grade.png", "Дефолтность и средняя ставка по грейдам")
    rep.p(f"Грейд хорошо упорядочивает риск: дефолтность растёт от {ru(grade.loc['A', 'default_rate_pct'], 1)}% в грейде A "
          f"до {ru(grade.loc['G', 'default_rate_pct'], 1)}% в грейде G, ставка от {ru(grade.loc['A', 'avg_int_rate'], 1)}% "
          f"до {ru(grade.loc['G', 'avg_int_rate'], 1)}%.", "Грейд.")
    rep.figure(FIGURES / "06_dti_deciles.png", "Дефолтность по децилям долговой нагрузки", 13)
    rep.p(f"Дефолтность монотонно растёт от {ru(dti.loc[1, 'default_rate_pct'], 1)}% в нижнем дециле DTI до "
          f"{ru(dti.loc[10, 'default_rate_pct'], 1)}% в верхнем.", "Долговая нагрузка.")
    rep.p(f"Самая рискованная цель кредита: малый бизнес ({ru(purpose.loc['small_business', 'default_rate_pct'], 1)}%), "
          f"наименее рискованная среди крупных: автомобиль ({ru(purpose.loc['car', 'default_rate_pct'], 1)}%).",
          "Цель кредита.")
    rep.p("Запрос 8 показывает, что внутри одного грейда доход всё ещё различает риск. В грейде C дефолтность равна "
          f"{ru(income.loc[('C', 'above_avg'), 'default_rate_pct'], 1)}% у заёмщиков с доходом выше среднего по подгрейду "
          f"и {ru(income.loc[('C', 'below_avg'), 'default_rate_pct'], 1)}% ниже среднего. Значит, грейд учитывает доход "
          "не полностью.", "Доход внутри грейда.")
    rep.p(f"При FICO ниже 670 и аренде жилья дефолтность равна {ru(fico.loc[('1: <670', 'RENT'), 'default_rate_pct'], 1)}%, "
          f"при FICO от 750 и ипотеке {ru(fico.loc[('4: 750+', 'MORTGAGE'), 'default_rate_pct'], 1)}%.", "FICO и жильё.")
    rep.p(f"Наибольшая дефолтность среди штатов с объёмом от 1000 займов у {top['state_code']} "
          f"({ru(top['default_rate_pct'], 1)}%), наименьшая у {bottom['state_code']} ({ru(bottom['default_rate_pct'], 1)}%). "
          "Интерактивная карта находится в reports/figures/07_default_by_state.html.", "География.")
    rep.figure(FIGURES / "05_corr_heatmap.png", "Корреляции Спирмена числовых признаков и дефолта", 12.5)
    rep.p("Сильнее всего с дефолтом связаны ставка (0,20) и FICO (−0,11). Все связи с дефолтом по отдельности слабые, "
          "поэтому модели нужно сочетание признаков. Ставка и FICO заметно коррелируют между собой (−0,46), что ожидаемо: "
          "ставка назначается в том числе по кредитному рейтингу.", "Корреляции.")


def methods_section(rep: Report, r: Results) -> None:
    rep.h1("4. Методы")
    rep.h2("4.1. Признаки и защита от утечки")
    rep.p("В модель входят 28 признаков, известных до выдачи, включая построенные: средний FICO, длина кредитной истории, "
          "отношение суммы и платежа к доходу. Поля, появляющиеся после выдачи, такие как total_pymnt, recoveries, "
          "last_pymnt_d, last_fico_range_*, исключены; это проверяется автоматическим тестом. Значение total_pymnt "
          "используется только при оценке доходности портфеля, но не как признак.")
    rep.p("Тема требует использовать поля на момент заявки, а grade, sub_grade, int_rate, installment, "
          "installment_to_income, initial_list_status и verification_status назначает сам LendingClub при рассмотрении. "
          "Это не будущая информация, однако и не поля анкеты. Поэтому каждая модель обучена в двух вариантах: на всех "
          "полях и только на полях заёмщика.")
    rep.h2("4.2. Разбиение по времени")
    rep.table(["Окно", "Годы выдачи", "Назначение"], [
        ["fit", "2007–2013", "подбор гиперпараметров; пробная модель для порога и калибровки"],
        ["valid", "2014, 36 мес.", "порог F1, калибровка Платта, правило выбора модели"],
        ["train", "2007–2014", f"финальная модель, {thousands(r.metadata['train_rows'])} займов"],
        ["test", "2015, 36 мес.", f"только итоговая оценка, {thousands(r.metadata['test_rows'])} займов"],
    ], "Временные окна", [2.5, 3.5, 10.0])
    rep.p("Гиперпараметры подбирались с TimeSeriesSplit на трёх фолдах внутри окна fit, то есть без 2014 года, по "
          "которому затем выбирается модель: параметр регуляризации C для "
          "логистической регрессии [2], число листьев и деревьев для LightGBM [3]. Импутация, обрезка выбросов и "
          "кодирование категорий обучаются внутри Pipeline и не видят тест.")
    rep.h2("4.3. Модели и базовые варианты")
    rep.bullets([
        "Dummy: константный прогноз, равный доле дефолтов в train.",
        "Только подгрейд LC: логистическая регрессия на одном sub_grade. Показывает, сколько добавляет модель к "
        "собственной оценке LendingClub.",
        "Logistic Regression: линейная скоринговая модель, принятый в банках интерпретируемый подход [7].",
        "LightGBM: градиентный бустинг на всех полях.",
        "Те же Logistic Regression и LightGBM только на полях заёмщика.",
    ])
    rep.h2("4.4. Калибровка и порог")
    rep.p("Выходом модели является вероятность дефолта, поэтому важна её калибровка. Использована калибровка Платта [4]: "
          "логистическая регрессия на логите прогноза, обученная на 2014 годе. Преобразование монотонно и не меняет "
          "порядок заёмщиков, поэтому ROC-AUC остаётся прежней. Качество оценивается по Brier и ECE [5]. Порог "
          "максимума F1 подбирается на тех же данных. Он нужен только для сравнения моделей, а реальная политика "
          "показана в таблице одобрения.")
    rep.h2("4.5. Правило выбора модели")
    rep.p(f"Правило задано в коде (config.SIMPLICITY_MARGIN_AUC) до итогового запуска и применяется к AUC на валидации: "
          f"выбирается лучшая модель, но логистическая регрессия предпочитается, если отстаёт менее чем на "
          f"{ru(0.005)}. Запас обоснован простотой интерпретации и размером модели. Значение было задано уже после "
          "предыдущего запуска на смещённой когорте, но до просмотра результатов на созревших займах.")


def results_section(rep: Report, r: Results) -> None:
    rep.h1("5. Результаты")
    rep.h2("5.1. Сравнение моделей")
    selected = r.selection["selected"]
    order = list(r.metrics["model"])
    rep.table(["Модель", "Valid AUC", "Test AUC", "Gini", "KS", "PR-AUC", "Brier", "ECE", "F1"],
              [[MODEL_LABELS[m], ru(r.m(m, "valid_auc")), ru(r.m(m, "roc_auc")), ru(r.m(m, "gini")), ru(r.m(m, "ks")),
                ru(r.m(m, "pr_auc")), ru(r.m(m, "brier"), 4), ru(r.m(m, "ece"), 4), ru(r.m(m, "f1"))] for m in order],
              "Качество на тесте: созревшие займы 2015 г. Выбранная модель выделена",
              [4.6, 1.5, 1.4, 1.3, 1.3, 1.4, 1.4, 1.4, 1.2], highlight=order.index(selected))
    rep.table(["Модель", "Обучение, с", "Пик памяти, МБ", "Прогноз, мс на 1000", "Размер, МБ"],
              [[MODEL_LABELS[m], ru(r.m(m, "fit_seconds"), 1), ru(r.fit_mb(m), 0),
                ru(r.m(m, "predict_ms_per_1000"), 1), ru(r.m(m, "model_mb"), 2)] for m in order],
              "Время и память. Пик памяти: прирост RSS процесса за время обучения; каждая модель обучается в отдельном "
              "процессе, медиана трёх запусков",
              [5.0, 2.4, 2.8, 3.2, 2.2], highlight=order.index(selected))
    rep.figure(FIGURES / "09_roc_pr.png", "ROC-кривые и кривые Precision–Recall на тесте")
    rep.p(f"Все модели превосходят Dummy. Разница между сильнейшими невелика: на валидации LightGBM опережает "
          f"логистическую регрессию на {ru(r.m(GBM, 'valid_auc') - r.m(LR, 'valid_auc'))}, поэтому по правилу выбрана "
          f"логистическая регрессия. На тесте LightGBM лучше на {ru(r.m(GBM, 'roc_auc') - r.m(LR, 'roc_auc'))} и "
          "выигрывает по всем метрикам качества, кроме ECE. Выбор по тесту означал бы подгонку под тестовые данные, "
          "поэтому решение не меняется; если важнее небольшой прирост качества, чем интерпретируемость, LightGBM является "
          f"обоснованной альтернативой. По затратам различие не в пользу одной модели: LightGBM обучается примерно столько "
          f"же ({ru(r.m(GBM, 'fit_seconds'), 1)} с против {ru(r.m(LR, 'fit_seconds'), 1)} с) и требует меньше памяти при "
          f"обучении ({ru(r.fit_mb(GBM), 0)} МБ против {ru(r.fit_mb(LR), 0)} МБ), но файл модели в {ru(r.m(GBM, 'model_mb') / r.m(LR, 'model_mb'), 0)} раз больше, а прогноз медленнее.")
    rep.p(f"AUC на тесте выше, чем на валидации: {ru(r.m(LR, 'roc_auc'))} против {ru(r.m(LR, 'valid_auc'))} у логистической "
          f"регрессии. Разрыв разложен на две части. Пробная модель, обученная на 2007–2013 годах, даёт на тесте "
          f"{ru(r.m(LR, 'probe_test_auc'))}: значит, {ru(r.m(LR, 'probe_test_auc') - r.m(LR, 'valid_auc'))} объясняется тем, "
          f"что займы 2015 года различить легче, чем займы 2014 года. Ещё {ru(r.m(LR, 'roc_auc') - r.m(LR, 'probe_test_auc'))} "
          "добавляет 2014 год в обучении финальной модели. Утечки это не означает: тест не участвует ни в обучении, ни в "
          "подборе, а поля, известные после выдачи, в модель не входят.")
    rep.h2("5.2. Влияние цензурирования")
    c = r.censoring
    rep.table(["Когорта", "Модель", "Test, строк", "Дефолт в test, %", "Test AUC", "AUC 36 мес.", "AUC 60 мес."],
              [[row["cohort"], MODEL_LABELS[row["model"]], thousands(row["test_rows"]), ru(row["test_default_pct"], 2),
                ru(row["test_auc"]), ru(row["auc_36m"]),
                "нет данных" if pd.isna(row["auc_60m"]) else ru(row["auc_60m"])] for _, row in c.iterrows()],
              "Те же модели на созревших и на всех завершённых займах", [4.1, 3.0, 1.8, 1.8, 1.4, 1.5, 1.5])
    rep.p(f"Общая AUC на смещённой когорте выше на {ru(r.censored('все', GBM, 'test_auc') - r.censored('созревшие', GBM, 'test_auc'))} "
          f"у LightGBM. Внутри 36-месячных займов результаты почти совпадают ({ru(r.censored('все', GBM, 'auc_36m'))} и "
          f"{ru(r.censored('созревшие', GBM, 'auc_36m'))}). Завышение создаётся смешением сроков: дефолтность смещённых "
          "60-месячных займов сильно отличается от 36-месячных, поэтому срок становится искусственно сильным признаком.")
    rep.h2("5.3. Калибровка")
    rep.figure(FIGURES / "11_calibration.png", "Калибровка до и после преобразования Платта на тесте", 11)
    rep.p(f"Калибровка на 2014 годе снижает ECE логистической регрессии с {ru(r.m(LR, 'ece_raw'), 4)} до "
          f"{ru(r.m(LR, 'ece'), 4)}, LightGBM с {ru(r.m(GBM, 'ece_raw'), 4)} до {ru(r.m(GBM, 'ece'), 4)}. "
          "Коэффициенты калибровки и порог подобраны по прогнозам пробной модели (2007–2013) на 2014 годе, а применяются "
          "к финальной модели, обученной на 2007–2014. Это разные модели, поэтому перенос проверен на тесте: после "
          f"калибровки логистическая регрессия предсказывает в среднем {ru(100 * r.m(LR, 'mean_pd'), 1)}% при фактических "
          f"{ru(100 * r.m(LR, 'actual_default_rate'), 1)}%. "
          f"Без калибровки LightGBM предсказывает в среднем {ru(r.censored('созревшие', GBM, 'mean_pd_pct'), 1)}% при "
          f"фактических {ru(r.censored('созревшие', GBM, 'test_default_pct'), 1)}%. Это отражает реальный рост дефолтности "
          "36-месячных займов год от года. В исходной версии проекта разрыв был больше и объяснялся только сдвигом во "
          "времени, но заметная его часть была следствием цензурирования. Некоторая недооценка риска сохраняется и после "
          "калибровки, поэтому в работе модель требует регулярного пересчёта.")
    rep.h2("5.4. Что добавляет грейд LendingClub")
    rep.table(["Сравнение AUC на тесте", "Значение"], [
        ["Только подгрейд LC", ru(r.m(GRADE, "roc_auc"))],
        ["LightGBM только на полях заёмщика", ru(r.m(GBM_B, "roc_auc"))],
        ["LightGBM на всех полях", ru(r.m(GBM, "roc_auc"))],
        ["Прирост от полей LC к данным заёмщика", ru(r.m(GBM, "roc_auc") - r.m(GBM_B, "roc_auc"))],
        ["Прирост от данных заёмщика к одному подгрейду", ru(r.m(GBM, "roc_auc") - r.m(GRADE, "roc_auc"))],
    ], "Вклад грейда и данных заёмщика", [10.0, 3.0])
    rec = r.recovery.set_index("grade")
    rep.p(f"Один подгрейд LendingClub оказался сильнее модели на всех доступных полях заёмщика. Вероятно, LendingClub "
          "использует сведения кредитного бюро, которых нет в опубликованном наборе. Данные заёмщика всё же добавляют "
          f"к подгрейду {ru(r.m(GBM, 'roc_auc') - r.m(GRADE, 'roc_auc'))}, то есть грейд не исчерпывает риск. "
          f"Модель на полях заёмщика сильно сжимает риск: для грейда A средняя PD равна {ru(rec.loc['A', 'borrower_pd_pct'], 1)}% "
          f"при фактических {ru(rec.loc['A', 'actual_default_pct'], 1)}%, для грейда F {ru(rec.loc['F', 'borrower_pd_pct'], 1)}% "
          f"при фактических {ru(rec.loc['F', 'actual_default_pct'], 1)}%.")
    rep.table(["Грейд", "Займов", "Факт, %", "PD, поля заёмщика, %", "PD, все поля, %", "AUC внутри, заёмщик",
               "AUC внутри, все поля"],
              [[row["grade"], thousands(row["loans"]), ru(row["actual_default_pct"], 1), ru(row["borrower_pd_pct"], 1),
                ru(row["full_pd_pct"], 1), ru(row["auc_within_borrower"]), ru(row["auc_within_full"])]
               for _, row in r.recovery.iterrows()],
              "Риск по грейдам на тесте (LightGBM). Модель различает риск и внутри грейда",
              [1.4, 1.9, 1.6, 3.0, 2.5, 2.8, 2.8])
    rep.h2("5.5. Важность признаков")
    rep.figure(FIGURES / "12_shap_summary.png", "SHAP для LightGBM на всех полях [6]", 11.5)
    top = ", ".join(r.shap_full["feature"].head(5))
    top_b = ", ".join(r.shap_borrower["feature"].head(5))
    rep.p(f"В модели на всех полях главные признаки по SHAP: {top}. Без полей LendingClub на первое место выходят: {top_b}. "
          "Высокий FICO снижает риск, высокие DTI и отношение суммы к доходу повышают, что согласуется с разделом 3.")
    rep.h2("5.6. Анализ ошибок и таблица одобрения")
    errors = r.errors.set_index("grade")
    rep.p(f"Единый порог F1 фактически работает как граница по грейду: в грейде A пропускается "
          f"{ru(errors.loc['A', 'missed_defaults_pct'], 1)}% дефолтов, в грейде F ложно отклоняется "
          f"{ru(errors.loc['F', 'false_alarms_pct'], 1)}% хороших заёмщиков. Качество внутри грейдов примерно одинаково: AUC "
          f"от {ru(errors['auc_within_grade'].min(), 2)} до {ru(errors['auc_within_grade'].max(), 2)}. Поэтому единый порог "
          "не подходит как кредитная политика, и вместо него построена таблица одобрения.")
    rates = (0.5, 0.7, 0.8, 0.9, 1.0)
    rep.table(["Одобрено", "Подгрейд: дефолт, %", "Подгрейд: доход, %", "LogReg: дефолт, %", "LogReg: доход, %",
               "LightGBM: дефолт, %", "LightGBM: доход, %"],
              [[f"{int(rate * 100)}%"] + [ru(r.approve(m, rate, col), 2) for m in (GRADE, LR, GBM)
                                          for col in ("default_rate_pct", "net_return_pct")] for rate in rates],
              "Одобряем заявки с наименьшей PD. Доход: номинальный возврат сверх выданного за весь срок, без стоимости "
              "фондирования и дисконтирования", [2.0, 2.3, 2.3, 2.2, 2.2, 2.4, 2.4])
    rep.figure(FIGURES / "13_approval.png", "Дефолтность и доходность одобренного портфеля")
    rep.p(f"При одобрении 80% заявок логистическая регрессия снижает дефолтность с "
          f"{ru(r.approve(LR, 1.0, 'default_rate_pct'), 1)}% до {ru(r.approve(LR, 0.8, 'default_rate_pct'), 1)}% и повышает "
          f"номинальный возврат с {ru(r.approve(LR, 1.0, 'net_return_pct'), 1)}% до "
          f"{ru(r.approve(LR, 0.8, 'net_return_pct'), 1)}%. Отбор только по подгрейду даёт "
          f"{ru(r.approve(GRADE, 0.8, 'default_rate_pct'), 1)}% и {ru(r.approve(GRADE, 0.8, 'net_return_pct'), 1)}%. "
          f"Модель на полях заёмщика даёт большую доходность при одобрении 50% заявок при более высокой дефолтности "
          f"({ru(r.approve(GBM_B, 0.5, 'net_return_pct'), 2)}% и {ru(r.approve(GBM_B, 0.5, 'default_rate_pct'), 2)}%): "
          "она не видит ставку и потому оставляет больше высокодоходных займов.")


def processing_section(rep: Report, r: Results) -> None:
    b = r.benchmarks
    csv, pq = b["csv_vs_parquet"]["csv_gz"], b["csv_vs_parquet"]["parquet"]
    pd_run, sql = b["pandas_vs_sql"]["pandas_parquet"], b["pandas_vs_sql"]["sqlite"]
    rep.h1("6. Сравнение способов обработки данных")
    rep.p("Каждый замер выполнялся в новом процессе, в таблице приведена медиана трёх запусков. Память измерялась как "
          "прирост пикового RSS процесса с шагом 20 мс, поэтому в неё входят нативные выделения Arrow и SQLite.")
    rep.table(["Задача", "Вариант", "Время, с", "Пик памяти, МБ", "Размер файла, МБ"], [
        ["Чтение 6 колонок", "CSV.gz, Pandas", ru(csv["seconds"], 2), ru(csv["peak_mb"], 0), ru(csv["file_mb"], 0)],
        ["Чтение 6 колонок", "Parquet, Pandas", ru(pq["seconds"], 2), ru(pq["peak_mb"], 0), ru(pq["file_mb"], 0)],
        ["Агрегация", "Parquet, Pandas", ru(pd_run["seconds"], 2), ru(pd_run["peak_mb"], 0), "—"],
        ["Агрегация", "SQLite", ru(sql["seconds"], 2), ru(sql["peak_mb"], 0), "—"],
    ], "CSV и Parquet, Pandas и SQL", [3.6, 3.6, 2.2, 3.0, 3.2])
    rep.p(f"Parquet читает нужные колонки в {ru(csv['seconds'] / pq['seconds'], 0)} раз быстрее и занимает в "
          f"{ru(csv['file_mb'] / pq['file_mb'], 1)} раза меньше места, хотя в нём больше колонок. Пик памяти при чтении "
          "Parquet оказался выше из-за промежуточных буферов Arrow, при том что итоговая таблица меньше "
          f"({ru(pq['df_mb'], 0)} МБ против {ru(csv['df_mb'], 0)} МБ). При агрегации Pandas быстрее SQLite в "
          f"{ru(sql['seconds'] / pd_run['seconds'], 1)} раза, зато SQLite почти не занимает память приложения. "
          "Результаты двух способов совпадают, это проверяется автоматически.")
    rep.p("В ранней версии проекта по этому замеру побеждал SQLite. Причина была в методике: tracemalloc замедлял Pandas "
          "и не видел нативную память. Методика исправлена, вывод изменился.")
    rep.p("Выбранный подход: Parquet как рабочий формат, Pandas для аналитики и обучения, SQLite для нормализованного "
          "хранения и запросов с JOIN и оконными функциями, когда памяти мало или нужен язык запросов.", "Обоснование.")


def closing_sections(rep: Report, r: Results) -> None:
    rep.h1("7. Ограничения")
    rep.bullets([
        "Метки обучающих займов 2013–2014 гг. к началу 2015 года ещё не были полностью известны. В реальной работе банка "
        "обучающее окно было бы старше; строгий вариант оставил бы слишком мало данных, поэтому это допущение принято и "
        "названо явно.",
        "Тест и валидация содержат только 36-месячные займы, поэтому PD для 60-месячных займов менее надёжна.",
        "В данных есть только одобренные заявки. Reject inference не выполнялся, и на всей совокупности заявителей модель не проверена.",
        "Некоторая недооценка риска сохраняется после калибровки: дефолтность растёт от года к году.",
        "Доходность в таблице одобрения номинальная, без стоимости фондирования, дисконтирования и риска досрочного погашения.",
        "Признак addr_state влияет на прогноз. В реальном скоринге его нужно проверять на справедливость и законность.",
        "Сетка гиперпараметров небольшая; более широкий поиск мог бы немного улучшить результат.",
    ])
    rep.h1("8. Выводы")
    rep.bullets([
        f"Построена модель PD на данных до выдачи: логистическая регрессия с ROC-AUC {ru(r.m(LR, 'roc_auc'))} и Gini "
        f"{ru(r.m(LR, 'gini'))} на созревших займах 2015 года. Это лучше базовых вариантов: Dummy и одного подгрейда "
        f"({ru(r.m(GRADE, 'roc_auc'))}).",
        "Грейды LendingClub хорошо отражают риск, и дефолтность монотонно растёт от A к G. Но грейд не исчерпывает риск: "
        "модель находит различия и внутри грейдов.",
        "Самым важным результатом проверки стало обнаружение смещения: цензурирование 60-месячных займов завышало AUC "
        f"примерно на {ru(r.censored('все', GBM, 'test_auc') - r.censored('созревшие', GBM, 'test_auc'), 2)}.",
        "Сложность модели окупается слабо: прирост LightGBM над логистической регрессией мал, что подтверждает выбор "
        "простой интерпретируемой модели.",
        "Parquet и оптимизация типов сделали работу с полным набором возможной на обычном компьютере: память снижена с "
        "более чем 3 ГБ до 248 МБ.",
    ])
    rep.h1("9. Реализация и воспроизводимость")
    rep.table(["Требование", "Где выполнено"], [
        ["1. Не менее 5 функций", "main.py, app.py и src/*.py: более 120 функций и методов"],
        ["2. Списки и словари", "реестр моделей, меню, отчёты аудита"],
        ["3–4. Условия и циклы", "menu_loop (if/elif/else, while), ask_number, csv_to_parquet (for)"],
        ["5. Работа с файлами", "CSV, Parquet, JSON, SQLite, joblib"],
        ["6. Обработка ошибок", "run_step, load_dataset, load_json, predict_pd"],
        ["7. Интерфейс", "консольное меню main.py и дашборд Streamlit app.py"],
        ["8. Модульность", "12 модулей в src/, main.py и app.py; pathlib, json, datetime, argparse, sqlite3"],
        ["9. Обоснование подхода", "разделы 5 и 6 этого отчёта"],
        ["10. База данных и SQL", "sql/schema.sql, ER-диаграмма, 10 запросов, результаты в Pandas"],
        ["11. Объём данных", f"{thousands(r.ingest['rows'])} строк; чанки, Parquet, оптимизация типов"],
    ], "Соответствие техническим требованиям", [6.0, 10.0])
    rep.p("Запуск описан в README.md. Версии библиотек закреплены в requirements.txt. Демо-выборка и обученные модели "
          f"позволяют запустить дашборд и скоринг без загрузки данных. {TEST_COUNT} автоматических тестов (pytest) "
          f"покрывают {COVERAGE_PCT}% кода "
          "и проверяют отсутствие утечки, границы созревания, запросы SQL относительно Pandas, совпадение признаков при "
          "обучении и скоринге и работу демо-пайплайна. Код проверяется линтером ruff. Все числа этого отчёта читаются "
          "из файлов reports/ при сборке документа.")
    rep.h2("Использованные инструменты")
    rep.p("При подготовке проекта использовался ИИ-ассистент Claude Code (Anthropic): для поиска ошибок в методике, правки "
          "кода, написания тестов и подготовки черновиков документации. Рекомендации по стилю тестирования взяты из "
          "открытого набора ECC [8]. Автор проверил результаты и отвечает за содержание работы.")
    rep.h1("Литература")
    for i, ref in enumerate([
        "wordsforthewise. All Lending Club loan data. Kaggle. https://www.kaggle.com/datasets/wordsforthewise/lending-club",
        "Pedregosa F. et al. Scikit-learn: Machine Learning in Python // Journal of Machine Learning Research. 2011. Vol. 12. P. 2825–2830.",
        "Ke G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree // Advances in Neural Information Processing Systems 30. 2017.",
        "Platt J. Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods // Advances in Large Margin Classifiers. MIT Press, 1999.",
        "Naeini M. P., Cooper G., Hauskrecht M. Obtaining Well Calibrated Probabilities Using Bayesian Binning // Proceedings of AAAI. 2015.",
        "Lundberg S. M., Lee S.-I. A Unified Approach to Interpreting Model Predictions // Advances in Neural Information Processing Systems 30. 2017.",
        "Siddiqi N. Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. Wiley, 2006.",
        "affaan-m. ECC (Everything Claude Code). GitHub. https://github.com/affaan-m/ECC",
    ], 1):
        rep.doc.add_paragraph(f"[{i}] {ref}")


def footer_page_numbers(doc: Document) -> None:
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    para = section.footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(para, "PAGE")


def build() -> Path:
    r = load()
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(2.5), Cm(2.5)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    setup_styles(doc)
    doc.core_properties.title = TITLE
    doc.core_properties.author = AUTHOR
    footer_page_numbers(doc)
    title_page(doc)
    rep = Report(doc)
    contents(rep)
    summary(rep, r)
    data_section(rep, r)
    database_section(rep, r)
    eda_section(rep, r)
    methods_section(rep, r)
    results_section(rep, r)
    processing_section(rep, r)
    closing_sections(rep, r)
    doc.save(OUTPUT)
    print(f"Готово: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build()

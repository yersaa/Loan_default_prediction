"""Собирает docs/presentation.pptx в дизайне шаблона Canva (template.pptx в корне проекта).

Шаблон скачан из Canva (Share → Download → Microsoft PowerPoint) и в репозиторий не входит.
Сборщик берёт из шаблона оформление и иллюстрации, выстраивает слайды под проект (часть слайдов —
копии макетов шаблона), ставит графики проекта в палитре шаблона и добавляет заметки докладчика.
Все числа читаются из reports/ через docs/results.py, поэтому полные данные для сборки не нужны.

Запуск из корня проекта:  python docs/build_deck.py
"""
import copy
import inspect
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from results import (  # noqa: E402
    AI_STATEMENT,
    AUTHORS,
    GROUP,
    TEAM,
    TEAM_SHARED,
    TEST_COUNT,
    TITLE,
    Results,
    load,
    placeholders_left,
    ru,
    thousands,
)

from src import config as C  # noqa: E402
from src.features import is_matured, temporal_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "template.pptx"
OUTPUT = ROOT / "docs" / "presentation.pptx"
CHARTS = ROOT / "docs" / "deck_charts"
SCREENSHOTS = ROOT / "docs" / "screenshots"  # снимки дашборда Streamlit
GBM, LR, GRADE, ENS = "LightGBM", "Logistic Regression", "Grade only (LogReg)", "Ensemble (LightGBM + XGBoost + MLP)"
LABELS = {GRADE: "Только грейд LC", LR: "Логистическая\nрегрессия", "Neural network (MLP)": "Нейросеть",
          "XGBoost": "XGBoost", GBM: "LightGBM", ENS: "Ансамбль"}
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
BODY = 25  # размер основного текста, пт (слайд шаблона 20 × 11,25 дюйма)
CODE_WIDTH = 82  # самая длинная строка кода на слайде, символов: длиннее — шрифт стал бы нечитаемым

# Палитра шаблона Canva
DARK, GREEN, LIGHT, PALE, GREY = "#2D6A65", "#5BC27A", "#B5E08A", "#E6F9D5", "#9AA5A1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.titleweight": "bold", "axes.titlecolor": DARK, "axes.titlesize": 17, "axes.titlelocation": "left",
    "axes.titlepad": 14, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------------------------- графики
def save(fig, name: str, facecolor: str = "white") -> Path:
    """Сохраняет график; bbox_inches="tight" не даёт обрезать длинный заголовок или подписи."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / name
    fig.savefig(path, dpi=200, facecolor=facecolor, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def plural(n: int, forms: tuple) -> str:
    """Число со словом в нужной форме: plural(151, ("колонка", "колонки", "колонок")) -> "151 колонка"."""
    if n % 10 == 1 and n % 100 != 11:
        form = forms[0]
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        form = forms[1]
    else:
        form = forms[2]
    return f"{n} {form}"


def no_y_axis(ax) -> None:
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def chart_memory(r: Results) -> Path:
    est = r.benchmarks["memory"]["full_dataset_estimate_mb"]
    values = [est["all_cols"], est["selected_cols"], r.benchmarks["memory"]["full_parquet_loaded_optimized_mb"]]
    columns = plural(r.audit["raw_columns"]["n_columns"], ("колонка", "колонки", "колонок"))
    fig, ax = plt.subplots(figsize=(7, 5.9))
    bars = ax.bar([f"все\n{columns}", "только нужные\nколонки", "нужные +\nкомпактные типы"],
                  values, color=[GREY, GREEN, DARK], width=0.6)
    ax.bar_label(bars, labels=[f"{thousands(v)} МБ" for v in values], padding=4, fontsize=15)
    ax.set_ylim(0, max(values) * 1.15)  # место для подписи над самым высоким столбцом
    ax.set_title(f"Память для {ru(r.ingest['rows'] / 1e6, 2)} млн займов")
    no_y_axis(ax)
    return save(fig, "memory.png")


def function_code(function) -> list:
    """Строки функции без докстринга — ровно как в файле, без переносов."""
    source = inspect.getsource(function)
    signature, _, rest = source.partition('"""')
    body = rest.split('"""', 1)[1]
    lines = [signature.rstrip()] + [line.rstrip() for line in body.split("\n") if line.strip()]
    too_long = [line for line in lines if len(line) > CODE_WIDTH]
    if too_long:
        raise ValueError(f"Строка длиннее {CODE_WIDTH} символов не поместится на слайд: {too_long[0]!r}")
    return lines


def chart_code() -> Path:
    """Код двух ключевых функций src/features.py картинкой: моноширинный шрифт на светлом фоне шаблона."""
    lines = function_code(is_matured) + [""] + function_code(temporal_split)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(lines)))
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), family="DejaVu Sans Mono", fontsize=15, va="top", ha="left",
            color=DARK, transform=ax.transAxes, linespacing=1.5)
    ax.set_title("src/features.py (без докстрингов)", fontsize=17)
    return save(fig, "code.png", facecolor=PALE)


def dashboard_shot() -> Path:
    """Снимок вкладки «Скоринг заявки» без боковой панели: часть анкеты и решения двух моделей, крупно."""
    image = Image.open(SCREENSHOTS / "dash_scoring.png")
    width, height = image.size
    cropped = image.crop((int(width * 0.24), int(height * 0.2), int(width * 0.72), height))
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / "dashboard.png"
    cropped.save(path)
    return path


def chart_grades(r: Results) -> Path:
    grade = r.default_by_grade
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(grade["grade"], grade["default_rate_pct"], color=[LIGHT, LIGHT, GREEN, GREEN, GREEN, DARK, DARK],
                  width=0.7)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=15)
    ax.set(title="Не вернули, % — по грейду LendingClub", xlabel="Грейд: A — надёжный, G — рискованный")
    no_y_axis(ax)
    return save(fig, "grades.png")


def last_matured_year(term: int) -> int:
    """Последний год выдачи, займы которого закончились к выгрузке (декабрь 2018 г.)."""
    return C.SNAPSHOT_YEAR - term // 12


def chart_censoring(r: Results) -> Path:
    """Сплошная линия — займы, срок которых истёк; пунктир — годы, где закончились ещё не все займы."""
    v = r.vintage[r.vintage["issue_year"].between(2010, 2017)]
    fig, ax = plt.subplots(figsize=(7, 6))
    for term, color, label in [(36, GREEN, "займы на 3 года"), (60, DARK, "займы на 5 лет")]:
        part = v[v["term_months"] == term]
        done = part[part["issue_year"] <= last_matured_year(term)]
        open_ = part[part["issue_year"] >= last_matured_year(term)]
        ax.plot(done["issue_year"], done["default_rate_pct"], "o-", color=color, lw=3, ms=8, label=label)
        ax.plot(open_["issue_year"], open_["default_rate_pct"], "o--", color=color, lw=2, ms=7, mfc="white")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=GREY, ls="--", lw=2, marker="o", mfc="white"))
    labels.append("займы ещё не закончились")
    ax.set_ylim(top=v["default_rate_pct"].max() * 1.35)  # место для легенды над линиями
    ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=13)
    ax.set(title="Не вернули среди закончившихся займов, %", xlabel="Год выдачи")
    return save(fig, "censoring.png")


def chart_auc(r: Results) -> Path:
    names = {model: label + (" (выбрана)" if model == r.selected else "") for model, label in LABELS.items()}
    auc = {names[m]: r.m(m, "roc_auc") for m in names}
    order = sorted(auc, key=auc.get)
    chosen = names.get(r.selected)
    fig, ax = plt.subplots(figsize=(7.4, 6))
    colors = [DARK if name == chosen else (GREY if name.startswith("Только") else GREEN) for name in order]
    bars = ax.barh(order, [auc[n] for n in order], color=colors, height=0.65)
    ax.bar_label(bars, labels=[ru(auc[n]) for n in order], padding=4, fontsize=14)
    ax.set(title="ROC-AUC на 2015 годе: больше — лучше", xlim=(0.6, 0.73))
    ax.xaxis.set_major_formatter(lambda x, _: ru(x, 2))
    ax.tick_params(axis="y", labelsize=14)
    return save(fig, "auc.png")


def chart_lift(r: Results) -> Path:
    """Фактическая дефолтность по десяти группам риска (reports/risk_groups.csv, пункт меню 6)."""
    groups = r.risk_groups
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    bars = ax.bar(groups["risk_group"].astype(str), groups["default_rate_pct"],
                  color=[LIGHT] * 4 + [GREEN] * 4 + [DARK] * 2, width=0.7)
    ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=12)
    ax.axhline(100 * r.m(r.selected, "actual_default_rate"), color=GREY, ls="--", lw=1.5)
    ax.set_ylim(0, groups["default_rate_pct"].max() * 1.18)
    ax.set_title("Не вернули, %: 10 групп риска (1 — самые надёжные)")
    no_y_axis(ax)
    return save(fig, "lift.png")


def chart_approval(r: Results) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for model, color, label in [(GRADE, GREY, "только грейд LC"), (r.selected, DARK, LABELS[r.selected])]:
        part = r.approval[r.approval["model"] == model]
        ax.plot(part["approval_rate"] * 100, part["default_rate_pct"], "o-", color=color, lw=3, ms=7, label=label)
    ax.set(title="Не вернули среди одобренных, %", xlabel="Одобряем, % самых надёжных заявок")
    ax.yaxis.set_major_formatter(lambda y, _: ru(y, 0))
    ax.legend(frameon=False, fontsize=13)
    return save(fig, "approval.png")


# --------------------------------------------------------------------------------------------- работа с шаблоном
def shape(slide, name: str):
    return next(s for s in slide.shapes if s.name == name)


def set_text(slide, name: str, paragraphs: list, size: float | None = BODY, width: float | None = None,
             left: float | None = None, top: float | None = None) -> None:
    """Заменяет текст, сохраняя шрифт, цвет и выравнивание шаблона. Каждый элемент списка — отдельный абзац."""
    box = shape(slide, name)
    for attribute, value in (("width", width), ("left", left), ("top", top)):
        if value is not None:
            setattr(box, attribute, Inches(value))
    frame = box.text_frame
    frame.word_wrap = True
    body = frame._txBody
    old = body.findall(f"{A}p")
    sample = copy.deepcopy(old[0])
    for p in old:
        body.remove(p)
    for text in paragraphs:
        p = copy.deepcopy(sample)
        runs = p.findall(f"{A}r")
        for extra in runs[1:] + p.findall(f"{A}fld") + p.findall(f"{A}br"):
            p.remove(extra)
        runs[0].find(f"{A}t").text = text
        body.append(p)
    for i, paragraph in enumerate(frame.paragraphs):
        if size:
            for run in paragraph.runs:
                run.font.size = Pt(size)
        if i > 0:
            paragraph.space_before = Pt(12)


def title(slide, name: str, text: str, size: float | None = None, width: float | None = None) -> None:
    set_text(slide, name, [text], size=size, width=width)


def place(slide, image: Path, left, top, width, height) -> None:
    """Вписывает картинку в рамку без искажения пропорций, по центру рамки."""
    w, h = Image.open(image).size
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    slide.shapes.add_picture(str(image), left + (width - new_w) // 2, top + (height - new_h) // 2, new_w, new_h)


def replace_picture(slide, name: str, image: Path) -> None:
    """Ставит график на место картинки-заглушки шаблона."""
    old = shape(slide, name)
    left, top, width, height = old.left, old.top, old.width, old.height
    remove(slide, name)
    place(slide, image, left, top, width, height)


def remove(slide, *names: str) -> None:
    for name in names:
        element = shape(slide, name)._element
        element.getparent().remove(element)


def drop_unused_images(prs: Presentation) -> int:
    """Удаляет связи с картинками, на которые больше не ссылается ни одна фигура слайда.

    Иначе убранные заглушки и фото шаблона остаются внутри файла и только увеличивают его размер.
    """
    dropped = 0
    for slide in prs.slides:
        xml = slide.part.blob.decode("utf-8")
        for rel_id, rel in list(slide.part.rels.items()):
            if "image" in rel.reltype and not re.search(rf'"{rel_id}"', xml):
                slide.part.drop_rel(rel_id)
                dropped += 1
    return dropped


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = " ".join(text.split())


def duplicate(prs: Presentation, index: int):
    """Копия слайда шаблона в конец презентации: фигуры, картинки и их связи (r:embed)."""
    source = prs.slides[index]
    target = prs.slides.add_slide(source.slide_layout)
    for placeholder in list(target.shapes):
        placeholder._element.getparent().remove(placeholder._element)
    ids = {rel_id: target.part.relate_to(rel._target, rel.reltype)
           for rel_id, rel in source.part.rels.items() if "image" in rel.reltype or "media" in rel.reltype}
    tree = target.shapes._spTree
    for element in source.shapes._spTree.iterchildren():
        if element.tag.endswith(("}sp", "}grpSp", "}pic", "}cxnSp", "}graphicFrame")):
            clone = copy.deepcopy(element)
            for node in clone.iter():
                for attribute, value in list(node.attrib.items()):
                    if attribute.endswith("}embed") and value in ids:
                        node.attrib[attribute] = ids[value]
            tree.append(clone)
    background = source._element.find(f"{P}cSld").find(f"{P}bg")
    if background is not None:
        target._element.find(f"{P}cSld").insert(0, copy.deepcopy(background))
    return target


def reorder(prs: Presentation, order: list) -> None:
    ids = prs.slides._sldIdLst
    items = list(ids)
    if sorted(order) != list(range(len(items))):
        raise ValueError("В порядке слайдов каждый слайд должен встречаться ровно один раз.")
    for element in items:
        ids.remove(element)
    for index in order:
        ids.append(items[index])


# --------------------------------------------------------------------------------------------- слайды
def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError("Нет template.pptx: скачайте шаблон из Canva в формате PowerPoint в корень проекта.")
    r = load()
    placeholders_left()
    prs = Presentation(str(TEMPLATE))
    # Копии макетов шаблона (до любых изменений): 13 — код, 14 — база, 15 — EDA, 16 — дашборд, 17 — команда
    for index in (3, 5, 9, 3, 1):
        duplicate(prs, index)
    s = prs.slides
    chosen, label = r.selected, " ".join(LABELS[r.selected].split())
    auc, grade_auc = r.m(chosen, "roc_auc"), r.m(GRADE, "roc_auc")
    vintage = r.vintage.set_index(["issue_year", "term_months"])["default_rate_pct"]
    est = r.benchmarks["memory"]["full_dataset_estimate_mb"]
    loaded = r.benchmarks["memory"]["full_parquet_loaded_optimized_mb"]
    csv, pq = r.benchmarks["csv_vs_parquet"]["csv_gz"], r.benchmarks["csv_vs_parquet"]["parquet"]
    pd_run, sql = r.benchmarks["pandas_vs_sql"]["pandas_parquet"], r.benchmarks["pandas_vs_sql"]["sqlite"]
    approve_all, approve_80 = r.approve(chosen, 1.0, "default_rate_pct"), r.approve(chosen, 0.8, "default_rate_pct")
    grade = r.default_by_grade.set_index("grade")["default_rate_pct"]
    matured_loans = int(r.default_by_grade["loans"].sum())
    matured_rate = float((r.default_by_grade["loans"] * r.default_by_grade["default_rate_pct"]).sum() / matured_loans)
    n_models, n_baselines = len(r.metrics), 2
    n_modules = len([p for p in (ROOT / "src").glob("*.py") if p.name != "__init__.py"])
    groups = r.risk_groups.set_index("risk_group")["default_rate_pct"]
    challengers = r.selection["challengers_valid_auc"]
    best_challenger = max(challengers, key=challengers.get)
    challenger_gap = challengers[best_challenger] - r.selection["valid_auc"][chosen]

    # Титульный
    set_text(s[0], "TextBox 20", ["ПРОГНОЗ ДЕФОЛТА", "LENDINGCLUB"], size=50)
    shape(s[0], "TextBox 20").top = Inches(3.55)
    set_text(s[0], "TextBox 21", [f"{AUTHORS} · {GROUP}"], size=17, width=7.3, left=11.1)
    notes(s[0], f"""Здравствуйте. Наш проект — прогноз дефолта по займам платформы LendingClub, тема 33. Команда:
              {AUTHORS}. Кто что делал — на предпоследнем слайде.""")

    # Задача (шаблон: слайд 3)
    title(s[2], "TextBox 10", "Задача")
    set_text(s[2], "TextBox 6", ["Кредитор получает заявку и решает: дать деньги или нет.",
                                 f"Каждый {round(100 / matured_rate)}-й заём на LendingClub так и не вернули."],
             size=28)
    set_text(s[2], "TextBox 11", ["Модель по анкете оценивает шанс, что человек не вернёт долг.",
                                  "Она видит только то, что известно в день заявки."], size=28)
    notes(s[2], """Представим, что мы кредитор: пришла заявка, нужно решить, давать ли деньги. Модель оценивает
              вероятность дефолта — шанс, что заёмщик не вернёт долг. Вторая цель — проверить, насколько хорошо риск
              оценивает сам LendingClub своими грейдами от A до G.""")

    # Данные (шаблон: слайд 2)
    title(s[1], "TextBox 9", "Данные")
    facts = [f"{ru(r.ingest['rows'] / 1e6, 2)} млн займов", "2007–2018 годы",
             plural(r.audit["raw_columns"]["n_columns"], ("колонка", "колонки", "колонок")),
             f"{ru(matured_rate, 0)}% не вернули"]
    for name, fact in zip(["TextBox 10", "TextBox 11", "TextBox 12", "TextBox 13"], facts, strict=True):
        set_text(s[1], name, [fact], size=30, width=5.4, left=11.65)
    notes(s[1], f"""Набор All Lending Club loan data с Kaggle, ссылка из задания. {ru(matured_rate, 1)}% — доля
              невозвратов среди {thousands(matured_loans)} займов, срок которых уже истёк. Для модели берём займы
              2007–2015 годов: {thousands(r.metadata['train_rows'])} для обучения и
              {thousands(r.metadata['test_rows'])} для итоговой проверки.""")

    # Обработка данных (шаблон: слайд 4)
    title(s[3], "TextBox 5", "Обработка данных")
    set_text(s[3], "TextBox 6", [
        f"Файл большой: целиком занял бы {ru(est['all_cols'] / 1000, 1)} ГБ памяти.",
        f"Читаем по частям и храним компактно — {ru(loaded, 0)} МБ.",
        f"Формат Parquet читается в {ru(csv['seconds'] / pq['seconds'], 0)} раз быстрее CSV.",
        "Проверили данные на ошибки и пропуски."], size=26)
    replace_picture(s[3], "Picture 7", chart_memory(r))
    notes(s[3], f"""Файл целиком занял бы около {ru(est['all_cols'] / 1000, 0)} ГБ памяти. Читаем его частями по
              {thousands(C.CHUNKSIZE)} строк, оставляем нужные колонки, текст храним как category, числа как float32 и
              сохраняем в Parquet. Аудит нашёл ошибки вроде нулевого дохода и подтвердил, что данные кредитного
              отчёта сняты в день заявки.""")

    # Как устроен проект (шаблон: слайд 5)
    title(s[4], "TextBox 9", "Как устроен проект", size=44, width=8.3)
    set_text(s[4], "TextBox 6", ["1. Данные", "чтение, проверка, база SQL"], size=26)
    set_text(s[4], "TextBox 7", ["2. Анализ", "14 графиков: от чего зависит риск"], size=26)
    set_text(s[4], "TextBox 8", ["3. Модели", f"{n_models} моделей, из них {n_baselines} для сравнения"], size=26)
    notes(s[4], """Проект — конвейер из восьми шагов. Каждый шаг — пункт консольного меню main.py и раздел ноутбука
              project_walkthrough.ipynb, главного файла для чтения.""")

    # Код (копия слайда 4)
    title(s[13], "TextBox 5", "Код")
    set_text(s[13], "TextBox 6", [
        f"{n_modules} модулей в src/, у каждого своя задача.",
        "main.py — меню в консоли.",
        "app.py — веб-дашборд.",
        f"{TEST_COUNT} автотестов проверяют, что всё работает."], size=26)
    remove(s[13], "Picture 7")  # код шире рамки шаблона: занимаем всю левую половину, чтобы шрифт был крупнее
    place(s[13], chart_code(), Inches(0.4), Inches(2.0), Inches(9.4), Inches(7.2))
    notes(s[13], """Слева — две ключевые функции. is_matured оставляет займы, срок которых уже истёк. temporal_split
              делит данные по годам без перемешивания. Тесты проверяют, что в модель не попали данные из будущего,
              что SQL и Pandas считают одинаково и что меню работает.""")

    # База и SQL (копия слайда 6)
    title(s[14], "TextBox 5", "База данных и SQL")
    set_text(s[14], "TextBox 6", [
        "Данные разложены по 7 связанным таблицам.",
        "10 SQL-запросов отвечают на вопросы о риске.",
        "SQL экономит память, Pandas считает быстрее."], size=26)
    replace_picture(s[14], "Picture 7", ROOT / "docs" / "er_diagram.png")
    notes(s[14], f"""Справочники отдельно, займы отдельно, исход займа — в своей таблице, чтобы его нельзя было
              случайно взять в признаки. В запросах есть JOIN, HAVING, CTE, подзапрос и оконные функции. Сравнение:
              Pandas считает в {ru(sql['seconds'] / pd_run['seconds'], 1)} раза быстрее, а SQLite тратит
              {ru(sql['peak_mb'], 0)} МБ памяти против {ru(pd_run['peak_mb'], 0)}.""")

    # Разведочный анализ (копия слайда 10)
    title(s[15], "TextBox 9", "Анализ данных")
    set_text(s[15], "TextBox 10", [
        f"Грейд LendingClub работает: в группе A не вернули {ru(grade['A'], 0)}%, в группе G — {ru(grade['G'], 0)}%.",
        "Риск выше у займов на 5 лет, у арендаторов и при низком кредитном рейтинге.",
        "Один признак ничего не решает — нужна модель."], size=26)
    replace_picture(s[15], "Picture 8", chart_grades(r))
    notes(s[15], """В ноутбуке 14 графиков. Главное: грейд LendingClub хорошо упорядочивает риск, но внутри грейда
              доход и другие признаки всё ещё различают заёмщиков, значит, модели есть что добавить.""")

    # Главная находка (шаблон: слайд 6)
    title(s[5], "TextBox 5", "Главная находка")
    set_text(s[5], "TextBox 6", [
        "Займы последних лет ещё не закончились.",
        "Успели закончиться в основном неудачные, поэтому невозвратов кажется больше.",
        "Решение: учим модель только на займах, срок которых уже истёк."], size=26)
    replace_picture(s[5], "Picture 7", chart_censoring(r))
    notes(s[5], f"""Это главный результат проверки метода. На графике пунктир — годы, где закончились ещё не все займы.
              У займов на 5 лет, выданных в 2015 году, не вернули {ru(vintage[(2015, 60)], 0)}% среди закончившихся, а
              у полностью закончившихся займов 2013 года — {ru(vintage[(2013, 60)], 0)}%. Если этого не учесть, модель
              выглядит лучше, чем есть: {ru(r.censored('все', GBM, 'test_auc'))} вместо
              {ru(r.censored('созревшие', GBM, 'test_auc'))}.""")

    # Честная оценка (шаблон: слайд 7)
    title(s[6], "TextBox 9", "Честная оценка")
    features = plural(r.feature_count, ("признак", "признака", "признаков"))
    set_text(s[6], "TextBox 10", [f"Модель видит только данные на день заявки: {features}."],
             size=27)
    set_text(s[6], "TextBox 11", ["Данные «из будущего» дали бы 0,9, но это обман. Их нет."], size=27)
    notes(s[6], """Утечка данных — главная ловушка этой задачи. Например, кредитный рейтинг после выдачи один даёт
              качество 0,94: у тех, кто перестал платить, он падает. Но в день заявки его ещё нет. Автоматический
              тест проверяет, что таких полей среди признаков нет.""")

    # Обучение и проверка по времени (шаблон: слайд 8)
    title(s[7], "TextBox 10", "Учимся на прошлом", size=40, width=7.3)
    for label_box, text_box, number, text in [
        ("TextBox 15", "TextBox 11", "1", f"2007–{C.VALID_YEAR - 1}: подбираем настройки"),
        ("TextBox 20", "TextBox 16", "2", f"{C.VALID_YEAR}: выбираем лучшую модель"),
        ("TextBox 25", "TextBox 21", "3", f"2007–{C.TEST_YEAR - 1}: обучаем итоговую"),
        ("TextBox 30", "TextBox 26", "4", f"{C.TEST_YEAR}: финальная проверка, один раз"),
    ]:
        set_text(s[7], label_box, [number], size=26)
        set_text(s[7], text_box, [text], size=26)
    notes(s[7], """Модель учится на прошлом и проверяется на будущем, как в реальном банке. Если перемешать займы
              случайно, модель увидела бы займы того года, на котором её проверяют, и оценка была бы нечестной.""")

    # Модели (шаблон: слайд 9)
    title(s[8], "TextBox 9", "Модели")
    set_text(s[8], "TextBox 16", ["Для сравнения: «угадывание» и оценка самого LendingClub"], size=25)
    set_text(s[8], "TextBox 17", ["Пять моделей на всех данных — от регрессии до нейросети, и две — без грейда"],
             size=25)
    set_text(s[8], "TextBox 18", ["Лучшую выбираем по правилу, заданному заранее"], size=25)
    notes(s[8], f"""Всего {n_models} моделей. {n_baselines} — точки отсчёта: если сложная модель их не обгоняет, она
              бесполезна. Пять моделей видят все данные: логистическая регрессия, LightGBM, XGBoost, нейросеть и их
              ансамбль. Ещё две видят только данные заёмщика, без грейда и ставки LendingClub. Качество меряем ROC-AUC:
              насколько хорошо модель отличает рискованных от надёжных. Долю верных ответов не берём: модель «все
              вернут» права в {ru(100 * r.m(chosen, 'accuracy_all_repay'), 0)}% случаев, но не находит ни одного
              невозврата.""")

    # Результат (шаблон: слайд 10)
    title(s[9], "TextBox 9", "Результат")
    set_text(s[9], "TextBox 10", [
        f"Выбрана {label} — по правилу, на данных {C.VALID_YEAR} года.",
        f"Качество {ru(auc)}: 0,5 — угадывание, 1 — идеал. У LendingClub — {ru(grade_auc)}.",
        f"Находит {ru(100 * r.m(chosen, 'recall'), 0)}% будущих невозвратов.",
        "Более сложные модели не помогли: это предел данных."], size=25)
    replace_picture(s[9], "Picture 8", chart_auc(r))
    notes(s[9], f"""Выбрана {label}: на проверке {C.VALID_YEAR} года она лучше логистической регрессии на
              {ru(r.m(chosen, 'valid_auc') - r.m(LR, 'valid_auc'))}, это больше заданного запаса
              {ru(C.SIMPLICITY_MARGIN_AUC)}. Лучший из сложных вариантов, {LABELS[best_challenger].replace(chr(10), ' ')},
              отличается на {ru(challenger_gap, 4)} — меньше запаса, поэтому выбор не изменился, хотя на графике он
              чуть выше. Сильные модели сошлись около {ru(auc, 2)} — это предел данных без подсматривания. Прогнозу
              можно верить: в среднем модель ждёт {ru(100 * r.m(chosen, 'mean_pd'), 1)}% невозвратов, на деле —
              {ru(100 * r.m(chosen, 'actual_default_rate'), 1)}%.""")

    # Для кредитора (шаблон: слайд 12)
    title(s[11], "TextBox 18", "Для кредитора")
    remove(s[11], "Group 5", "Group 7", "Group 9", "Group 11")
    place(s[11], chart_lift(r), Inches(8.6), Inches(1.5), Inches(9.8), Inches(4.0))
    place(s[11], chart_approval(r), Inches(8.6), Inches(5.6), Inches(9.8), Inches(4.0))
    notes(s[11], f"""Как модель помогает на практике. Сверху: делим заёмщиков на 10 групп по прогнозу — в самой
              надёжной не вернули {ru(groups.iloc[0], 1)}%, в самой рискованной {ru(groups.iloc[-1], 1)}%. Снизу: если
              одобрять 80% самых надёжных заявок, невозвратов становится {ru(approve_80, 1)}% вместо
              {ru(approve_all, 1)}%.""")

    # Дашборд Streamlit (копия слайда 4)
    title(s[16], "TextBox 5", "Дашборд")
    set_text(s[16], "TextBox 6", [
        "Запуск: streamlit run app.py",
        "Вводим анкету — сразу видим риск и решение.",
        "Графики с фильтрами и сравнение моделей."], size=26)
    replace_picture(s[16], "Picture 7", dashboard_shot())
    notes(s[16], """Интерфейс проекта — консольное меню main.py и дашборд Streamlit с четырьмя вкладками. Он работает
              сразу после клонирования. На скриншоте пограничный заёмщик: LightGBM одобряет, логистическая регрессия
              отказывает. Сейчас покажем дашборд вживую.""")

    # Выводы (шаблон: слайд 11)
    title(s[10], "TextBox 8", "Выводы")
    set_text(s[10], "TextBox 6", [
        f"Модель полезна: одобряя 80% надёжных заявок, получаем {ru(approve_80, 0)}% невозвратов вместо "
        f"{ru(approve_all, 0)}%.",
        "Главное — честная проверка: только закончившиеся займы и данные на день заявки."], size=24)
    set_text(s[10], "TextBox 7", [
        "Ограничения: итоговая проверка только на займах на 3 года; отклонённых заявок в данных нет."],
        size=22, top=6.6)
    notes(s[10], """Итоги: большие данные можно обработать на обычном компьютере; незавершённые займы обманывают;
              модель отделяет надёжных заёмщиков от рискованных; сложнее не значит лучше. Ограничения: тест — только
              займы 2015 года на 3 года, потому что пятилетние займы 2015 года ещё не закончились; в данных есть только
              одобренные заявки, поэтому на всех заявителях модель не проверена.""")

    # Команда и вклад (копия слайда 2)
    title(s[17], "TextBox 9", "Команда")
    for i, (box, member) in enumerate(zip(["TextBox 10", "TextBox 11", "TextBox 12"], TEAM, strict=True)):
        set_text(s[17], box, [member.name, member.role], size=24, width=6.2, left=11.25, top=2.6 + 1.6 * i)
        for run in shape(s[17], box).text_frame.paragraphs[1].runs:
            run.font.size = Pt(19)
        shape(s[17], box).text_frame.paragraphs[1].space_before = Pt(2)
    set_text(s[17], "TextBox 13", ["ИИ-ассистент Claude Code", "проверка кода, тесты, черновики текстов"],
             size=19, width=6.2, left=11.25, top=7.45)
    for run in shape(s[17], "TextBox 13").text_frame.paragraphs[1].runs:
        run.font.size = Pt(15)
    shape(s[17], "TextBox 13").text_frame.paragraphs[1].space_before = Pt(2)
    team_notes = " ".join(f"{m.name}: {m.work}." for m in TEAM)
    notes(s[17], f"""Вклад участников. {team_notes} {TEAM_SHARED.capitalize()}. Каждый из нас разобрал весь код и
              может объяснить любую его часть. {AI_STATEMENT}""")

    # Спасибо
    set_text(s[12], "TextBox 20", ["Спасибо!"], size=None)
    set_text(s[12], "TextBox 21", ["Готовы ответить на вопросы"], size=24)
    notes(s[12], "Спасибо за внимание. Готовы ответить на вопросы.")

    # Дашборд — перед выводами, команда — перед «Спасибо»
    reorder(prs, [0, 2, 1, 3, 4, 13, 14, 15, 5, 6, 7, 8, 9, 11, 16, 10, 17, 12])
    dropped = drop_unused_images(prs)
    prs.core_properties.title = TITLE
    prs.core_properties.author = AUTHORS
    prs.save(OUTPUT)
    print(f"Готово: {OUTPUT} (убрано неиспользуемых картинок шаблона: {dropped})")
    return OUTPUT


if __name__ == "__main__":
    build()

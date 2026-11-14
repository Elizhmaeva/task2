"""Эксперименты с поиском ассоциативных правил на наборе данных baskets.csv.

Порог поддержки фиксирован, порог достоверности меняется от 70 % до 95 %
с шагом 5 %. Для каждого порога измеряется время поиска правил и число
найденных правил. Результаты сохраняются в папку results:
    support_selection.csv - обоснование выбора порога поддержки;
    experiments.csv       - сводная таблица экспериментов;
    rules_XX.csv          - список правил для порога достоверности XX %;
    time.png              - диаграмма быстродействия;
    rules_count.png       - диаграмма количества найденных правил.

Запуск:  python experiments.py
"""

import csv
import os
import statistics
import time

import matplotlib.pyplot as plt

from apriori import apriori, load_transactions
from association_rules import generate_rules, save_rules, sort_rules

DATASET = "baskets.csv"
MIN_SUPPORT = 0.001                                  # фиксированный порог поддержки
CONFIDENCES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]   # пороги достоверности
MAX_ITEMS = 7                                        # не более 7 объектов в правиле
REPEATS = 15                                         # число запусков на каждый порог
SUPPORTS_TO_COMPARE = [0.10, 0.05, 0.01, 0.005, 0.001]
RESULTS_DIR = "results"
# подпись порога поддержки для заголовков диаграмм: "0,1 %"
SUPPORT_LABEL = f"{MIN_SUPPORT * 100:g} %".replace(".", ",")

# цвета диаграмм
BLUE = "#2a78d6"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def compare_supports(transactions):
    """Показывает, какие правила возможны при разных порогах поддержки.

    Параметры:
        transactions (list[tuple]) - список транзакций.

    Возвращает:
        list[dict] - по одному словарю на каждый порог поддержки с ключами:
            min_support    - порог поддержки (доля);
            itemsets       - число частых наборов из двух и более объектов;
            max_confidence - наибольшая достоверность среди всех правил
                             (0, если правил нет);
            rules_70       - число правил с достоверностью не ниже 70 %.
    """
    rows = []
    for min_support in SUPPORTS_TO_COMPARE:
        frequent, _ = apriori(transactions, min_support)
        # порог достоверности 0 - получаем все возможные правила
        rules = generate_rules(frequent, 0.0, MAX_ITEMS)
        rows.append({
            "min_support": min_support,
            "itemsets": sum(1 for itemset in frequent if len(itemset) >= 2),
            "max_confidence": max((rule["confidence"] for rule in rules), default=0),
            "rules_70": len(generate_rules(frequent, 0.70, MAX_ITEMS)),
        })
    return rows


def run_experiments(transactions):
    """Запускает поиск правил для каждого порога достоверности.

    Параметры:
        transactions (list[tuple]) - список транзакций.

    Возвращает:
        list[dict] - по одному словарю на каждый порог с ключами:
            min_confidence - порог достоверности (доля);
            apriori_ms     - время поиска частых наборов, мс (медиана);
            rules_ms       - время генерации правил, мс (медиана);
            total_ms       - полное время поиска правил, мс (медиана);
            rules          - список найденных правил.
    """
    apriori_times = {c: [] for c in CONFIDENCES}
    rules_times = {c: [] for c in CONFIDENCES}
    total_times = {c: [] for c in CONFIDENCES}
    found = {}
    # Пороги перебираются внутри каждого повтора, чтобы случайные колебания
    # загрузки компьютера одинаково сказывались на всех порогах.
    for _ in range(REPEATS):
        for min_confidence in CONFIDENCES:
            start = time.perf_counter()
            frequent, _ = apriori(transactions, MIN_SUPPORT)
            middle = time.perf_counter()
            rules = generate_rules(frequent, min_confidence, MAX_ITEMS)
            end = time.perf_counter()
            apriori_times[min_confidence].append(middle - start)
            rules_times[min_confidence].append(end - middle)
            total_times[min_confidence].append(end - start)
            found[min_confidence] = rules

    rows = []
    for min_confidence in CONFIDENCES:
        rows.append({
            "min_confidence": min_confidence,
            "apriori_ms": round(statistics.median(apriori_times[min_confidence]) * 1000, 1),
            "rules_ms": round(statistics.median(rules_times[min_confidence]) * 1000, 2),
            "total_ms": round(statistics.median(total_times[min_confidence]) * 1000, 1),
            "rules": found[min_confidence],
        })
    return rows


def save_tables(support_rows, rows):
    """Сохраняет в папку результатов таблицы и списки правил.

    Параметры:
        support_rows (list[dict]) - результат функции compare_supports;
        rows (list[dict]) - результат функции run_experiments.

    Возвращаемого значения нет.
    """
    path = os.path.join(RESULTS_DIR, "support_selection.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["порог поддержки, %", "частых наборов из 2+ объектов",
                         "наибольшая достоверность", "правил с достоверностью >= 70 %"])
        for row in support_rows:
            writer.writerow([f"{row['min_support'] * 100:g}", row["itemsets"],
                             f"{row['max_confidence']:.3f}", row["rules_70"]])

    path = os.path.join(RESULTS_DIR, "experiments.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["порог достоверности, %", "поиск частых наборов, мс",
                         "генерация правил, мс", "всего, мс", "найдено правил"])
        for row in rows:
            writer.writerow([f"{row['min_confidence'] * 100:g}", f"{row['apriori_ms']:.1f}",
                             f"{row['rules_ms']:.2f}", f"{row['total_ms']:.1f}",
                             len(row["rules"])])

    for row in rows:
        percent = round(row["min_confidence"] * 100)
        path = os.path.join(RESULTS_DIR, f"rules_{percent}.csv")
        save_rules(sort_rules(row["rules"], "support"), path)


def style_axes(ax):
    """Общее оформление диаграммы: без рамки, только горизонтальная сетка.

    Параметры:
        ax (matplotlib.axes.Axes) - оси диаграммы, которые нужно оформить.

    Возвращаемого значения нет.
    """
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_time(rows):
    """Диаграмма 1: полное время поиска правил при разных порогах достоверности.

    Параметры:
        rows (list[dict]) - результат функции run_experiments.

    Возвращаемого значения нет, диаграмма сохраняется в файл time.png.
    """
    labels = [f"{row['min_confidence'] * 100:g} %" for row in rows]
    values = [row["total_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels, values, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    for x, value in enumerate(values):
        ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=11, color=TEXT)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_xlabel("Порог достоверности", fontsize=11, color=MUTED)
    ax.set_ylabel("Время работы, мс", fontsize=11, color=MUTED)
    ax.set_title(f"Время поиска правил (порог поддержки {SUPPORT_LABEL})",
                 fontsize=13, color=TEXT, loc="left", pad=14)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "time.png"), dpi=200, facecolor="white")
    plt.close(fig)


def plot_rules_count(rows):
    """Диаграмма 2: количество найденных правил при разных порогах достоверности.

    Параметры:
        rows (list[dict]) - результат функции run_experiments.

    Возвращаемого значения нет, диаграмма сохраняется в файл rules_count.png.
    """
    labels = [f"{row['min_confidence'] * 100:g} %" for row in rows]
    values = [len(row["rules"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, width=0.6, color=BLUE)
    for x, value in enumerate(values):
        ax.annotate(str(value), (x, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=11, color=TEXT)
    ax.set_ylim(0, max(values) * 1.12)
    ax.set_xlabel("Порог достоверности", fontsize=11, color=MUTED)
    ax.set_ylabel("Количество правил", fontsize=11, color=MUTED)
    ax.set_title(f"Количество найденных правил (порог поддержки {SUPPORT_LABEL})",
                 fontsize=13, color=TEXT, loc="left", pad=14)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rules_count.png"), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    """Точка входа: проводит эксперименты, выводит таблицы на экран,
    сохраняет таблицы, списки правил и диаграммы.

    Параметров и возвращаемого значения нет.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    transactions = load_transactions(DATASET)
    print(f"Транзакций: {len(transactions)}")

    support_rows = compare_supports(transactions)
    print("\nВыбор порога поддержки")
    print(f"{'Поддержка':>10} {'Наборов 2+':>11} {'Макс. дост.':>12} {'Правил >= 70 %':>15}")
    for row in support_rows:
        print(f"{row['min_support'] * 100:>9g}% {row['itemsets']:>11} "
              f"{row['max_confidence']:>12.3f} {row['rules_70']:>15}")

    rows = run_experiments(transactions)
    print(f"\nЭксперименты при пороге поддержки {MIN_SUPPORT * 100:g} %")
    print(f"{'Достоверность':>14} {'Apriori, мс':>12} {'Правила, мс':>12} "
          f"{'Всего, мс':>10} {'Правил':>7}")
    for row in rows:
        print(f"{row['min_confidence'] * 100:>13g}% {row['apriori_ms']:>12.1f} "
              f"{row['rules_ms']:>12.2f} {row['total_ms']:>10.1f} {len(row['rules']):>7}")

    save_tables(support_rows, rows)
    plot_time(rows)
    plot_rules_count(rows)
    print(f"\nТаблицы, списки правил и диаграммы сохранены в папку {RESULTS_DIR}")


if __name__ == "__main__":
    main()

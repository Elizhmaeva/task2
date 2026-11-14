"""Поиск частых наборов объектов с помощью алгоритма Apriori.

Запуск из командной строки:

    python apriori.py baskets.csv 0.05 --order support
    python apriori.py baskets.csv 0.03 --order lex --output result.csv

Параметры программы:
    набор данных  - CSV-файл, одна строка = одна транзакция (товары через запятую);
    порог         - минимальная поддержка, доля от 0 до 1 (0.05 = 5 %);
    --order       - способ упорядочивания результата:
                    support - по убыванию значения поддержки,
                    lex     - лексикографический.
"""

import argparse
import csv
import math
from itertools import combinations


def load_transactions(path):
    """Читает набор данных из CSV-файла.

    Параметры:
        path (str) - путь к CSV-файлу; одна строка файла - одна транзакция,
                     названия товаров разделены запятыми.

    Возвращает:
        list[tuple[str, ...]] - список транзакций. Каждая транзакция - кортеж
        уникальных названий товаров, упорядоченных лексикографически
        (алгоритм Apriori опирается на этот порядок).
    """
    # baskets.csv сохранён в кодировке Windows-1251; на случай, если файл
    # пересохранят в UTF-8, сначала пробуем её.
    try:
        with open(path, encoding="utf-8", newline="") as file:
            rows = list(csv.reader(file))
    except UnicodeDecodeError:
        with open(path, encoding="cp1251", newline="") as file:
            rows = list(csv.reader(file))

    transactions = []
    for row in rows:
        # множество убирает повторы товара внутри одной покупки
        items = {item.strip() for item in row if item.strip()}
        if items:
            transactions.append(tuple(sorted(items)))
    return transactions


def find_frequent_1_itemsets(transactions, min_count):
    """Первый проход по данным: находит частые 1-наборы L1.

    Параметры:
        transactions (list[tuple]) - список транзакций;
        min_count (int) - минимальное число транзакций, в которых должен
                          встречаться набор, чтобы считаться частым.

    Возвращает:
        dict - {1-набор (кортеж из одного товара): число транзакций,
        содержащих этот товар}; только для частых 1-наборов.
    """
    counts = {}
    for transaction in transactions:
        for item in transaction:
            counts[(item,)] = counts.get((item,), 0) + 1
    return {itemset: n for itemset, n in counts.items() if n >= min_count}


def has_infrequent_subset(candidate, prev_frequent):
    """Проверяет, есть ли у k-набора редкое (k-1)-подмножество.

    Параметры:
        candidate (tuple) - набор-кандидат из k товаров;
        prev_frequent (dict) - частые (k-1)-наборы L(k-1) (ключи словаря).

    Возвращает:
        bool - True, если хотя бы одно (k-1)-подмножество кандидата не входит
        в L(k-1), то есть кандидат заведомо редкий; иначе False.
    """
    for subset in combinations(candidate, len(candidate) - 1):
        if subset not in prev_frequent:
            return True
    return False


def apriori_gen(prev_frequent):
    """Строит кандидатов C(k) из частых (k-1)-наборов L(k-1).

    Шаг соединения: два набора соединяются, если у них совпадают первые
    k-2 элемента, а последний элемент первого набора меньше последнего
    элемента второго.
    Шаг отбрасывания: кандидат удаляется, если хотя бы одно его
    (k-1)-подмножество не является частым (принцип Априори).

    Параметры:
        prev_frequent (dict) - частые (k-1)-наборы L(k-1) (ключи словаря),
                               каждый набор - упорядоченный кортеж товаров.

    Возвращает:
        list[tuple] - список кандидатов C(k), каждый - упорядоченный кортеж
        из k товаров.
    """
    itemsets = sorted(prev_frequent)
    candidates = []
    for i, first in enumerate(itemsets):
        for second in itemsets[i + 1:]:
            # Список отсортирован, поэтому наборы с одинаковым началом идут
            # подряд: как только начало изменилось, дальше искать незачем.
            if first[:-1] != second[:-1]:
                break
            candidate = first + (second[-1],)
            if not has_infrequent_subset(candidate, prev_frequent):
                candidates.append(candidate)
    return candidates


def count_support(candidates, transactions, k):
    """Проход по данным: подсчитывает поддержку кандидатов.

    Параметры:
        candidates (list[tuple]) - кандидаты C(k);
        transactions (list[tuple]) - список транзакций;
        k (int) - длина кандидатов.

    Возвращает:
        dict - {кандидат: число транзакций, содержащих все его товары}.
    """
    counts = dict.fromkeys(candidates, 0)
    # товары, которые входят хотя бы в одного кандидата
    useful_items = {item for candidate in candidates for item in candidate}
    for transaction in transactions:
        items = [item for item in transaction if item in useful_items]
        if len(items) < k:
            continue
        # subset(Ck, t): k-подмножества транзакции, являющиеся кандидатами
        for subset in combinations(items, k):
            if subset in counts:
                counts[subset] += 1
    return counts


def apriori(transactions, min_support):
    """Алгоритм Apriori: находит все частые наборы.

    Параметры:
        transactions (list[tuple]) - список транзакций, товары в каждой
                                     упорядочены лексикографически;
        min_support (float) - порог поддержки, доля от 0 до 1.

    Возвращает кортеж из двух словарей:
        frequent (dict) - {набор (кортеж товаров): поддержка (доля от 0 до 1)}
                          для всех частых наборов;
        candidates_per_level (dict) - {k: число кандидатов C(k)},
                          статистика для экспериментов.
    """
    n = len(transactions)
    # Набор частый, если count / n >= min_support, то есть count >= min_support * n.
    # Вычитаем 1e-9, чтобы погрешность вещественных чисел не завысила порог.
    min_count = math.ceil(min_support * n - 1e-9)

    frequent_counts = {}
    candidates_per_level = {}

    all_items = {item for transaction in transactions for item in transaction}
    candidates_per_level[1] = len(all_items)          # C1 = все товары
    level = find_frequent_1_itemsets(transactions, min_count)

    k = 2
    while level:                                     # пока L(k-1) не пусто
        frequent_counts.update(level)
        candidates = apriori_gen(level)
        if not candidates:
            break
        candidates_per_level[k] = len(candidates)
        counts = count_support(candidates, transactions, k)
        level = {c: cnt for c, cnt in counts.items() if cnt >= min_count}
        k += 1

    frequent = {itemset: cnt / n for itemset, cnt in frequent_counts.items()}
    return frequent, candidates_per_level


def sort_itemsets(frequent, order):
    """Упорядочивает список частых наборов.

    Параметры:
        frequent (dict) - {набор: поддержка}, результат функции apriori;
        order (str) - 'support' - по убыванию значения поддержки,
                      'lex' - лексикографически.

    Возвращает:
        list[tuple] - список пар (набор, поддержка) в заданном порядке.
    """
    if order == "support":
        # при равной поддержке наборы идут в лексикографическом порядке
        return sorted(frequent.items(), key=lambda pair: (-pair[1], pair[0]))
    if order == "lex":
        return sorted(frequent.items(), key=lambda pair: pair[0])
    raise ValueError("order должен быть 'support' или 'lex'")


def main():
    """Точка входа: разбирает параметры командной строки, выполняет поиск
    частых наборов и выводит результат на экран (и в файл, если он задан).

    Параметров и возвращаемого значения нет.
    """
    parser = argparse.ArgumentParser(
        description="Поиск частых наборов объектов алгоритмом Apriori")
    parser.add_argument("dataset", help="CSV-файл с транзакциями")
    parser.add_argument("min_support", type=float,
                        help="порог поддержки, доля от 0 до 1 (например, 0.05)")
    parser.add_argument("--order", choices=["support", "lex"], default="support",
                        help="упорядочивание: support - по убыванию поддержки, "
                             "lex - лексикографическое")
    parser.add_argument("--output", help="CSV-файл для сохранения результата")
    args = parser.parse_args()

    if not 0 < args.min_support <= 1:
        parser.error("порог поддержки должен быть в пределах (0; 1]")

    transactions = load_transactions(args.dataset)
    frequent, _ = apriori(transactions, args.min_support)
    result = sort_itemsets(frequent, args.order)

    print(f"Транзакций: {len(transactions)}, порог поддержки: {args.min_support}")
    print(f"Найдено частых наборов: {len(result)}")
    print()
    print(f"{'Поддержка':<11}Набор")
    for itemset, support in result:
        print(f"{support:<11.4f}{{{', '.join(itemset)}}}")

    if args.output:
        # utf-8-sig, чтобы файл корректно открывался в Excel
        with open(args.output, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["набор", "длина", "поддержка"])
            for itemset, support in result:
                writer.writerow([", ".join(itemset), len(itemset), f"{support:.4f}"])
        print(f"\nРезультат сохранён в {args.output}")


if __name__ == "__main__":
    main()

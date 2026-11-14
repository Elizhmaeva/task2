"""Поиск ассоциативных правил.

Программа является доработкой программы поиска частых наборов (apriori.py):
сначала алгоритмом Apriori находятся частые наборы, затем из них строятся
ассоциативные правила вида «антецедент → консеквент».

Запуск из командной строки:

    python association_rules.py baskets.csv 0.001 0.7 --order support
    python association_rules.py baskets.csv 0.01 0.3 --order lex --output rules.csv

Параметры программы:
    набор данных        - CSV-файл, одна строка = одна транзакция;
    порог поддержки     - доля от 0 до 1 (0.001 = 0,1 %);
    порог достоверности - доля от 0 до 1 (0.7 = 70 %);
    --order             - способ упорядочивания списка правил:
                          support - по убыванию значения поддержки,
                          lex     - лексикографический;
    --max-items         - наибольшее суммарное число объектов в антецеденте
                          и консеквенте (по умолчанию 7);
    --output            - CSV-файл для сохранения списка правил.
"""

import argparse
import csv
import sys
from itertools import combinations

from apriori import apriori, load_transactions

# Допуск при сравнении достоверности с порогом. Поддержка хранится в виде
# вещественных чисел, поэтому, например, 0.0012 / 0.0016 может получиться
# равным 0.7499999999999999 вместо 0.75.
EPSILON = 1e-9


def generate_rules(frequent, min_confidence, max_items=None):
    """Строит ассоциативные правила из частых наборов.

    Каждый частый набор I из двух и более объектов разбивается всеми
    способами на антецедент S и консеквент I \\ S. Для каждого разбиения
    вычисляется достоверность conf(S → I \\ S) = sup(I) / sup(S); правило
    сохраняется, если достоверность не ниже порога.

    Параметры:
        frequent (dict) - {набор (кортеж объектов): поддержка}, результат
                          функции apriori;
        min_confidence (float) - порог достоверности, доля от 0 до 1;
        max_items (int или None) - наибольшее суммарное число объектов
                          в антецеденте и консеквенте; None - без ограничения.

    Возвращает:
        list[dict] - список правил; каждое правило - словарь с ключами:
            antecedent - антецедент (кортеж объектов);
            consequent - консеквент (кортеж объектов);
            support    - поддержка правила, sup(антецедент ∪ консеквент);
            confidence - достоверность правила;
            lift       - лифт: достоверность / sup(консеквент).
    """
    rules = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue                      # из одного объекта правило не построить
        if max_items is not None and len(itemset) > max_items:
            continue
        # перебираем все непустые собственные подмножества набора
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                # По принципу Априори любое подмножество частого набора тоже
                # частое, поэтому его поддержка уже есть в словаре frequent.
                confidence = support / frequent[antecedent]
                if confidence >= min_confidence - EPSILON:
                    consequent = tuple(item for item in itemset
                                       if item not in antecedent)
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": support,
                        "confidence": confidence,
                        "lift": confidence / frequent[consequent],
                    })
    return rules


def sort_rules(rules, order):
    """Упорядочивает список правил.

    Параметры:
        rules (list[dict]) - список правил, результат функции generate_rules;
        order (str) - 'support' - по убыванию значения поддержки (при равной
                      поддержке - по убыванию достоверности),
                      'lex' - лексикографически по антецеденту и консеквенту.

    Возвращает:
        list[dict] - тот же список правил в заданном порядке.
    """
    if order == "support":
        return sorted(rules, key=lambda rule: (-rule["support"], -rule["confidence"],
                                               rule["antecedent"], rule["consequent"]))
    if order == "lex":
        return sorted(rules, key=lambda rule: (rule["antecedent"], rule["consequent"]))
    raise ValueError("order должен быть 'support' или 'lex'")


def format_rule(rule):
    """Представляет правило в удобочитаемом виде.

    Параметры:
        rule (dict) - правило (см. generate_rules).

    Возвращает:
        str - строка вида «{антецедент} → {консеквент}».
    """
    antecedent = ", ".join(rule["antecedent"])
    consequent = ", ".join(rule["consequent"])
    return f"{{{antecedent}}} → {{{consequent}}}"


def save_rules(rules, path):
    """Сохраняет список правил в CSV-файл.

    Параметры:
        rules (list[dict]) - список правил;
        path (str) - имя создаваемого файла.

    Возвращаемого значения нет.
    """
    # utf-8-sig, чтобы файл корректно открывался в Excel
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["антецедент", "консеквент", "объектов",
                         "поддержка", "достоверность", "лифт"])
        for rule in rules:
            writer.writerow([
                ", ".join(rule["antecedent"]),
                ", ".join(rule["consequent"]),
                len(rule["antecedent"]) + len(rule["consequent"]),
                f"{rule['support']:.4f}",
                f"{rule['confidence']:.4f}",
                f"{rule['lift']:.2f}",
            ])


def main():
    """Точка входа: разбирает параметры командной строки, находит частые
    наборы и ассоциативные правила, выводит список правил на экран
    (и в файл, если он задан).

    Параметров и возвращаемого значения нет.
    """
    parser = argparse.ArgumentParser(description="Поиск ассоциативных правил")
    parser.add_argument("dataset", help="CSV-файл с транзакциями")
    parser.add_argument("min_support", type=float,
                        help="порог поддержки, доля от 0 до 1 (например, 0.001)")
    parser.add_argument("min_confidence", type=float,
                        help="порог достоверности, доля от 0 до 1 (например, 0.7)")
    parser.add_argument("--order", choices=["support", "lex"], default="support",
                        help="упорядочивание: support - по убыванию поддержки, "
                             "lex - лексикографическое")
    parser.add_argument("--max-items", type=int, default=7,
                        help="наибольшее суммарное число объектов в правиле "
                             "(по умолчанию 7)")
    parser.add_argument("--output", help="CSV-файл для сохранения списка правил")
    args = parser.parse_args()

    if not 0 < args.min_support <= 1:
        parser.error("порог поддержки должен быть в пределах (0; 1]")
    if not 0 < args.min_confidence <= 1:
        parser.error("порог достоверности должен быть в пределах (0; 1]")

    # если консоль не умеет выводить символ «→», он заменится на «?»
    sys.stdout.reconfigure(errors="replace")

    transactions = load_transactions(args.dataset)
    frequent, _ = apriori(transactions, args.min_support)
    rules = generate_rules(frequent, args.min_confidence, args.max_items)
    rules = sort_rules(rules, args.order)

    print(f"Транзакций: {len(transactions)}, порог поддержки: {args.min_support}, "
          f"порог достоверности: {args.min_confidence}")
    print(f"Частых наборов: {len(frequent)}, найдено правил: {len(rules)}")
    print()
    print(f"{'Поддержка':<11}{'Достоверность':<15}{'Лифт':<8}Правило")
    for rule in rules:
        print(f"{rule['support']:<11.4f}{rule['confidence']:<15.4f}"
              f"{rule['lift']:<8.2f}{format_rule(rule)}")

    if args.output:
        save_rules(rules, args.output)
        print(f"\nСписок правил сохранён в {args.output}")


if __name__ == "__main__":
    main()

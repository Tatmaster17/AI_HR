def generate_report(score: float, matched: list, missing: list, strong_points: list, gaps: list) -> str:
    report = f"Процент соответствия: {score}%\n"
    report += "\nСильные стороны:\n" + "\n".join([f"- {p}" for p in strong_points])
    report += "\nПробелы:\n" + "\n".join([f"- {g}" for g in gaps])
    report += "\nПодтверждено:\n" + "\n".join([f"- {m}" for m in matched])
    report += "\nОтсутствует:\n" + "\n".join([f"- {m}" for m in missing])
    recommendation = "На следующий этап" if score > 70 else "Отказ" if score < 50 else "Требуется уточнение"
    report += f"\nРекомендация: {recommendation}"
    return report
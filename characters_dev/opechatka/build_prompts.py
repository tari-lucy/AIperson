"""Собирает Characters/Opechatka/personalities.yml из parts/.

Загрузчик NetTyan отдаёт промпт как есть и ничего не подставляет, поэтому общую
часть (core) склеиваем с зоной заранее. Запуск: python build_prompts.py
Зависимостей нет, только стандартная библиотека.
"""

from pathlib import Path

ROOT = Path(__file__).parent
PARTS = ROOT / "parts"
REPO = ROOT.parents[1]  # корень репозитория: characters_dev/opechatka -> корень
OUT = REPO / "resources" / "Characters" / "Opechatka" / "personalities.yml"

# ключ в yml -> файл зоны. streamer_abstract читает ChatAgent по умолчанию,
# поэтому он указывает на безопасный режим.
PROMPTS = {
    "reporter_strangers": "zone_strangers.txt",
    "reporter_companions": "zone_companions.txt",
    "streamer_abstract": "zone_strangers.txt",
}
# общие части, которые идут после зоны в обоих режимах
COMMON_AFTER = ["situations.txt"]

HEADER = """\
# ФАЙЛ СОБРАН СКРИПТОМ build_prompts.py. Правьте parts/*.txt, а не этот файл.
#
# reporter_strangers  - "Незнакомые места": чужие серверы, живые незнакомые игроки (безопасный режим)
# reporter_companions - "Попутчики": свой сервер, знакомые участники и другие ИИ
# streamer_abstract   - ключ по умолчанию (его читает ChatAgent), равен "Незнакомым местам"

"""


def block(key: str, text: str) -> str:
    body = "\n".join(("  " + line) if line else "" for line in text.rstrip().splitlines())
    return f"{key}: |\n{body}\n"


def compose(key: str) -> str:
    """Итоговый текст промпта для ключа (core + зона + общие части)."""
    core = (PARTS / "core.txt").read_text(encoding="utf-8").rstrip()
    zone = (PARTS / PROMPTS[key]).read_text(encoding="utf-8").rstrip()
    tail = [(PARTS / name).read_text(encoding="utf-8").rstrip() for name in COMMON_AFTER]
    return "\n\n".join([core, zone, *tail])


def main() -> None:
    chunks = [HEADER]
    for key in PROMPTS:
        chunks.append(block(key, compose(key)))
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

"""Правила, которые надёжнее держать в коде, а не в промпте.

Правило: инструменты нужны, только когда есть что делать. Модель плохо выполняет «если
никого нет, в чат не пиши», поэтому на такие ходы инструмент просто не передаётся.
- chat: есть кому читать (кто-то рядом или игрок только что написал);
- действия (agent_action): то же самое, плюс повод вроде урона или смерти;
- иначе инструментов нет вообще, остаётся только голос.
Нельзя убирать один chat и оставлять действия: модель тогда использует @sign как «выход» и
пишет туда мусор (проверено на тесте). Бонус: нет вызова инструмента, значит нет второго
раунда «Tools executed, дай голос», то есть вдвое меньше запросов и голос приходит сразу.

Файл без зависимостей. Его использует test_scenarios.py. Для NetTyan подключение
описано в README (раздел «Правило чата в коде»), в код NetTyan оно не внесено.
"""

import re

CHAT_TOOL = "chat"

# Добавляется в хвост запроса, когда инструменты не передаются. Без неё модель пишет команды
# (@stop, @sign) текстом, потому что в системном промпте про действия сказано.
VOICE_ONLY_NOTE = "(No tools this turn. Reply with one short voice report only, ending with one *emotion* tag. No commands, no labels.)"


def nobody_near(players_text: str) -> bool:
    """Для теста: текст блока Players из запроса. Пустой или '(nobody nearby)' = никого."""
    t = (players_text or "").strip().lower()
    return (not t) or t.startswith("(nobody")


def nearby_count_is_empty(nearby_players: dict) -> bool:
    """Для NetTyan: get_nearby_players(ctx_swarm) возвращает словарь, в который входит и сама
    героиня (в minecraft_tools.py комментарий '# self'), поэтому пусто = 0 или 1 запись."""
    return len(nearby_players or {}) <= 1


def has_player_chat(events: list) -> bool:
    """Был ли в последних событиях чат от игрока (строка 'CHAT [время] ник: текст' с ником)."""
    for e in events or []:
        m = re.match(r"^\s*[A-Z_ ]*CHAT\s*\[[^\]]*\]\s*([^:]+):", str(e))
        if m and not m.group(1).strip().startswith("("):
            return True
    return False


def chat_allowed(players_text: str, events: list) -> bool:
    """Чат разрешён, если кто-то рядом ИЛИ игрок только что написал."""
    return (not nobody_near(players_text)) or has_player_chat(events)


TRIGGER_TYPES = ("MC_DAMAGED", "DEATH", "KILL")


def has_trigger(events: list) -> bool:
    """Повод для действия без людей рядом: урон, смерть, убийство."""
    return any(str(e).lstrip().startswith(TRIGGER_TYPES) for e in events or [])


def select_tools(tools: list, players_text: str, events: list) -> list:
    """Список инструментов для запроса: все, только действия или пусто (голос)."""
    if chat_allowed(players_text, events):
        return list(tools)
    if has_trigger(events):
        return [t for t in tools if t.get("function", {}).get("name") != CHAT_TOOL]
    return []


if __name__ == "__main__":
    assert nobody_near("(nobody nearby)") and not nobody_near("Mila: 6m away.")
    assert has_player_chat(["MINECRAFT CHAT [12:00:03] Vasya: привет"])
    assert not has_player_chat(["SYSTEM [12:07:00] (server): [Nothing is happening.]"])
    assert not has_player_chat(["PHOTO [12:01:10] (scene): табличка"])
    assert chat_allowed("(nobody nearby)", ["MINECRAFT CHAT [12:00:03] Vasya: привет"])
    assert not chat_allowed("(nobody nearby)", ["PHOTO [12:01:10] (scene): x"])
    tl = [{"function": {"name": "chat"}}, {"function": {"name": "agent_action"}}]
    assert len(select_tools(tl, "(nobody nearby)", ["PHOTO [1] (scene): x"])) == 0
    assert [t["function"]["name"] for t in select_tools(tl, "(nobody nearby)", ["MC_DAMAGED [1] Opechatka: x"])] == ["agent_action"]
    assert len(select_tools(tl, "Mila: 5m away.", ["PHOTO [1] (scene): x"])) == 2
    assert nearby_count_is_empty({"Opechatka": {}}) and not nearby_count_is_empty({"Opechatka": {}, "Mila": {}})
    print("engine_rules: ok")

"""Прогон сценариев на промпте Opechatka.

Берёт промпт из parts/ (тот же код, что build_prompts.py), шлёт сценарии из
scenarios.json на OpenAI-совместимый адрес и пишет отчёт в results/.
Только стандартная библиотека.

Настройки (адрес, модель, ключ) берутся из .env рядом со скриптом (LLM_BASE_URL, LLM_MODEL,
LLM_API_KEY), переменных окружения или аргументов; приоритет: аргумент > окружение > .env.
Шаблон: .env.example.

Примеры:
  python3 test_scenarios.py --dry-run                     # показать промпт и первый запрос, ничего не слать
  python3 test_scenarios.py                               # LM Studio на localhost:1234
  python3 test_scenarios.py --base-url http://host:8000/v1 --model qwen3-14b
  python3 test_scenarios.py --only 03 05 --runs 3         # выбранные сценарии, по 3 прогона
  python3 test_scenarios.py --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key ...

Автопроверки мягкие (регулярки и простые правила): они ловят явные провалы,
но не заменяют чтение ответов глазами.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import build_prompts
import engine_rules

ROOT = Path(__file__).parent


def load_env(path: Path) -> None:
    """Минимальный разбор .env: KEY=value, кавычки, комментарии, ${VAR}.
    В файле побеждает последнее значение (как в python-dotenv). Уже заданные в окружении
    переменные не перезаписываются (приоритет у оболочки)."""
    if not path.exists():
        return
    local = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        val = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                     lambda m: local.get(m.group(1), os.environ.get(m.group(1), "")), val)
        local[key] = val
    for key, val in local.items():
        os.environ.setdefault(key, val)


load_env(ROOT.parents[1] / ".env")  # корень репозитория
load_env(ROOT / ".env")  # локальный, если есть

EMOTIONS = ["neutral", "happy", "sad", "angry", "scared", "whispering", "disgusted", "sarcastic"]

# Слова, которых героиня не говорит о мире (русские формы; английские тоже ловим).
FORBIDDEN_WORLD_WORDS = r"(minecraft|майнкрафт|\bигр[аыеуо]\b|в игре|сервер|\bмод\b|\bбот\b|\bchat\b)"

# Копия блока "Output rules" из NetTyan ChatAgent._build_system_prompt (chat_agent.py).
# Если там изменится, обновите здесь: в тесте нужны те же правила вывода.
OUTPUT_RULES = (
    "\n\n## Output rules"
    "\nYou are a STREAMER with voice and game chat."
    "\n"
    "\nVOICE (your text output): viewers hear this on stream. Players DON'T."
    "\n- Always end with SINGLE *emotion* in single asterisks: {examples} etc."
    "\n- Allowed emotions: {emotions}. No others."
    "\n- Never use double asterisks **like this**. Only single: *happy*"
    "\n- No roleplay *actions*. Only emotion tag at the end."
    "\n"
    "\nCHAT (chat tool): players read this in-game. Viewers see on screen."
    "\n- When someone writes to you in chat — ALWAYS call chat tool to reply."
    "\n- Keep under 80 chars."
    "\n"
    "\nACTIONS: if a player asks you to DO something (follow, come, attack, go) —"
    "\n  ACTUALLY DO IT with agent_action tool. Don't just talk about it."
    "\n  Never write commands as text. Always use tool calls."
)

# В тесте инструментов нет, поэтому вызовы заменяем строками-метками.
TOOL_STUB = (
    "\n\n(TEST MODE: tools are not connected. Where you would call the chat tool, write a line starting with "
    "'CHAT: '. Where you would call agent_action, write a line starting with 'ACTION: '. "
    "Everything else is your voice text and must end with one *emotion* tag.)"
)

# Инструменты как у ChatAgent (имена и смысл): chat и agent_action. speak не нужен: голос - текст ответа.
TOOLS = [
    {"type": "function", "function": {
        "name": "chat", "description": "Write a message to players in the in-game chat. Under 80 characters.",
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "agent_action",
        "description": "Move or interact in the world. Action like '@stop', '@follow NickName', '@goto X Y Z', '@sign text', '@gesture NickName Hey'.",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}}},
]
USE_TOOLS = True

REMINDER = "React to the latest events.\nText = voice *emotion*. Use game tools if needed."


def system_prompt(zone: str) -> str:
    """В режиме инструментов метки CHAT:/ACTION: не нужны."""
    key = "reporter_companions" if zone == "companions" else "reporter_strangers"
    emotions = ", ".join(EMOTIONS)
    examples = " ".join(f"*{e}*" for e in EMOTIONS[:4])
    return build_prompts.compose(key) + OUTPUT_RULES.format(emotions=emotions, examples=examples) + ("" if USE_TOOLS else TOOL_STUB)


def tail(players: str) -> str:
    return f"# Current Situation:\n\n## Players:\n{players}\n\n{REMINDER}"


def user_message(events: list, players: str, voice_only: bool = False) -> str:
    note = ("\n\n" + engine_rules.VOICE_ONLY_NOTE) if voice_only else ""
    return "--- Happening now ---\n" + "\n".join(events) + "\n\n" + tail(players) + note


RETRY_STATUSES = {429, 500, 502, 503, 504}


def call_llm(base_url, model, api_key, messages, temperature, max_tokens, timeout, retries=6, delay=0.0, tools=None):
    """Один запрос. На 429/5xx ждёт (Retry-After или 10, 20, 40... с) и повторяет."""
    global USE_TOOLS
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    use = tools if tools is not None else TOOLS
    if USE_TOOLS and use:
        payload["tools"] = use
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    if delay:
        time.sleep(delay)
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 400 and USE_TOOLS:
                print("    сервис не принял tools (400): перехожу на метки CHAT:/ACTION:", flush=True)
                USE_TOOLS = False
                return call_llm(base_url, model, api_key,
                                [{"role": "system", "content": messages[0]["content"] + TOOL_STUB}, *messages[1:]],
                                temperature, max_tokens, timeout, retries, 0.0, tools)
            if e.code not in RETRY_STATUSES or attempt == retries:
                raise
            try:
                wait = float(e.headers.get("Retry-After", ""))
            except ValueError:
                wait = 10.0 * (2 ** attempt)
            wait = min(wait, 120.0)
            print(f"    HTTP {e.code}, жду {wait:.0f} с (попытка {attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
    msg = data["choices"][0]["message"]
    lines = [(msg.get("content") or "").strip()]
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        if fn.get("name") == "chat":
            lines.append("CHAT: " + str(args.get("message", "")))
        elif fn.get("name") == "agent_action":
            lines.append("ACTION: " + str(args.get("action", "")))
        else:
            lines.append(f"ACTION: [неизвестный инструмент {fn.get('name')}]")
    return "\n".join(l for l in lines if l).strip(), time.time() - t0, msg


MAX_ROUNDS = 3  # как MAX_TOOL_ROUNDS в ChatAgent


def converse(args, messages, tools=None):
    """Повторяет логику ChatAgent._respond_with_tools: если после вызова инструмента текста (голоса) нет,
    отправляет 'Tools executed...' и просит голос ещё раз. Возвращает (общий текст, время, число раундов)."""
    work = list(messages)
    parts, total, rounds = [], 0.0, 0
    while rounds < MAX_ROUNDS:
        rounds += 1
        text, dt, msg = call_llm(args.base_url, args.model, args.api_key, work, args.temperature,
                                 args.max_tokens, args.timeout, args.retries, args.delay, tools)
        total += dt
        parts.append(text)
        voice, _, _ = split_reply(text)
        calls = msg.get("tool_calls") or []
        if not USE_TOOLS or not calls or voice.strip():
            break
        work.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for tc in calls:
            work.append({"role": "tool", "tool_call_id": tc.get("id", "call"), "content": "ok"})
        work.append({"role": "user", "content": "Tools executed. Continue or finish with voice *emotion*."})
    return "\n".join(p for p in parts if p).strip(), total, rounds


def split_reply(text: str):
    """-> (voice, chat_lines, action_lines)"""
    voice, chat, action = [], [], []
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("CHAT:"):
            chat.append(s[5:].strip())
        elif s.upper().startswith("ACTION:"):
            action.append(s[7:].strip())
        elif s:
            voice.append(s)
    return " ".join(voice), chat, action


# Нейроштампы: шаблонные обороты "машинного" текста. Мягкая проверка по регуляркам.
NEURO_PATTERNS = [
    (r"(?:\bэто|\bона|\bон|\bоно)\s+(?:уже\s+|ещё\s+|даже\s+|вовсе\s+|совсем\s+)?не\s+[^.!?;\n]{1,60}?,\s*(?:а\s+)?(?:это|она|он|оно)\b", "«это не X, это Y»"),
    (r"\bэто\s+(?:уже\s+|ещё\s+|даже\s+|вовсе\s+|совсем\s+)?не\s+[^.!?;\n]{1,60}?[.;]\s*это\b", "«Это не X. Это Y.»"),
    (r"\bэто\s+не\s+[^.!?;\n]{1,40}?,\s*а\b", "«это не X, а Y»"),
    (r"не\s+просто\s+[^.!?\n]{1,60}?,\s*(?:а|но)\b", "«не просто X, а Y»"),
    (r"стоит\s+отметить|своего\s+рода|по\s+сути|в\s+конечном\s+счёте|что-то\s+большее|в\s+мире,\s+где|вся\s+суть", "канцелярско-ИИ оборот"),
    (r"симфони|мурашк|время\s+остановилось|\bэхо\b|тихий\s+шёпот", "штамп-образ"),
    (r"и\s+знаете,?\s+что\?", "«и знаете что?»"),
]


def neuro_checks(text: str):
    voice, chat, _ = split_reply(text)
    body = voice + " " + " ".join(chat)
    return [f"нейроштамп {label}: «{m.group(0)[:50]}»"
            for pat, label in NEURO_PATTERNS for m in [re.search(pat, body, re.I)] if m]


def generic_checks(text: str):
    """Правила, общие для всех ответов. Возвращает список проблем."""
    voice, chat, action = split_reply(text)
    problems = []
    if not voice:
        problems.append("нет голосовой реплики")
    else:
        m = re.search(r"\*([A-Za-z]+)\*\s*$", voice)
        if not m:
            problems.append("в конце голоса нет тега эмоции *emotion*")
        elif m.group(1).lower() not in EMOTIONS:
            problems.append(f"недопустимая эмоция: {m.group(1)}")
        plain = re.sub(r"\*[A-Za-z]+\*", "", voice)
        words = len(plain.split())
        sentences = [s for s in re.split(r"(?<=[.!?…])\s+", plain.strip()) if s]
        if words > 45:
            problems.append(f"голос длиннее ~35 слов ({words})")
        elif len(sentences) > 5:
            problems.append(f"слишком рубленый голос: {len(sentences)} предложений")
    for c in chat:
        if len(c) > 80:
            problems.append(f"реплика в чат длиннее 80 символов ({len(c)})")
    if re.search(r"\*\*|^\s*[-•]\s|^#", text, re.M):
        problems.append("разметка (жирный, список или заголовок)")
    if re.search(FORBIDDEN_WORLD_WORDS, voice + " " + " ".join(chat), re.I):
        problems.append("слово, которого героиня не говорит о мире")
    if re.search(r"@(strike|chase)\b", " ".join(action) + text, re.I):
        problems.append("использует @strike/@chase")
    problems.extend(neuro_checks(text))
    return problems


def scenario_checks(sc: dict, text: str):
    problems = []
    for pat in sc.get("must", []):
        if not re.search(pat, text, re.I):
            problems.append(f"не найдено ожидаемое: /{pat}/")
    for pat in sc.get("must_not", []):
        m = re.search(pat, text, re.I)
        if m:
            problems.append(f"найдено запрещённое: «{m.group(0)}»")
    return problems


def _tools_label(tl):
    names = [t["function"]["name"] for t in tl]
    return "+".join(names) if names else "нет (только голос)"


def run_scenarios(args, scenarios, report):
    stats = {"ok": 0, "bad": 0, "err": 0}
    for sc in scenarios:
        report.append(f"\n### {sc['id']}: {sc['title']}  (зона: {sc['zone']})\n")
        report.append("События:\n```\n" + "\n".join(sc["events"]) + "\n```")
        for run in range(1, args.runs + 1):
            tl = engine_rules.select_tools(TOOLS, sc["players"], sc["events"])
            allow = _tools_label(tl)
            messages = [
                {"role": "system", "content": system_prompt(sc["zone"])},
                {"role": "user", "content": user_message(sc["events"], sc["players"], voice_only=not tl)},
            ]
            try:
                text, dt, rounds = converse(args, messages, tl)
            except Exception as e:  # noqa: BLE001 - показываем любую сетевую ошибку
                stats["err"] += 1
                report.append(f"\n**прогон {run}: ОШИБКА** {type(e).__name__}: {e}")
                print(f"  {sc['id']} #{run}: ERROR {e}")
                continue
            problems = generic_checks(text) + scenario_checks(sc, text)
            stats["bad" if problems else "ok"] += 1
            mark = "✗" if problems else "✓"
            print(f"  {sc['id']} #{run}: {mark} ({dt:.1f}s, раундов {rounds})" + (f" - {problems[0]}" if problems else ""))
            report.append(f"\n**прогон {run} {mark}** ({dt:.1f} c, раундов: {rounds}, инструменты: {allow})\n```\n{text}\n```")
            for p in problems:
                report.append(f"- ⚠ {p}")
    return stats


def run_series(args, series, report):
    for sr in series:
        report.append(f"\n### {sr['id']}: {sr['title']}  (зона: {sr['zone']})\n")
        messages = [{"role": "system", "content": system_prompt(sr["zone"])}]
        voices, chats = [], []
        for i, turn in enumerate(sr["turns"], 1):
            tl = engine_rules.select_tools(TOOLS, sr["players"], turn["events"])
            allow = _tools_label(tl)
            messages.append({"role": "user", "content": user_message(turn["events"], sr["players"], voice_only=not tl)})
            try:
                text, dt, rounds = converse(args, messages, tl)
            except Exception as e:  # noqa: BLE001
                report.append(f"\n**ход {i}: ОШИБКА** {type(e).__name__}: {e}")
                print(f"  {sr['id']} ход {i}: ERROR {e}")
                break
            voice, chat_lines, _ = split_reply(text)
            messages.append({"role": "assistant", "content": voice})
            voices.append(re.sub(r"\*[A-Za-z]+\*", "", voice).strip())
            chats.extend(chat_lines)
            problems = generic_checks(text)
            print(f"  {sr['id']} ход {i}: {'✗' if problems else '✓'} ({dt:.1f}s, раундов {rounds})")
            report.append("\n**ход %d** (%.1f c, раундов %d, инструменты: %s) `%s`\n```\n%s\n```" % (
                i, dt, rounds, allow, turn["events"][0][:60], text))
            for p in problems:
                report.append(f"- ⚠ {p}")
        if chats:
            q = sum(1 for c in chats if c.rstrip().endswith("?"))
            cs = [difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
                  for i, a in enumerate(chats) for b in chats[i + 1:]]
            cmax = max(cs) if cs else 0.0
            verdict = "много вопросов подряд" if q / len(chats) > 0.5 and len(chats) >= 3 else "в норме"
            report.append(f"\n**Чат:** реплик {len(chats)}, из них вопросов {q} ({verdict}); максимальное сходство реплик {cmax:.2f}.")
            print(f"  {sr['id']}: чат {len(chats)} реплик, вопросов {q}, макс. сходство {cmax:.2f}")
        pairs = [(a, b, difflib.SequenceMatcher(None, voices[a], voices[b]).ratio())
                 for a in range(len(voices)) for b in range(a + 1, len(voices))]
        if pairs:
            a, b, r = max(pairs, key=lambda x: x[2])
            verdict = "похоже на повтор" if r > 0.6 else "в норме"
            report.append(f"\n**Повторы:** максимальное сходство реплик {a + 1} и {b + 1} = {r:.2f} ({verdict}). "
                          "Сходство по буквам грубое: тематические повторы без совпадающих слов оно не ловит.")
            print(f"  {sr['id']}: макс. сходство {r:.2f} ({verdict})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.getenv("LLM_BASE_URL") or "http://localhost:1234/v1")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL") or "local-model")
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--delay", type=float, default=4.0, help="пауза перед каждым запросом, с (против 429)")
    ap.add_argument("--retries", type=int, default=6, help="повторов при 429/5xx")
    ap.add_argument("--runs", type=int, default=1, help="прогонов на сценарий")
    ap.add_argument("--only", nargs="*", help="префиксы id сценариев, например 03 05 S1")
    ap.add_argument("--no-tools", action="store_true", help="не передавать tools, использовать метки CHAT:/ACTION:")
    ap.add_argument("--no-series", action="store_true", help="не гонять серию на повторы")
    ap.add_argument("--dry-run", action="store_true", help="показать промпт и запрос, ничего не отправлять")
    args = ap.parse_args()
    global USE_TOOLS
    USE_TOOLS = not args.no_tools

    data = json.loads((ROOT / "scenarios.json").read_text(encoding="utf-8"))
    pick = lambda items: [x for x in items if not args.only or any(x["id"].startswith(o) for o in args.only)]
    scenarios, series = pick(data["scenarios"]), ([] if args.no_series else pick(data.get("series", [])))

    if args.dry_run:
        sc = scenarios[0] if scenarios else data["scenarios"][0]
        print("=== SYSTEM (%s) ===\n%s\n\n=== USER ===\n%s" % (
            sc["zone"], system_prompt(sc["zone"]), user_message(sc["events"], sc["players"])))
        return

    report = [f"# Прогон Opechatka\n\nМодель: `{args.model}`, адрес: `{args.base_url}`, "
              f"температура {args.temperature}, прогонов на сценарий {args.runs}.\n"
              f"Время: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    if args.model == "local-model" or not args.api_key:
        print("Внимание: модель или ключ не заданы (.env / аргументы). Для локальной LM Studio ключ не нужен.")
    print(f"Модель {args.model} @ {args.base_url}")
    stats = run_scenarios(args, scenarios, report)
    if series:
        run_series(args, series, report)

    summary = f"ИТОГО: ✓ {stats['ok']}, ✗ {stats['bad']}, ошибок связи {stats['err']}"
    report.insert(1, f"\n**{summary}**\n")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{safe_model}.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n{summary}\nОтчёт: {out}")
    if stats["err"] and not (stats["ok"] or stats["bad"]):
        sys.exit(2)


if __name__ == "__main__":
    main()

"""Anonymization tool (Interface 4): personal data out of legal documents.

Three passes, cheapest first:
1. REGEX for machine-recognizable data: ЕМБГ (13 digits), phone numbers,
   emails, bank accounts, document numbers, dates of birth, vehicle plates,
   insurance policies, and web addresses.
2. LLM for people and role-dependent anonymization.
3. LLM for legal entities, institutions, addresses, and places.

"""
from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL


_LABEL_SEPARATOR = r"\s*(?:[,;:#№\-–—]\s*)?"
_DOCUMENT_NUMBER = r"[A-Z]\d{7}"


def _labelled_value_pattern(label: str, value: str) -> re.Pattern:
    """Match a sensitive value after a label, while preserving the label."""
    return re.compile(
        rf"(?<!\w)(?i:{label}){_LABEL_SEPARATOR}"
        rf"(?P<value>{value})(?!\w)"
    )


REGEX_RULES = [
    ("ЕМБГ", re.compile(r"\b\d{13}\b"), "..."),
    ("трансакциска сметка", re.compile(r"\b\d{3}[- ]?\d{10,12}[- ]?\d{2}\b"), "..."),
    ("телефон", re.compile(r"\b(?:\+389|0)7\d[\s/-]?\d{3}[\s/-]?\d{3}\b"), "..."),
    ("е-пошта", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "..."),
    (
        "веб-адреса",
        re.compile(
            r"(?<![@\w])(?:"
            r"(?:(?:https?|ftp):)?//[A-Za-z0-9.-]+\.[A-Za-z.]{2,63}"
            r"(?:[/?#][^\s<>\"'“”‘’]*)?"
            r"|www\.[A-Za-z0-9.-]+\.[A-Za-z.]{2,63}"
            r"(?:[/?#][^\s<>\"'“”‘’]*)?"
            r"|(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
            r"[A-Za-z]{2,63}"
            r"(?:/[^\s<>\"'“”‘’]*)?"
            r")(?<![.,;:!?)}\]])",
            re.IGNORECASE,
        ),
        "...",
    ),
    (
        "број на лична карта",
        _labelled_value_pattern(
            r"(?:лична(?:та)?\s+карта(?:\s+(?:бр(?:ој)?\.?))?"
            r"|(?:бр(?:ој)?\.?\s+на\s+лична(?:та)?\s+карта))",
            _DOCUMENT_NUMBER,
        ),
        "...",
    ),
    (
        "датум на раѓање",
        _labelled_value_pattern(
            r"(?:датум(?:от)?\s+на\s+раѓање|роден[ао]?\s+на)",
            r"(?:"
            r"(?:0?[1-9]|[12]\d|3[01])\s*[./-]\s*(?:0?[1-9]|1[0-2])"
            r"\s*[./-]\s*(?:\d{4}|\d{2})"
            r"|(?:19|20)\d{2}\s*[./-]\s*(?:0?[1-9]|1[0-2])"
            r"\s*[./-]\s*(?:0?[1-9]|[12]\d|3[01])"
            r")",
        ),
        "...",
    ),
    (
        "број на пасош",
        _labelled_value_pattern(
            r"(?:пасош(?:от)?(?:\s+(?:бр(?:ој)?\.?))?"
            r"|(?:бр(?:ој)?\.?\s+на\s+пасош))",
            _DOCUMENT_NUMBER,
        ),
        "...",
    ),
    (
        "број на возачка дозвола",
        _labelled_value_pattern(
            r"(?:возачка(?:та)?(?:\s+дозвола)?"
            r"(?:\s+(?:бр(?:ој)?\.?))?"
            r"|(?:бр(?:ој)?\.?\s+на\s+возачка"
            r"(?:\s+дозвола)?))",
            _DOCUMENT_NUMBER,
        ),
        "...",
    ),
    (
        "регистарска ознака",
        _labelled_value_pattern(
            r"(?:регистарска\s+ознака"
            r"|регистарск[аи]\s+табличк[аи]|регистрација)"
            r"(?:\s+(?:бр(?:ој)?\.?))?",
            r"[A-Z]{2}[\s\-\u2013\u2014./]+\d{3,4}[\s\-\u2013\u2014./]+[A-Z]{2}",
        ),
        "...",
    ),
    (
        "број на полиса за осигурување",
        _labelled_value_pattern(
            r"(?:полиса(?:та)?(?:\s+(?:на|за)\s+осигурување)?"
            r"(?:\s+(?:бр(?:ој)?\.?))?"
            r"|(?:бр(?:ој)?\.?\s+на\s+полиса(?:та)?"
            r"(?:\s+(?:на|за)\s+осигурување)?))",
            r"POL[\s\-\u2013\u2014/]\d{4}[\s\-\u2013\u2014/]\d{6}",
        ),
        "...",
    ),
]


def _apply_regex_rules(text: str) -> tuple[str, list[dict]]:
    """Apply deterministic rules and return the redacted text and audit rows."""
    result = text
    replacements = []

    for label, pattern, token in REGEX_RULES:
        matched_values = []
        seen = set()
        for match in pattern.finditer(result):
            value = match.groupdict().get("value") or match.group(0)
            if value not in seen:
                seen.add(value)
                matched_values.append(value)

        for value in matched_values:
            replacements.append({"original": value, "replacement": token,
                                 "method": f"правило ({label})"})
            result = result.replace(value, token)

    return result, replacements


_LLM_OUTPUT_INSTRUCTIONS = """

Општи правила:
- Менувај само податоци опфатени со овој чекор. Не менувај факти, јавни податоци, правна терминологија, износи, законски членови,
  броеви на предмети или веќе внесените ознаки „...“.
- Биди доследен низ целиот текст: истиот субјект добива иста ознака; броевите се за различни субјекти, не за повторени појавувања.
- Врати само JSON со точно две полињ: "reasoning" (кратко, на македонски) и "replacements" (листа од {"original":"...","replacement":"..."}).
  Секој "original" мора буквално да постои во текстот, а листата да ги содржи сите потребни замени. Ако нема замени, врати празна листа. Не го препишувај документот."""


PEOPLE_LLM_PROMPT = """Анонимизирај Македонски судски одлуки САМО според следните правила: 
При анонимизација на судски одлуки со име и презиме на странките, односно називот на правното лице се анонимизираат следните податоци: 
- адресата на живеалиштето на странките и другите учесници во постапката; 
- датумот и местото на раѓање на странките и другите учесници во постапката; 
- името и презимето и другите лични податоци на сведокот, вештакот, толкувачот, социјалниот работник, психологот, педагогот, дефектологот, лекарот и сл. 
При анонимизација на судските одлуки НЕ се анонимизира името и презимето на судиите, јавните обвинители, државните правобранители, законските застапници, бранителите и полномошниците на странките. 
Анонимизацијата на личните податоци од ова упатство се врши со замена или испуштање на податоци во судските одлуки на следниов начин: 
1. Анонимизација на име и презиме 
а). Името и презимето се анонимизира со замена со иницијали од почетните големи печатни букви по кои се додава точка; Пример: Петар Петровски се заменува со: П. П. 
б). Ако во иста судска одлука има повеќе лица со исти иницијали анонимизацијата се врши на следниов начин: 
- кај првото лице името се заменува со иницијал со точка, а презимето се заменува со иницијал со точка и се додава реден број 1; 
- кај второто , третото или другите лица името се заменува со иницијал со точка, а презимето се заменува со иницијал со точка и се додава следниот реден број. Пример: Крсте Крстевски, Кочо Кочовски и Кире Костовски се заменуваат со К. К. 1, К. К. 2 и К. К. 3 
2. Анонимизација на називот на правните лица 
- Називот на правното лице (трговско друштво ) се анонимизира така што називот на трговското друштво се заменува со голема почетна печатна буква по која се става точка и се додаваат големи печатни букви со точка кои го означуваат обликот на трговското друштво. Ако во текстот на судската одлука е наведено јавно претпријатие или друг вид на правно лице не се анонимизираат податоците за правниот облик на тоа прав но лице. Пример: „Скопје” ДООЕЛ се заменува со С. ДООЕЛ, Јавно претпријатие „Водовод и канализација” се за менува со Јавно претпријатие В.К. 
""" + _LLM_OUTPUT_INSTRUCTIONS

ENTITIES_ADDRESSES_LLM_PROMPT = """Анонимизирај Македонски судски одлуки САМО според следните правила: 
Анонимизацијата на личните податоци од ова упатство се врши со замена или испуштање на податоци во судските одлуки на следниов начин: 
1. Анонимизација на називите на органи на државна управа, управни организации, установи и институции како и единиците на локалната самоуправа 
-Називот на органи на државна управа, управни организации, установи и институции како и единиците на локалната самоуправа се анонимизира така што зборовите од кои што се состои називот се заменуваат само со еден збор кој го означува правниот облик на субјектот. Ако називот содржи зборови со наводници, тие зборови се заменуваат со иницијалите на почетните големи печатни букви на кои им се додава точка, а наводниците се изоставуваат. Пример: Министерство за правда се заменува со Министерство, Државен завод за статистика се заменува со Завод, Основно училиште „Гоце Делчев” се заменува со Училиште Г.Д. 
2. Особено внимание се посветува на анонимизацијата на адресата и местото на раѓање 
- Имињата на државите, градовите и местата на раѓање се анонимизираат на начин што името на градот се заменува со почетните големи печатни букви по кои се додава точка. Останатиот дел од адресата на живеење или живеалиштето како што е улицата и бројот СЕ ОТСТРАНУВААТ ВО СЕКОЈ СЛУЧАЈ (Пример: ул. „Благој Ѓорев“ 104Б-3 Велес се заменува со В.). Во случај во иста судска одлука да има исти почетни букви на држави, градови или места, на секои нови почетни букви се до дава реден број со почеток од 1. Пример: ул. „Димитрие Чупоски“ бр. 9, Скопје, Република Македонија се заменува со С.Р.М; Ул. „522” бр.1, Демир Хисар, Република Македони ја се заменува со Д.Х.Р.М. 
""" + _LLM_OUTPUT_INSTRUCTIONS



def _apply_llm_rules(llm, text: str, replacements: list[dict],
                     prompt: str, method: str) -> tuple[str, str]:
    """Apply one focused LLM ruleset to the current version of the text."""
    response = llm.invoke([("system", prompt), ("user", text)])

    try:
        parsed = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return text, "Грешка при читање на АИ-одговорот; овој чекор не е применет."

    for item in parsed.get("replacements", []):
        if not isinstance(item, dict):
            continue
        original = str(item.get("original", ""))
        if original and original in text:
            replacement = str(item.get("replacement", "..."))
            replacements.append({"original": original,
                                 "replacement": replacement,
                                 "method": method})
            text = text.replace(original, replacement)

    return text, str(parsed.get("reasoning", ""))


def anonymize(text: str) -> dict:
    # pass 1: regex
    result, replacements = _apply_regex_rules(text)

    # passes 2 and 3: focused LLM rulesets
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    llm_passes = [
        ("Лица", PEOPLE_LLM_PROMPT, "ВИ (лица)"),
        ("Правни лица, органи и адреси",
         ENTITIES_ADDRESSES_LLM_PROMPT, "ВИ (правно лице/орган/адреса)"),
    ]
    reasoning_parts = []
    for label, prompt, method in llm_passes:
        result, reasoning = _apply_llm_rules(
            llm, result, replacements, prompt, method
        )
        if reasoning:
            reasoning_parts.append(f"{label}: {reasoning}")

    return {"anonymized": result, "replacements": replacements,
            "reasoning": "\n".join(reasoning_parts)}

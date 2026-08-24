"""Anonymization tool (Interface 4): personal data out of legal documents.

Two passes, cheapest first:
1. REGEX for machine-recognizable data: ЕМБГ (13 digits), phone numbers,
   emails, bank accounts. Deterministic, free, never misses the pattern.
2. LLM for what regex can't see: names of people, addresses, company names —
   returns a JSON list of replacements that we apply with plain str.replace.

"""
from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL


REGEX_RULES = [
    ("ЕМБГ", re.compile(r"\b\d{13}\b"), "«ЕМБГ»"),
    ("трансакциска сметка", re.compile(r"\b\d{3}[- ]?\d{10,12}[- ]?\d{2}\b"), "«СМЕТКА»"),
    ("телефон", re.compile(r"\b(?:\+389|0)7\d[\s/-]?\d{3}[\s/-]?\d{3}\b"), "«ТЕЛЕФОН»"),
    ("е-пошта", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "«Е-ПОШТА»"),
]

LLM_PROMPT = """Ти си алатка за анонимизација на македонски правни документи.
Анонимизирај исклучиво според следните правила. Не предлагај замена за текст што не е
опфатена со нив.

1. Физички лица
- Името и презимето на странка замени ги со иницијали со точки и празно место, на пример:
  „Петар Петровски“ -> „П. П.“.
- Ако различни лица имаат исти иницијали, разликувај ги со реден број:
  „К. К. 1“, „К. К. 2“ итн.
- Анонимизирај ги името, презимето и другите лични податоци на сведоци, вештаци,
  толкувачи, социјални работници, психолози, педагози, дефектолози, лекари и слични
  учесници.
- НЕ ги анонимизирај имињата на судии, јавни обвинители, државни правобранители,
  законски застапници, бранители и полномошници. Улогата утврди ја од контекстот.

2. Правни лица
- Називот на правно лице замени го со иницијал или иницијали, но секогаш задржи го
  правниот облик.
- Примери: „Скопје“ ДООЕЛ -> С. ДООЕЛ; Јавно претпријатие „Водовод и канализација“
  -> Јавно претпријатие В. К.

3. Државни органи, установи и институции
- Називот замени го со зборот што го означува видот или правниот облик на субјектот,
  на пример: Министерство за правда -> Министерство.
- Ако субјектот има посебен назив во наводници, замени го називот со иницијали без
  наводници, на пример: Основно училиште „Гоце Делчев“ -> Училиште Г. Д.

4. Адреси и места
- Изостави ги улицата, живеалиштето, бројот и другите прецизни адресни податоци.
- Држава, град и место на раѓање замени ги со иницијали со точки.
- Ако различни места имаат исти иницијали, додај реден број за разликување.

5. Податоци што СЕКОГАШ се заменуваат со „...“
- датум на раѓање;
- ЕМБГ;
- број на лична карта, пасош, возачка дозвола или друг личен документ;
- број на полиса за осигурување;
- регистарска ознака на возило;
- e-mail адреса;
- веб-адреса.

6. Општи правила
- Применувај ги замените доследно низ целиот документ: истиот субјект мора секогаш да
  ја добие истата ознака. Редните броеви се однесуваат на различни субјекти, не на
  повторните појавувања на истиот субјект.
- Не анонимизирај податоци што според овие правила треба да останат јавни.
- Не менувај факти, правна терминологија, износи, членови од закони, броеви на предмети
  или друг текст што не содржи податок предвиден за анонимизација.
- Некои машински препознатливи податоци можеби веќе се заменети со ознаки во текстот;
  не ги менувај тие ознаки повторно.

Не го препишувај документот и не додавај објаснувања во неговиот текст. Врати JSON со
точно две полиња:
- "reasoning": кратко објаснување за замените, на македонски;
- "replacements": листа од објекти {"original": "...", "replacement": "..."}.

Секоја вредност "original" мора буквално да постои во текстот, а "replacement" мора да
биде точната замена според правилата погоре. Листата мора да биде доволна за со нејзина
примена да се добие целиот анонимизиран текст, без други промени. Ако нема ништо за
замена, врати празна листа replacements."""


def anonymize(text: str) -> dict:
    replacements = []

    # pass 1: regex
    result = text
    for label, pattern, token in REGEX_RULES:
        for match in set(pattern.findall(result)):
            replacements.append({"original": match, "replacement": token,
                                 "method": f"правило ({label})"})
            result = result.replace(match, token)

    # pass 2: LLM for names/addresses/companies
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    response = llm.invoke([("system", LLM_PROMPT), ("user", result)])

    reasoning = ""
    try:
        parsed = json.loads(response.content)
        reasoning = parsed.get("reasoning", "")
        for r in parsed.get("replacements", []):
            original = str(r.get("original", ""))
            if original and original in result:
                replacement = str(r.get("replacement", "«XXX»"))
                replacements.append({"original": original,
                                     "replacement": replacement,
                                     "method": "АИ (име/адреса/фирма)"})
                result = result.replace(original, replacement)
    except json.JSONDecodeError:
        reasoning = "Грешка при читање на АИ-одговорот; применети се само правилата."

    return {"anonymized": result, "replacements": replacements,
            "reasoning": reasoning}

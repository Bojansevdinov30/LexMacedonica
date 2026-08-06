"""Quick RAG quality check: run typical questions through the full chain.

Run:  python -m app.rag.eval_questions
Prints the retrieval confidence, computed probability, cited cases and answer
for each question — the fastest way to see whether tuning is needed.
"""

from app.rag.chains import answer_question
from app.rag.retriever import retrieve

QUESTIONS = [
    # typical labor-dispute situations (should answer with cases + %):
    "Работодавачот не ми ги исплати последните три плати. Што можам да направам и какви се шансите да ги добијам преку суд?",
    "Добив отказ без никакво образложение по 8 години работа во фирмата. Дали отказот е незаконски?",
    "Не ми се платени прекувремените часови веќе една година. Дали вреди да тужам?",
    "Фирмата не ми уплаќала придонеси за пензиско осигурување. Како да го докажам стажот?",
    # off-topic (MUST return the honest no-answer):
    "Која е најдобрата рецепта за ајвар?",
]


def main() -> None:
    for q in QUESTIONS:
        print("=" * 80)
        print("ПРАШАЊЕ:", q)
        conf = retrieve(q)
        print(f"[confidence] max cosine similarity = {conf.max_similarity:.3f}")
        res = answer_question(q)
        print(f"[probability] {res['probability']}")
        for c in res["cases"]:
            print(f"[case] {c['case_number']} | {c['court']} | {c['date']} | {c['outcome']}")
        print("[answer]", res["answer"][:600])
        print()


if __name__ == "__main__":
    main()

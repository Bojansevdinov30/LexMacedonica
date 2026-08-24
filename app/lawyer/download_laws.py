"""Download official consolidated law texts (PDFs) into data/laws/."""
import app.config  # noqa: F401  (TLS fix must load before requests)

import requests

from app.config import DATA_DIR

LAWS_DIR = DATA_DIR / "laws"

LAWS = {
    # filename (also used as the law's display name) -> official PDF URL
    "Закон за работните односи (пречистен текст 2023)":
        "https://cms.mtsp.gov.mk/content/pdf/2023/trud/ZRO,%20precisten%20tekst%202023.pdf",
    "Закон за облигационите односи":
        "https://gazibaba.gov.mk/wp-content/uploads/2023/04/16.-zakon-za-obligacionite-odnosi.pdf",
}


def main() -> None:
    LAWS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in LAWS.items():
        dest = LAWS_DIR / f"{name}.pdf"
        if dest.exists():
            print(f"already have: {name}")
            continue
        print(f"downloading: {name}")
        try:
            resp = requests.get(url, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                print(f"  NOT a PDF, skipped ({url})")
                continue
            dest.write_bytes(resp.content)
            print(f"  saved {len(resp.content) // 1024} KB")
        except Exception as e:  # one bad URL must not kill the rest
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()

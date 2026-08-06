"""Scraper for the Macedonian court decisions portal (sud.mk / vsrm.mk)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.config import RAW_PDF_DIR, RAW_HTML_DIR, DATA_DIR

ODLUKI_URL = "http://www.sud.mk/wps/portal/central/sud/odluki"

LEGAL_AREA = {"civil": "2", "criminal": "1", "administrative": "3"}
TYPE_OF_CASE = {
    "labor": "98",          # Работни спорови
    "obligations": "177",   # Облигациони спорови
    "civil_disputes": "90", # Граѓански спорови
    "small_claims": "93",   # Граѓански спорови од мала вредност
}

# GUIDs from the court <select> on the site.
COURTS = {
    "skopje_gragjanski": "5BD07829-2A09-430E-888D-F97D68F14F2D",  # Основен граѓански суд Скопје
    "bitola": "E17AF6B3-9F38-413A-8FEF-BDCED145ECF8",
    "kumanovo": "3BEDC89D-C521-4444-884A-8994720BD919",
    "tetovo": "676A87AC-2722-453A-849E-70C11058564D",
    "prilep": "8378CCFB-619D-4AD4-9484-7EF1660B65A2",
    "ohrid": "B117CD6C-E511-46B0-9E85-18A5885BF915",
    "strumica": "844D88BD-5604-4C87-BEBE-32F3CD7D67BA",
    "shtip": "CA246ADC-72E3-40FA-B6A0-98CCFAB9EE77",
    "veles": "1DFB94C9-5E83-4113-A610-D327985EE3C2",
    "gostivar": "E79EAAF1-14E5-40DC-9191-BC7D84DFBB8D",
    "struga": "DF6AF6B2-49B6-4266-9389-35790A7FB403",
    "kavadarci": "2ED34234-7408-45D0-8755-BAD5CAD535C8",
    "kochani": "D5FC2B36-FFD1-48C5-B59C-3C3FC9F8B63C",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "mk-MK,mk;q=0.9,en;q=0.8",
}

REQUEST_DELAY_SECONDS = 2.5  # politeness


@dataclass
class CaseResult:
    case_id: str
    case_number: str
    court: str
    date: str
    preview: str
    download_href: str
    # from the collapsed «Повеќе податоци» section of each result box;
    judge: str = ""            # Судија
    legal_area: str = ""       # Правна област
    case_type: str = ""        # Вид предмет
    case_subtype: str = ""     # Подвид предмет
    foundation_type: str = ""  # Вид основ
    foundation: str = ""       # Основ


class SudMkScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.page_url: str | None = None       # URL the search page ended up on
        self.form_action: str | None = None    # session-encoded POST target
        self.base_fields: dict[str, str] = {}  # every form field with defaults

    # ---------- step 1: open the page, capture session + form ----------

    def open_search_page(self) -> None:
        resp = self.session.get(ODLUKI_URL, timeout=30)
        resp.raise_for_status()
        self._save_debug_html("00_search_page.html", resp.text)
        self._parse_form(resp.text, resp.url)

    def _parse_form(self, html: str, fallback_url: str) -> None:
        soup = BeautifulSoup(html, "html.parser")

        base = soup.find("base", href=True)
        self.page_url = base["href"] if base else fallback_url

        form = soup.find("form", id="search_form")
        if form is None:
            raise RuntimeError(
                "search_form not found — the portal layout changed or we got "
                "an error page. Check data/raw_html/00_search_page.html"
            )
        self.form_action = urljoin(self.page_url, form["action"])

        # Collect ALL fields with their default values
        fields: dict[str, str] = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            if inp.get("type") == "checkbox":
                # unchecked checkboxes are NOT submitted by browsers
                if inp.has_attr("checked"):
                    fields[name] = inp.get("value", "on")
            else:
                fields[name] = inp.get("value", "")
        for sel in form.find_all("select"):
            name = sel.get("name")
            if not name:
                continue
            selected = sel.find("option", selected=True)
            fields[name] = selected.get("value", "") if selected else ""
        for ta in form.find_all("textarea"):
            name = ta.get("name")
            if name:
                fields[name] = ta.get_text()
        self.base_fields = fields

    # ---------- step 2+4: search & paginate ----------

    def search(self, page: int = 1, **filters: str) -> tuple[list[CaseResult], int | None]:
        """POST the search form. Returns (results, total_count)."""
        if self.form_action is None:
            self.open_search_page()

        data = dict(self.base_fields)
        data.update(filters)
        data["currentPage"] = str(page)
        data["paging_in_progress"] = "1" if page > 1 else "0"
        data["advancedState"] = "1"  # advanced filters panel open

        time.sleep(REQUEST_DELAY_SECONDS)
        resp = self.session.post(self.form_action, data=data, timeout=60)
        resp.raise_for_status()
        self._save_debug_html(f"search_p{page}.html", resp.text)
        # the response carries a NEW state URL + form action — refresh both so
        # pagination and downloads keep working within the session
        try:
            self._parse_form(resp.text, resp.url)
        except RuntimeError:
            pass  # results page without the form; keep previous state
        return self._parse_results(resp.text)

    def _parse_results(self, html: str) -> tuple[list[CaseResult], int | None]:
        soup = BeautifulSoup(html, "html.parser")

        total = None
        h3 = soup.find("h3", string=re.compile(r"Број на резултати"))
        if h3:
            m = re.search(r"(\d[\d.,]*)", h3.get_text())
            if m:
                total = int(re.sub(r"[.,]", "", m.group(1)))

        results: list[CaseResult] = []
        for box in soup.select("div.smkBoxContent"):
            link = box.find("a", href=re.compile(r"caseId="))
            if link is None:
                continue
            m = re.search(r"caseId=([A-Fa-f0-9-]+)", link["href"])
            if not m:
                continue

            # label/value rows: Суд / Број на предмет / Датум на одлука
            values: dict[str, str] = {}
            for row in box.select("div.smkRow"):
                label_el = row.select_one("h5")
                value_el = row.select_one("h4")
                if label_el and value_el:
                    label = label_el.get_text(strip=True).rstrip(" :")
                    values[label] = value_el.get_text(strip=True)

            # «Повеќе податоци» — the collapsed extra-data section. It is
            # ALREADY in this same HTML (jQuery only toggles its visibility),
            # so no extra request is needed. Select by CLASS scoped to this
            # box: the portal reuses the same id= on every result box, so
            # document-wide id lookups would always hit the first box only.
            extended: dict[str, str] = {}
            for cell in box.select("div.smkContentToggle div.cell50"):
                label_el = cell.select_one("h5")
                value_el = cell.select_one("h4")
                if label_el and value_el:
                    # label text is typeset as "Судија\n   :" — collapse the
                    # whitespace before stripping the colon
                    label = re.sub(r"\s+", " ", label_el.get_text()).strip().rstrip(" :")
                    extended[label] = value_el.get_text(strip=True)

            preview_el = box.select_one("div.smkRow > p")
            results.append(CaseResult(
                case_id=m.group(1).upper(),
                case_number=values.get("Број на предмет", ""),
                court=values.get("Суд", ""),
                date=values.get("Датум на одлука", ""),
                preview=preview_el.get_text(" ", strip=True) if preview_el else "",
                download_href=link["href"],
                judge=extended.get("Судија", ""),
                legal_area=extended.get("Правна област", ""),
                case_type=extended.get("Вид предмет", ""),
                case_subtype=extended.get("Подвид предмет", ""),
                foundation_type=extended.get("Вид основ", ""),
                foundation=extended.get("Основ", ""),
            ))
        return results, total

    # ---------- step 5: download the PDF ----------

    def download_case(self, result: CaseResult, dest_dir: Path = RAW_PDF_DIR) -> Path | None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{result.case_id}.pdf"
        if dest.exists():  # resume support: skip what we already have
            return dest

        url = urljoin(self.page_url, result.download_href)
        time.sleep(REQUEST_DELAY_SECONDS)
        resp = self.session.get(url, timeout=120)
        resp.raise_for_status()

        if not resp.content.startswith(b"%PDF"):
            # site sometimes serves an HTML error page instead of the PDF
            self._save_debug_html(f"bad_download_{result.case_id}.html", resp.text)
            return None
        dest.write_bytes(resp.content)
        return dest

    # ---------- helpers ----------

    @staticmethod
    def _save_debug_html(name: str, html: str) -> None:
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_HTML_DIR / name).write_text(html, encoding="utf-8")


def append_metadata(results: list[CaseResult], path: Path = DATA_DIR / "cases_meta.jsonl") -> None:
    """Append scraped metadata as JSON-lines; the ingest step loads it into SQLite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            seen = {json.loads(line)["case_id"] for line in f if line.strip()}
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            if r.case_id not in seen:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                seen.add(r.case_id)

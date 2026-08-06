"""CLI to scrape court decisions from sud.mk.

    # count labor cases per court, download nothing:
    python -m app.scraper.run_scrape --prefix РО --limit 0

    # download up to 40 labor-dispute PDFs per court, from all listed courts:
    python -m app.scraper.run_scrape --prefix РО --limit 40

    # a single court:
    python -m app.scraper.run_scrape --prefix РО --court bitola --limit 60
"""
import argparse

from app.scraper.sudmk import SudMkScraper, COURTS, append_metadata


def scrape_court(court_key: str, prefix: str, date_from: str, date_to: str,
                 limit: int) -> int:
    """One fresh session per court (the server session is sticky)."""
    scraper = SudMkScraper()
    scraper.open_search_page()

    filters = {
        "court": COURTS[court_key],
        "casenumber": prefix,
        "dateVerifyFrom": date_from,
        "dateVerifyTo": date_to,
    }

    downloaded, page = 0, 1
    while True:
        results, total = scraper.search(page=page, **filters)
        if page == 1:
            print(f"[{court_key}] total results: {total}")
        if not results:
            break

        append_metadata(results)
        if limit == 0:
            break
        for r in results:
            path = scraper.download_case(r)
            if path:
                downloaded += 1
                print(f"  [{court_key} {downloaded}/{limit}] {r.case_number} ({r.date}) -> {path.name}")
            else:
                print(f"  [{court_key}] {r.case_number} -> download FAILED")
            if downloaded >= limit:
                return downloaded
        page += 1
    return downloaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape sud.mk court decisions")
    ap.add_argument("--prefix", default="РО",
                    help="case-number prefix, e.g. РО (labor), МАЛВП, П1 (default: РО)")
    ap.add_argument("--court", choices=COURTS.keys(), default=None,
                    help="single court (default: iterate over all listed courts)")
    ap.add_argument("--from", dest="date_from", default="", help="dd.mm.yyyy")
    ap.add_argument("--to", dest="date_to", default="", help="dd.mm.yyyy")
    ap.add_argument("--limit", type=int, default=10,
                    help="max PDFs per court (0 = only count + save metadata)")
    args = ap.parse_args()

    courts = [args.court] if args.court else list(COURTS.keys())
    grand_total = 0
    for key in courts:
        try:
            grand_total += scrape_court(key, args.prefix, args.date_from,
                                        args.date_to, args.limit)
        except Exception as e:  # one court failing must not kill the whole run
            print(f"[{key}] ERROR: {e}")
    print(f"\nDone. {grand_total} PDFs total in data/raw_pdfs/")


if __name__ == "__main__":
    main()

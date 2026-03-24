import pandas as pd
from pathlib import Path
import re
import sys
from google_play_scraper import Sort, reviews, search


LANGUAGE_OPTIONS = {
    "de": [("de", "de"), ("at", "de"), ("ch", "de")],
    "en": [("us", "en"), ("gb", "en"), ("in", "en")],
    "both": [
        ("de", "de"),
        ("at", "de"),
        ("ch", "de"),
        ("us", "en"),
        ("gb", "en"),
        ("in", "en"),
    ],
}

LANGUAGE_NAME_BY_CHOICE = {
    "de": "de",
    "en": "en",
    "both": "de+en",
}


def search_apps(query: str, n_hits: int = 30) -> list[dict]:
    if not query or not query.strip():
        return []

    query = query.strip()
    query_lower = query.lower()
    combined = []

    search_regions = [
        ("de", "de"),
        ("de", "en"),
        ("at", "de"),
        ("ch", "de"),
        ("us", "en"),
        ("gb", "en"),
        ("in", "en"),
    ]

    for country, lang in search_regions:
        try:
            combined.extend(search(query, country=country, lang=lang, n_hits=n_hits))
        except Exception:
            continue

    app_by_id = {}
    for item in combined:
        app_id = item.get("appId")
        if app_id and app_id not in app_by_id:
            app_by_id[app_id] = item

    results = list(app_by_id.values())
    query_tokens = [token for token in query_lower.split() if token]

    def is_relevant(item: dict) -> bool:
        title = item.get("title", "") or ""
        app_id = item.get("appId", "") or ""
        developer = item.get("developer", "") or ""
        text = f"{title} {app_id} {developer}".lower()

        if not query_tokens:
            return False

        return all(token in text for token in query_tokens)

    filtered_results = [item for item in results if is_relevant(item)]
    if filtered_results:
        results = filtered_results

    return sorted(
        results,
        key=lambda item: (
            query_lower not in (item.get("title", "") or "").lower()
            and query_lower not in (item.get("appId", "") or "").lower(),
            -(item.get("score") or 0),
        ),
    )


def get_review_regions(language_choice: str = "both") -> list[tuple[str, str]]:
    return LANGUAGE_OPTIONS.get(language_choice, LANGUAGE_OPTIONS["both"])


def normalize_language_choice(language_choice: str = "both") -> str:
    normalized = (language_choice or "both").strip().lower()
    return normalized if normalized in LANGUAGE_OPTIONS else "both"


def language_choice_to_name(language_choice: str = "both") -> str:
    normalized = normalize_language_choice(language_choice)
    return LANGUAGE_NAME_BY_CHOICE[normalized]


def build_output_path(
    app_title: str,
    language_choice: str = "en",
    output_dir: str | Path = "data",
) -> Path:
    safe_name = re.sub(r'[<>:"/\\|?*]+', "", app_title).strip().replace(" ", "_")
    if not safe_name:
        safe_name = "app"

    language_suffix = language_choice_to_name(language_choice)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path / f"{safe_name}_{language_suffix}_raw.csv"


def write_reviews_to_csv(
    app_title: str,
    review_data: list[dict],
    language_choice: str = "en",
    output_dir: str | Path = "data",
) -> Path:
    output_path = build_output_path(
        app_title,
        language_choice=language_choice,
        output_dir=output_dir,
    )
    pd.DataFrame(review_data).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def parse_selected_index(raw_selection: str, max_items: int) -> int | None:
    if not raw_selection.strip().isdigit():
        return None

    index = int(raw_selection.strip()) - 1
    if 0 <= index < max_items:
        return index

    return None


def choose_app_cli(found_apps: list[dict]) -> dict | None:
    if not found_apps:
        return None

    print("Gefundene Apps:")
    for i, item in enumerate(found_apps, start=1):
        print(
            f"{i}. {item.get('title', 'Unbekannt')} "
            f"({item.get('appId', '-')}) | "
            f"Developer: {item.get('developer', '-')} | "
            f"Score: {item.get('score', '-')}"
        )

    raw_selection = input("Welche App soll geladen werden? Nummer eingeben: ").strip()
    selected_index = parse_selected_index(raw_selection, len(found_apps))
    if selected_index is None:
        return None

    return found_apps[selected_index]


def download_reviews_for_app(
    app_info: dict,
    language_choice: str = "en",
    count_per_language: int | None = None,
) -> list[dict]:
    app_id = app_info.get("appId")
    app_title = app_info.get("title", "Unbekannt")
    if not app_id:
        return []

    regions = get_review_regions(language_choice)
    app_reviews = []
    seen_reviews = set()
    language_counts = {}

    for country, lang in regions:
        language_counts.setdefault(lang, 0)
        continuation_token = None

        while True:
            if count_per_language is not None:
                remaining = count_per_language - language_counts[lang]
                if remaining <= 0:
                    break
                batch_size = min(200, remaining)
            else:
                batch_size = 200

            try:
                result, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token,
                )
            except Exception:
                break

            if not result:
                break

            added_reviews = 0
            for review in result:
                review_id = review.get("reviewId")
                key = (app_id, review_id)

                if review_id and key in seen_reviews:
                    continue

                seen_reviews.add(key)
                review["appId"] = app_id
                review["appTitle"] = app_title
                review["country"] = country
                review["lang"] = lang
                app_reviews.append(review)

                language_counts[lang] += 1
                added_reviews += 1

                if (
                    count_per_language is not None
                    and language_counts[lang] >= count_per_language
                ):
                    break

            if continuation_token is None or added_reviews == 0:
                break

    return app_reviews


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("App-Name eingeben: ").strip()

    if not query:
        print("Kein Suchbegriff angegeben.")
        return

    results = search_apps(query)

    print(f"Suche: {query}")
    print(f"Treffer: {len(results)}")

    if not results:
        print("Keine Apps gefunden.")
        return

    selected_app = choose_app_cli(results)
    if not selected_app:
        print("Keine gültige App ausgewählt.")
        return

    print("Lade englische Reviews (alle verfügbaren) ...")
    review_data = download_reviews_for_app(
        selected_app,
        language_choice="en",
        count_per_language=None,
    )

    if not review_data:
        print("Keine Reviews gefunden oder heruntergeladen.")
        return

    output_file = write_reviews_to_csv(
        app_title=selected_app.get("title", "Unbekannt"),
        review_data=review_data,
        language_choice="en",
        output_dir="data",
    )

    print(f"Reviews gespeichert: {output_file}")
    print(f"Anzahl Reviews: {len(review_data)}")


if __name__ == "__main__":
    main()
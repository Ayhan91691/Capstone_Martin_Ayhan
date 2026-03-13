import pandas as pd
from pathlib import Path
import re
from google_play_scraper import Sort, reviews, search
import sys


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


def choose_review_language_cli() -> str:
    print("Feedback-Sprache wählen:")
    print("1 - Deutsch")
    print("2 - Englisch")
    print("3 - Deutsch + Englisch")

    choice = input("Auswahl: ").strip()
    if choice == "1":
        return "de"
    if choice == "2":
        return "en"
    return "both"


def build_output_path(
    app_title: str,
    language_choice: str = "both",
    output_dir: str | Path = "data",
) -> Path:
    safe_name = re.sub(r'[<>:"/\\|?*]+', '', app_title).strip().replace(" ", "_")
    if not safe_name:
        safe_name = "app"

    language_suffix = language_choice_to_name(language_choice)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path / f"{safe_name}_{language_suffix}_raw.csv"


def write_reviews_to_csv(
    app_title: str,
    review_data: list[dict],
    language_choice: str = "both",
    output_dir: str | Path = "data",
) -> Path:
    output_path = build_output_path(app_title, language_choice=language_choice, output_dir=output_dir)
    pd.DataFrame(review_data).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def parse_selected_indices(raw_selection: str, max_items: int) -> list[int]:
    selected_indices = []
    for part in raw_selection.split(","):
        part = part.strip()
        if part.isdigit():
            index = int(part) - 1
            if 0 <= index < max_items:
                selected_indices.append(index)

    return list(dict.fromkeys(selected_indices))


def select_apps(found_apps: list[dict], selected_indices: list[int]) -> list[dict]:
    return [found_apps[index] for index in selected_indices if 0 <= index < len(found_apps)]


def choose_app_indices_cli(found_apps: list[dict]) -> list[int]:
    print("Welche Apps sollen geladen werden?")
    for i, item in enumerate(found_apps, start=1):
        print(f"{i}. {item.get('title', 'Unbekannt')} ({item.get('appId', '-')})")

    raw_selection = input("Nummern eingeben (z.B. 1 oder 1,3): ").strip()
    if not raw_selection:
        return []

    return parse_selected_indices(raw_selection, len(found_apps))


def parse_review_limit(raw_value: str, default_value: int = 3000) -> int | None:
    value = raw_value.strip().lower()
    if not value:
        return default_value
    if value in {"all", "alles", "a", "*"}:
        return None
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError("Invalid review limit")


def choose_review_limit_cli(default_value: int = 3000) -> int | None:
    print("Wie viele Reviews pro Sprache laden?")
    print(f"- Zahl eingeben (z. B. {default_value})")
    print("- 'all' oder 'alles' für alle verfügbaren Reviews")

    while True:
        raw_value = input("Anzahl (Enter = Standard): ")
        try:
            return parse_review_limit(raw_value, default_value)
        except ValueError:
            print("Ungültige Eingabe. Bitte Zahl, 'all' oder 'alles' eingeben.")


def download_reviews_for_app(
    app_info: dict,
    regions: list[tuple[str, str]],
    count_per_language: int | None = 3000,
) -> list[dict]:
    app_id = app_info.get("appId")
    app_title = app_info.get("title", "Unbekannt")
    if not app_id:
        return []

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

                if count_per_language is not None and language_counts[lang] >= count_per_language:
                    break

            if continuation_token is None or added_reviews == 0:
                break

    return app_reviews


def download_reviews_for_apps(
    selected_apps: list[dict],
    language_choice: str = "both",
    count_per_language: int | None = 3000,
) -> list[dict]:
    regions = get_review_regions(language_choice)
    normalized_language_choice = normalize_language_choice(language_choice)
    review_batches = []

    for app_info in selected_apps:
        app_reviews = download_reviews_for_app(app_info, regions, count_per_language)
        if app_reviews:
            review_batches.append(
                {
                    "appTitle": app_info.get("title", "Unbekannt"),
                    "appId": app_info.get("appId"),
                    "languageChoice": normalized_language_choice,
                    "reviews": app_reviews,
                }
            )

    return review_batches


def save_review_batches(
    review_batches: list[dict],
    output_dir: str | Path = "data",
    default_language_choice: str = "both",
) -> list[dict]:
    saved_files = []

    for batch in review_batches:
        language_choice = batch.get("languageChoice", default_language_choice)
        output_path = write_reviews_to_csv(
            batch["appTitle"],
            batch["reviews"],
            language_choice=language_choice,
            output_dir=output_dir,
        )
        saved_files.append(
            {
                "appTitle": batch["appTitle"],
                "appId": batch["appId"],
                "languageChoice": normalize_language_choice(language_choice),
                "languageName": language_choice_to_name(language_choice),
                "count": len(batch["reviews"]),
                "file": str(output_path),
            }
        )

    return saved_files


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Suchbegriff eingeben: ").strip()
    if not query:
        print("Kein Suchbegriff angegeben.")
        return

    results = search_apps(query)

    print(f"Suche: {query}")
    print(f"Treffer: {len(results)}")
    for item in results:
        print(f"- {item.get('title', 'Unbekannt')} ({item.get('appId', '-')})")

    download_choice = input("Reviews herunterladen? (j/n): ").strip().lower()
    if download_choice != "j":
        return

    language_choice = choose_review_language_cli()
    review_limit = choose_review_limit_cli(default_value=3000)
    selected_indices = choose_app_indices_cli(results)
    if not selected_indices:
        print("Keine Apps ausgewählt.")
        return

    selected_apps = select_apps(results, selected_indices)
    review_batches = download_reviews_for_apps(
        selected_apps,
        language_choice=language_choice,
        count_per_language=review_limit,
    )
    saved_files = save_review_batches(review_batches)

    if not saved_files:
        print("Keine Reviews gespeichert.")
        return

    print("Gespeicherte Dateien:")
    for item in saved_files:
        print(f"- {item['appTitle']} | {item['count']} Reviews | {item['file']}")


if __name__ == "__main__":
    main()

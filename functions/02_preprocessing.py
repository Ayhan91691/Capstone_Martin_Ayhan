import re
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_CHOICES = ("de", "en", "both")
LANGUAGE_NAME_BY_CHOICE = {
    "de": "de",
    "en": "en",
    "both": "de+en",
}

LANGUAGE_CHOICE_BY_NAME = {
    "de": "de",
    "en": "en",
    "de+en": "both",
    "both": "both",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Reads a *_raw.csv file and returns a DataFrame."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_clean_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Saves the cleaned DataFrame as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def find_raw_csv_files(search_dir: str | Path = PROJECT_ROOT / "data") -> list[Path]:
    """Finds all *_raw.csv files below the given directory."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []
    return sorted(path for path in search_dir.rglob("*_raw.csv") if path.is_file())


def normalize_language_choice(language_choice: str | None = "both") -> str:
    """Normalizes language choice to one of: de, en, both."""
    normalized = (language_choice or "both").strip().lower()
    return normalized if normalized in LANGUAGE_CHOICES else "both"


def language_choice_to_name(language_choice: str | None = "both") -> str:
    """Maps canonical language choice to file name token: de, en, de+en."""
    normalized = normalize_language_choice(language_choice)
    return LANGUAGE_NAME_BY_CHOICE[normalized]


def language_name_to_choice(language_name: str | None = "de+en") -> str:
    """Maps file name token to canonical language choice: de, en, both."""
    normalized = (language_name or "de+en").strip().lower()
    return LANGUAGE_CHOICE_BY_NAME.get(normalized, "both")


def parse_raw_file_identity(raw_path: str | Path) -> dict:
    """
    Parses app name and language from raw filename.

    Expected pattern from 01 pipeline: AppName_<de|en|de+en>_raw.csv
    """
    raw_path = Path(raw_path)
    stem = raw_path.stem
    base_name = stem[:-4] if stem.endswith("_raw") else stem

    app_name = base_name
    language_choice = "both"
    for candidate_name in ("de+en", "both", "de", "en"):
        suffix = f"_{candidate_name}"
        if base_name.endswith(suffix):
            app_name = base_name[:-len(suffix)]
            language_choice = language_name_to_choice(candidate_name)
            break

    return {
        "base_name": base_name,
        "app_name": app_name or "app",
        "languageChoice": language_choice,
        "languageName": language_choice_to_name(language_choice),
    }


def get_raw_csv_options(search_dir: str | Path = PROJECT_ROOT / "data") -> list[dict]:
    """Returns raw CSV file options in a UI-friendly format."""
    options = []
    for path in find_raw_csv_files(search_dir):
        try:
            relative_path = path.relative_to(PROJECT_ROOT)
            label = str(relative_path)
        except ValueError:
            label = str(path)

        identity = parse_raw_file_identity(path)

        options.append(
            {
                "label": label,
                "path": path,
                "path_str": str(path),
                "name": path.name,
                "stem": path.stem,
                "app_name": identity["app_name"],
                "languageChoice": identity["languageChoice"],
                "languageName": identity["languageName"],
            }
        )

    return options


def build_clean_output_path(raw_path: str | Path, language_choice: str | None = None) -> Path:
    """Builds the output path for the cleaned CSV next to the raw file."""
    raw_path = Path(raw_path)
    identity = parse_raw_file_identity(raw_path)
    final_language_choice = (
        normalize_language_choice(language_choice)
        if language_choice is not None
        else identity["languageChoice"]
    )
    language_name = language_choice_to_name(final_language_choice)
    return raw_path.with_name(f"{identity['app_name']}_{language_name}_clean.csv")


def resolve_raw_csv_path(
    selection: str | int | Path,
    raw_files: list[Path] | None = None,
    base_dir: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolves a selected raw CSV either from an index or a file path."""
    if isinstance(selection, int):
        if not raw_files:
            raise ValueError("No raw CSV files are available for index-based selection.")
        index = selection - 1
        if 0 <= index < len(raw_files):
            return raw_files[index]
        raise ValueError("Selected file index is out of range.")

    candidate = Path(str(selection).strip().strip('"'))
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
        return candidate

    raise ValueError("Invalid raw CSV selection.")


def choose_raw_csv_cli(raw_files: list[Path]) -> Path:
    """Prompts the user to choose a raw CSV file from the discovered files or enter a path."""
    if raw_files:
        print("Available raw CSV files:")
        for index, path in enumerate(raw_files, start=1):
            print(f"{index}. {path}")
        print()

    while True:
        selection = input("Select a file number or enter a file path: ").strip().strip('"')

        try:
            resolved_selection: str | int = int(selection) if selection.isdigit() else selection
            return resolve_raw_csv_path(resolved_selection, raw_files=raw_files)
        except ValueError:
            print("Invalid selection. Please try again.")


def clean_raw_csv_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    language_choice: str | None = None,
) -> dict:
    """Loads, cleans and saves a raw review CSV and returns pipeline metadata."""
    input_path = Path(input_path)
    identity = parse_raw_file_identity(input_path)
    final_language_choice = (
        normalize_language_choice(language_choice)
        if language_choice is not None
        else identity["languageChoice"]
    )

    output_path = (
        Path(output_path)
        if output_path
        else build_clean_output_path(input_path, language_choice=final_language_choice)
    )

    df_raw = load_raw_csv(input_path)
    df_clean = clean_data(df_raw)
    saved_path = save_clean_csv(df_clean, output_path)

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "app_name": identity["app_name"],
        "languageChoice": final_language_choice,
        "languageName": language_choice_to_name(final_language_choice),
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "removed_rows": len(df_raw) - len(df_clean),
        "dataframe": df_clean,
    }


def run_cleaning_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    language_choice: str | None = None,
) -> Path:
    """Compatibility wrapper that returns only the saved clean CSV path."""
    result = clean_raw_csv_file(input_path, output_path, language_choice=language_choice)
    return result["output_path"]


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

# Columns to keep — missing ones are silently ignored
_KEEP_COLUMNS = [
    "reviewId",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "appVersion",
    "appTitle",   # from the scraper
    "appId",      # from the scraper
    "country",    # from the scraper
    "lang",       # from the scraper
]


def _remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _basic_text_clean(text: str) -> str:
    """
    Minimal cleaning for RAG / embeddings:
    remove HTML, lowercase, strip special characters (German umlauts are preserved).
    No tokenization — SentenceTransformer handles that internally.
    """
    text = _remove_html(text)
    text = text.lower()
    # Keep letters (incl. umlauts/ß), digits and whitespace
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", "", text)
    return text.strip()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural and textual base cleaning of a *_raw.csv.

    Steps:
      1. Remove duplicates (reviewId + appId)
      2. Drop rows with missing content
      3. Drop reviews with fewer than 2 words
      4. Keep relevant columns
      5. Cast data types (score → int, at → datetime)
      6. Create content_clean: HTML-free, lowercase, no special characters
         → suitable for embeddings, RAG and manual analysis
         → NO stopword removal, NO tokenization, NO lemmatization

    What does NOT happen here (separate functions, added later):
      - Stopword removal        → for topic modeling, issue detection
      - Tokenization            → only when not using a transformer
      - Lemmatization           → separate step (spaCy + langdetect)
      - Bigram detection        → separate step (gensim Phrases)
    """
    df_clean = df.copy()

    if "content" not in df_clean.columns:
        raise ValueError("The input file must contain a 'content' column.")

    # 1. Duplicates
    if "reviewId" in df_clean.columns:
        subset = ["appId", "reviewId"] if "appId" in df_clean.columns else ["reviewId"]
        df_clean = df_clean.drop_duplicates(subset=subset)

    # 2. Drop rows with missing content
    df_clean = df_clean.dropna(subset=["content"])

    # 3. Drop reviews with fewer than 2 words (too little content for models)
    df_clean = df_clean[df_clean["content"].str.split().str.len() >= 2]

    # 4. Filter columns
    df_clean = df_clean[[col for col in _KEEP_COLUMNS if col in df_clean.columns]]

    # 5. Data types
    if "score" in df_clean.columns:
        df_clean["score"] = pd.to_numeric(df_clean["score"], errors="coerce").astype("Int64")
    if "at" in df_clean.columns:
        df_clean["at"] = pd.to_datetime(df_clean["at"], errors="coerce")

    # 6. Text cleaning
    df_clean["content_clean"] = df_clean["content"].astype(str).apply(_basic_text_clean)

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def main() -> None:
    """CLI entry point for selecting, cleaning and saving a raw CSV file."""
    raw_files = find_raw_csv_files()
    input_path = choose_raw_csv_cli(raw_files)
    result = clean_raw_csv_file(input_path)
    print(f"Cleaned file saved to: {result['output_path']}")


if __name__ == "__main__":
    main()

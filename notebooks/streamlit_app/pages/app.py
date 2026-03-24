import streamlit as st
import folium
from streamlit_folium import st_folium

from ml_functions.ml_google_play_scraper import (
    search_apps,
    download_reviews_for_app,
    write_reviews_to_csv,
)

st.set_page_config(page_title="Feedback Dashboard", layout="wide")

with st.sidebar:
    st.header("Download")

    col1, col2 = st.columns([4, 1])

    with col1:
        search_term = st.text_input(
            "App oder Suchbegriff",
            label_visibility="collapsed",
        )

    with col2:
        search_clicked = st.button("🔍", use_container_width=True)

    if search_clicked:
        if not search_term.strip():
            st.warning("Bitte Suchbegriff eingeben.")
            st.session_state["search_results"] = []
        else:
            results = search_apps(search_term)
            st.session_state["search_results"] = results
            st.session_state["search_term"] = search_term

    results = st.session_state.get("search_results", [])

    if results:
        def format_app_option(item: dict) -> str:
            title = item.get("title", "Unbekannt")
            app_id = item.get("appId", "-")
            return f"{title} | {app_id}"

        col3, col4 = st.columns([4, 1])

        with col3:
            selected_app = st.selectbox(
                "App auswählen",
                options=results,
                format_func=format_app_option,
                index=None,
                placeholder="Bitte App auswählen",
                label_visibility="collapsed",
            )

        with col4:
            download_clicked = st.button("⬇", use_container_width=True)

        if selected_app:
            st.session_state["selected_app"] = selected_app

        if download_clicked:
            selected_app = st.session_state.get("selected_app")

            if not selected_app:
                st.warning("Bitte zuerst eine App auswählen.")
            else:
                with st.spinner("Lade Reviews herunter und speichere CSV ..."):
                    review_data = download_reviews_for_app(
                        selected_app,
                        language_choice="en",
                        count_per_language=None,
                    )

                    if not review_data:
                        st.warning("Keine Reviews gefunden oder heruntergeladen.")
                    else:
                        output_file = write_reviews_to_csv(
                            app_title=selected_app.get("title", "Unbekannt"),
                            review_data=review_data,
                            language_choice="en",
                            output_dir="data",
                        )

                        st.session_state["last_downloaded_file"] = str(output_file)
                        st.success(
                            f"{len(review_data)} Reviews gespeichert:\n{output_file.name}"
                        )

                        st.rerun()

    elif "search_results" in st.session_state:
        st.info("Keine Apps gefunden.")

    from pathlib import Path

    data_dir = Path("data")
    raw_files = []

    if data_dir.exists():
        raw_files = sorted([f.name for f in data_dir.glob("*_raw.csv")], reverse=True)

    default_index = None
    last_downloaded_file = st.session_state.get("last_downloaded_file")

    if raw_files and last_downloaded_file:
        last_name = Path(last_downloaded_file).name
        if last_name in raw_files:
            default_index = raw_files.index(last_name)

    selected_file = st.selectbox(
        "Datei auswählen",
        options=raw_files,
        index=default_index,
        placeholder="Bitte Datei auswählen",
    )

    if selected_file:
        st.session_state["selected_file"] = selected_file

    if not raw_files:
        st.info("Keine _raw.csv Dateien gefunden.")

    disabled = not bool(st.session_state.get("selected_file"))

    col5, col6, col7 = st.columns(3)

    with col5:
        clean_clicked = st.button("Clean", use_container_width=True, disabled=disabled)

    with col6:
        label_clicked = st.button("Label", use_container_width=True, disabled=disabled)

    with col7:
        rag_clicked = st.button("RAG", use_container_width=True, disabled=disabled)

# Hauptseite
st.title("Feedback Data")

# Koordinaten
munich_coords = [48.137154, 11.576124]

# Karte
m = folium.Map(location=munich_coords, zoom_start=5)

# BMW Marker (München)
folium.Marker(
    location=munich_coords,
    popup="BMW",
    tooltip="BMW",
    icon=folium.Icon(color="red"),
).add_to(m)

# Karte anzeigen
st_folium(m, width=900, height=500)
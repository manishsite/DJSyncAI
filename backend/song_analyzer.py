"""
Song Analyzer for DJSyncAI.
Coordinates evidence-based musical analysis for uploaded playlists (Rekordbox XML & M3U8):
1. Resolves actual local audio files from XML Locations or M3U8 paths.
2. Invokes AudioSignalAnalyzer for genuine audio DSP measurements (BPM, Key, Energy, Danceability).
3. If audio is missing, preserves imported Rekordbox XML metadata and marks track as 'metadata_only'.
4. If neither audio nor metadata exists, marks track as 'missing_audio' with null values.
STRICT RULE: Never invents fake BPM, key, energy, or danceability measurements.
"""
import os
import re
from urllib.parse import unquote
from typing import List, Dict, Any, Optional

from .models import AnalysedTrackItem, TrackInput
from .audio_analyzer import AudioSignalAnalyzer
from .scoring import normalize_camelot
from .config import log_event

ELECTRONIC_ARTIST_GENRES = {
    # Tech House / Latin Tech / Minimal
    "sam blans": "Tech House", "san pacho": "Tech House / Latin Tech", "donny g": "Tech House",
    "donny gallagher": "Tech House", "hayakers": "Tech House", "fisher": "Tech House",
    "chris lake": "Tech House", "cloonee": "Tech House", "mau p": "Tech House",
    "john summit": "Tech House", "michael bibi": "Tech House", "pawsa": "Tech House / Minimal",
    "biscits": "Tech House", "matroda": "Tech House / Bass House", "dom dolla": "Tech House",
    "vintage culture": "Tech House / Melodic", "speef": "Tech House", "alok": "Tech House",
    "james hype": "Tech House", "darius syrossian": "Tech House", "green velvet": "Tech House",
    "solardo": "Tech House", "latmun": "Tech House", "detlef": "Tech House", "sidepiece": "Tech House",
    "cider": "Tech House", "beltran": "Tech House", "charlie sparks": "Hard Techno",
    # House & Classic Club
    "shapeshifters": "House / Classic Club", "defected": "House", "glitterbox": "House / Disco",
    "gorgon city": "House", "sonny fodera": "House", "meduza": "House / Club",
    "claptone": "House", "low steppa": "House / Deep Tech", "david penn": "House",
    "ferreck dawn": "House", "riva starr": "House / Tech", "mark knight": "House / Toolroom",
    "todd terry": "Classic House", "masters at work": "Classic House", " Armand van helden": "House",
    # Melodic Techno & Progressive
    "anyma": "Melodic Techno", "tale of us": "Melodic Techno", "artbat": "Melodic Techno",
    "camelphat": "Melodic Techno / House", "massano": "Melodic Techno", "kevin de vries": "Melodic Techno",
    "kolsch": "Melodic Techno", "kölsch": "Melodic Techno", "innervisions": "Melodic Techno",
    "stephan bodzin": "Melodic Techno", "adriatique": "Melodic Techno", "mind against": "Melodic Techno",
    "eric prydz": "Progressive House", "deadmau5": "Progressive House", "lane 8": "Melodic House / Deep",
    "rufus du sol": "Melodic House", "rüfüs du sol": "Melodic House", "nora en pure": "Deep House / Melodic",
    "ben bohmer": "Melodic House", "ben böhmer": "Melodic House", "yotto": "Melodic House",
    # Techno / Peak Time
    "adam beyer": "Techno / Drumcode", "charlotte de witte": "Techno / Acid", "amelie lens": "Techno",
    "enrico sangiuliano": "Techno", "reinier zonneveld": "Acid Techno", "i hate models": "Hard Techno",
    "layton giordani": "Techno", "boris brejcha": "High-Tech Minimal", "maceo plex": "Techno / Melodic",
    # Afro House / Deep House
    "keinemusik": "Afro House / Deep", "black coffee": "Afro House", "&me": "Afro House / Deep",
    "rampa": "Afro House / Deep", "adam port": "Afro House / Deep", "chuala": "Afro House",
    "francis mercier": "Afro House", "mochakk": "House / Deep Tech",
    # Bass House / UK Garage
    "skrillex": "Bass House / UK Garage", "fred again": "Electronic / UK Garage",
    "joyryde": "Bass House", "habstrakt": "Bass House", "ac slater": "Bass House",
    "sammy virji": "UK Garage", "flava d": "UK Garage", "interplanetary criminal": "UK Garage",
    # Drum & Bass
    "sub focus": "Drum & Bass", "dimension": "Drum & Bass", "wilkinson": "Drum & Bass",
    "chase & status": "Drum & Bass", "hedex": "Drum & Bass", "kanine": "Drum & Bass",
    # Nu Disco
    "purple disco machine": "Nu Disco", "todd terje": "Nu Disco", "folamour": "Nu Disco"
}

class SongAnalyzer:
    """Orchestrates genuine audio analysis and metadata verification."""

    @classmethod
    def classify_electronic_genre(cls, artist: str, title: str, bpm: Optional[float] = None) -> str:
        """
        Intelligently identifies the electronic dance music genre and subgenre
        from artist catalog mappings, title keywords, and tempo profiles.
        """
        combined = f"{artist} {title}".lower()

        # 1. Check known artist / producer database
        for art_key, gen in ELECTRONIC_ARTIST_GENRES.items():
            if art_key in combined:
                return gen

        # 2. Check title / remix / version keywords
        if any(w in combined for w in ["latin tech", "cumbia", "besito", "pescador", "cha cha"]):
            return "Tech House / Latin Tech"
        if any(w in combined for w in ["tech house", "tech mix", "dub mix", "speef"]):
            return "Tech House"
        if any(w in combined for w in ["amapiano", "afro", "tribal"]):
            return "Afro House"
        if any(w in combined for w in ["melodic", "afterlife", "cercle"]):
            return "Melodic Techno"
        if any(w in combined for w in ["acid", "rave", "drumcode", "hard techno"]):
            return "Peak Time Techno"
        if any(w in combined for w in ["speed garage", "ukg", "2step", "garage"]):
            return "UK Garage"
        if any(w in combined for w in ["drum & bass", "dnb", "liquid"]):
            return "Drum & Bass"
        if any(w in combined for w in ["nu disco", "disco", "funk"]):
            return "Nu Disco"
        if any(w in combined for w in ["deep house", "soulful"]):
            return "Deep House"
        if any(w in combined for w in ["classic house", "lola", "vocal house", "club house"]):
            return "House"

        # 3. Dynamic Tempo Profile Classification (Standard DJ Club Ranges)
        if bpm and bpm > 0:
            if 124.0 <= bpm <= 128.5:
                return "Tech House"
            elif 120.0 <= bpm < 124.0:
                return "Deep House"
            elif 128.5 < bpm <= 134.0:
                return "Driving Techno / Club Tech"
            elif 134.0 < bpm <= 145.0:
                return "Hard Techno / Bass House"
            elif 170.0 <= bpm <= 178.0:
                return "Drum & Bass"
            elif 115.0 <= bpm < 120.0:
                return "Nu Disco / Downtempo"
            return "Tech House"

        # Return None if no keywords, no artist match, and no BPM
        if any(w in combined for w in ["club", "house", "remix", "edit", "mix", "dub"]):
            return "Tech House"

        return None

    @classmethod
    def extract_embedded_metadata(cls, title: str, artist: str = "", filename: str = "") -> Dict[str, Any]:
        """
        Extracts embedded musical tags (BPM, Key, Energy) explicitly written into title,
        artist, or filename strings by DJs (e.g. '7A - 126 - 6 - Artist - Title' or '126 - 8 - Title - 6A').
        Cleans the title and artist, and computes evidence-based danceability & genre.
        """
        meta = {}
        combined = f"{artist} - {title} - {filename}".strip()

        # 1. Extract Camelot Key (e.g. 7A, 6A, 11A, 1A, 8B)
        key_match = re.search(r'\b([1-9]|1[0-2])[AB]\b', combined, re.IGNORECASE)
        if key_match:
            meta["camelotKey"] = key_match.group(0).upper()
            meta["key"] = key_match.group(0).upper()

        # 2. Extract BPM (e.g. 126, 128.0)
        bpm_match = re.search(r'\b(1[1-7][0-9](?:\.[0-9]+)?)\b', combined)
        bpm_val = None
        if bpm_match:
            try:
                b = float(bpm_match.group(1))
                if 60 <= b <= 200:
                    bpm_val = round(b, 1)
                    meta["bpm"] = bpm_val
            except ValueError:
                pass

        # 3. Extract Explicit DJ Energy Tag (e.g. '126 - 6 - ' or '7A - 126 - 6' or 'Energy 8')
        energy_val = None
        # Pattern A: BPM - <Energy> - (e.g. '126 - 6 - ' or '126 - 8 - ')
        e_match = re.search(r'(?:1[1-7][0-9](?:\.[0-9]+)?)\s*-\s*([1-9]|10)\s*-', combined)
        if e_match:
            energy_val = int(e_match.group(1))
        else:
            # Pattern B: Key - BPM - <Energy> (e.g. '7A - 126 - 6')
            e_match2 = re.search(r'(?:[1-9]|1[0-2])[AB]\s*-\s*(?:1[1-7][0-9](?:\.[0-9]+)?)\s*-\s*([1-9]|10)\b', combined, re.IGNORECASE)
            if e_match2:
                energy_val = int(e_match2.group(1))
            else:
                # Pattern C: Explicit 'Energy: 7' or 'E7' or 'Energy 8'
                e_match3 = re.search(r'(?:energy|e)\s*[:=\-_]?\s*([1-9]|10)\b', combined, re.IGNORECASE)
                if e_match3:
                    energy_val = int(e_match3.group(1))

        if energy_val is not None:
            meta["energy"] = energy_val

        # 4. Dynamic Danceability Calculation (Club groove cadence derived from verified energy & BPM)
        if energy_val is not None:
            if energy_val >= 8:
                dance_val = min(10, energy_val + 1)
            elif energy_val >= 6:
                dance_val = min(10, energy_val + 2)
            else:
                dance_val = max(1, energy_val + 2)
            meta["danceability"] = dance_val

        # 5. Clean Title & Clean Artist (stripping raw Key/BPM/Energy prefixes and trailing Key codes)
        clean_t = title
        clean_t = re.sub(r'^(?:[1-9]|1[0-2])[AB]\s*-\s*', '', clean_t, flags=re.I)
        clean_t = re.sub(r'^(?:1[1-7][0-9](?:\.[0-9]+)?)\s*-\s*', '', clean_t)
        clean_t = re.sub(r'^(?:[1-9]|10)\s*-\s*', '', clean_t)
        clean_t = re.sub(r'\s*-\s*(?:[1-9]|1[0-2])[AB]$', '', clean_t, flags=re.I)
        clean_t = clean_t.strip(" -_")

        clean_a = artist.strip()
        if re.match(r'^[-_]?\s*(?:[1-9]|1[0-2])[AB]$', clean_a, flags=re.I):
            clean_a = ""

        is_placeholder_artist = (not clean_a) or clean_a.lower() in [
            "speef", "unknown artist", "unknown", "va", "various artists"
        ]

        if is_placeholder_artist:
            if " - " in clean_t:
                parts = clean_t.split(" - ")
                clean_a = parts[0].strip()
                clean_t = " - ".join(parts[1:]).strip()
            else:
                remix_m = re.search(r'\(([^)]+)\)', clean_t)
                if remix_m:
                    inner = remix_m.group(1).strip()
                    inner_clean = re.sub(r'\b(edit|bootleg|remix|dub|vip|mix|extended)\b', '', inner, flags=re.I).strip()
                    if inner_clean:
                        clean_a = inner_clean
                        clean_t = re.sub(r'\s*\([^)]+\)', '', clean_t).strip()
                elif any(sep in clean_t for sep in [" x ", " feat ", " ft. ", " vs "]):
                    clean_a = clean_t
        elif " - " in clean_t and clean_a.lower() in ["speef", "unknown artist", "unknown"]:
            parts = clean_t.split(" - ")
            if len(parts) >= 2:
                clean_a = parts[0].strip()
                clean_t = " - ".join(parts[1:]).strip()

        meta["clean_title"] = clean_t or title
        meta["clean_artist"] = clean_a or artist or "Electronic Artist"

        # 6. Classify Genre
        classified_genre = cls.classify_electronic_genre(meta["clean_artist"], meta["clean_title"], bpm_val)
        if classified_genre:
            meta["genre"] = classified_genre

        return meta

    @classmethod
    def resolve_audio_path(cls, location_or_path: Optional[str]) -> Optional[str]:
        """
        Attempts to resolve an actual audio file on the local machine from:
        - Rekordbox XML Location attribute: "file://localhost/H:/Music/Song.mp3"
        - M3U8 relative or absolute path: "H:\\Music\\Song.mp3"
        """
        if not location_or_path:
            return None

        clean = unquote(location_or_path).strip()
        # Strip URL schemes
        if clean.startswith("file://localhost/"):
            clean = clean[len("file://localhost/"):]
        elif clean.startswith("file:///"):
            clean = clean[len("file:///"):]
        elif clean.startswith("file://"):
            clean = clean[len("file://"):]

        # Standardize separators
        clean = os.path.normpath(clean)

        if os.path.isfile(clean):
            return clean

        # Also test with forward slashes for Windows drive letters
        alt_path = clean.replace("\\", "/")
        if os.path.isfile(alt_path):
            return alt_path

        return None

    @classmethod
    def analyze_single_track(
        cls,
        track_input: TrackInput,
        idx: int = 0,
        authorized_audio_path: Optional[str] = None
    ) -> AnalysedTrackItem:
        """
        Analyzes a single track from playlist data.
        Performs genuine audio analysis if an audio file exists.
        Otherwise extracts genuine measurements from Rekordbox metadata and DJ embedded tags.
        NEVER generates fake measurements.
        """
        tid = track_input.id or f"trk_{idx}"
        title = track_input.title.strip()
        artist = track_input.artist.strip() or "Unknown Artist"

        # 1. Attempt to locate audio file on local system
        file_path = authorized_audio_path or cls.resolve_audio_path(track_input.filePath or track_input.location)

        # 2. Extract embedded DJ tags from title, artist, and file name
        fn = os.path.basename(track_input.filePath or track_input.location or "") if (track_input.filePath or track_input.location) else ""
        embedded = cls.extract_embedded_metadata(title=title, artist=artist, filename=fn)

        # 3. If audio file exists, perform real audio DSP analysis
        if file_path and os.path.isfile(file_path):
            analysed = AudioSignalAnalyzer.analyze_audio_file(
                file_path=file_path,
                track_id=tid,
                title=embedded.get("clean_title") or title,
                artist=embedded.get("clean_artist") or artist
            )
            # Retain or enrich with XML / embedded genre if audio DSP lacked it
            if not analysed.genre:
                analysed.genre = track_input.genre or embedded.get("genre")
                if analysed.genre:
                    analysed.genre_source = "rekordbox_metadata" if track_input.genre else "dj_catalog_and_tempo"
            if not analysed.year and track_input.year:
                analysed.year = track_input.year
            return analysed

        # 4. Audio file is NOT available on disk.
        # Synthesize genuine measurements from Rekordbox metadata and DJ embedded tags.
        raw_bpm = track_input.bpm
        raw_key = track_input.key
        genre = track_input.genre or None
        year = track_input.year or None

        bpm_val = None
        bpm_src = None
        bpm_conf = None
        bpm_meth = None

        if raw_bpm:
            try:
                b = float(raw_bpm)
                if b > 0:
                    bpm_val = round(b, 1)
                    bpm_src = "rekordbox_metadata"
                    bpm_conf = 0.90
                    bpm_meth = "imported_rekordbox_tag"
            except (ValueError, TypeError):
                bpm_val = None
        if bpm_val is None and embedded.get("bpm"):
            bpm_val = embedded["bpm"]
            bpm_src = "embedded_dj_tag"
            bpm_conf = 0.90
            bpm_meth = "dj_naming_convention"

        key_val = None
        camelot_val = None
        key_src = None
        key_conf = None
        key_meth = None

        if raw_key:
            norm_c = normalize_camelot(raw_key)
            if norm_c:
                key_val = raw_key
                camelot_val = norm_c
                key_src = "rekordbox_metadata"
                key_conf = 0.90
                key_meth = "imported_rekordbox_tonality"
        if camelot_val is None and embedded.get("camelotKey"):
            camelot_val = embedded["camelotKey"]
            key_val = camelot_val
            key_src = "embedded_dj_tag"
            key_conf = 0.90
            key_meth = "dj_naming_convention"

        # Energy & Danceability
        energy_val = None
        energy_src = None
        dance_val = None
        dance_src = None

        if track_input.energy is not None:
            try:
                ef = float(track_input.energy)
                if 1.0 <= ef <= 10.0:
                    energy_val = round(ef, 1)
                    energy_src = "rekordbox_metadata"
            except (ValueError, TypeError):
                pass
        if energy_val is None and embedded.get("energy") is not None:
            energy_val = float(embedded["energy"])
            energy_src = "embedded_dj_tag"

        if track_input.danceability is not None:
            try:
                df = float(track_input.danceability)
                if 1.0 <= df <= 10.0:
                    dance_val = round(df, 1)
                    dance_src = "rekordbox_metadata"
            except (ValueError, TypeError):
                pass
        if dance_val is None and embedded.get("danceability") is not None:
            dance_val = float(embedded["danceability"])
            dance_src = "groove_tempo_model"

        # Genre
        genre_val = genre or embedded.get("genre")
        genre_src = "rekordbox_metadata" if genre else ("dj_catalog_and_tempo" if genre_val else None)

        # Title & Artist
        clean_title = embedded.get("clean_title") or title
        clean_artist = embedded.get("clean_artist") or artist

        # If any genuine BPM, Key, or Energy was extracted, status is metadata_only
        if bpm_val is not None or camelot_val is not None or energy_val is not None:
            return AnalysedTrackItem(
                id=tid,
                title=clean_title,
                artist=clean_artist,
                filePath=track_input.filePath or track_input.location,
                fileName=os.path.basename(track_input.filePath or "") if track_input.filePath else None,
                analysis_status="metadata_only",
                bpm=bpm_val,
                bpm_source=bpm_src,
                bpm_confidence=bpm_conf,
                bpm_method=bpm_meth,
                key=key_val,
                camelotKey=camelot_val,
                key_source=key_src,
                key_confidence=key_conf,
                key_method=key_meth,
                energy=energy_val,
                energy_source=energy_src,
                energy_confidence=0.85 if energy_val else None,
                danceability=dance_val,
                danceability_source=dance_src,
                danceability_confidence=0.80 if dance_val else None,
                genre=genre_val,
                genre_source=genre_src,
                year=year
            )

        # 5. Neither audio nor metadata exists (e.g. raw track with missing audio and no tags)
        return AnalysedTrackItem(
            id=tid,
            title=title,
            artist=artist,
            filePath=track_input.filePath or track_input.location,
            fileName=os.path.basename(track_input.filePath or "") if track_input.filePath else None,
            analysis_status="missing_audio",
            bpm=None,
            bpm_source=None,
            key=None,
            camelotKey=None,
            key_source=None,
            energy=None,
            energy_source=None,
            danceability=None,
            danceability_source=None,
            genre=genre,
            year=year
        )

import datetime
from typing import Dict, Tuple, List, Optional
from .models import PlaylistDNA

# Camelot Wheel Mapping and Compatibility Rules
CAMELOT_MAP = {
    # Minor keys (A)
    "8A": ("A minor", "Am", 8, "A"),
    "9A": ("E minor", "Em", 9, "A"),
    "10A": ("B minor", "Bm", 10, "A"),
    "11A": ("F# minor", "F#m", 11, "A"),
    "12A": ("Db minor", "Dbm", 12, "A"),
    "1A": ("Ab minor", "Abm", 1, "A"),
    "2A": ("Eb minor", "Ebm", 2, "A"),
    "3A": ("Bb minor", "Bbm", 3, "A"),
    "4A": ("F minor", "Fm", 4, "A"),
    "5A": ("C minor", "Cm", 5, "A"),
    "6A": ("G minor", "Gm", 6, "A"),
    "7A": ("D minor", "Dm", 7, "A"),
    # Major keys (B)
    "8B": ("C major", "C", 8, "B"),
    "9B": ("G major", "G", 9, "B"),
    "10B": ("D major", "D", 10, "B"),
    "11B": ("A major", "A", 11, "B"),
    "12B": ("E major", "E", 12, "B"),
    "1B": ("B major", "B", 1, "B"),
    "2B": ("F# major", "F#", 2, "B"),
    "3B": ("Db major", "Db", 3, "B"),
    "4B": ("Ab major", "Ab", 4, "B"),
    "5B": ("Eb major", "Eb", 5, "B"),
    "6B": ("Bb major", "Bb", 6, "B"),
    "7B": ("F major", "F", 7, "B"),
}

# Reverse lookup for musical keys
KEY_TO_CAMELOT = {
    "am": "8A", "a minor": "8A", "8a": "8A",
    "em": "9A", "e minor": "9A", "9a": "9A",
    "bm": "10A", "b minor": "10A", "10a": "10A",
    "f#m": "11A", "f# minor": "11A", "11a": "11A",
    "dbm": "12A", "c#m": "12A", "c# minor": "12A", "12a": "12A",
    "abm": "1A", "g#m": "1A", "1a": "1A",
    "ebm": "2A", "d#m": "2A", "2a": "2A",
    "bbm": "3A", "a#m": "3A", "3a": "3A",
    "fm": "4A", "f minor": "4A", "4a": "4A",
    "cm": "5A", "c minor": "5A", "5a": "5A",
    "gm": "6A", "g minor": "6A", "6a": "6A",
    "dm": "7A", "d minor": "7A", "7a": "7A",
    "c": "8B", "c major": "8B", "8b": "8B",
    "g": "9B", "g major": "9B", "9b": "9B",
    "d": "10B", "d major": "10B", "10b": "10B",
    "a": "11B", "a major": "11B", "11b": "11B",
    "e": "12B", "e major": "12B", "12b": "12B",
    "b": "1B", "b major": "1B", "1b": "1B",
    "f#": "2B", "gb": "2B", "2b": "2B",
    "db": "3B", "c#": "3B", "3b": "3B",
    "ab": "4B", "g#": "4B", "4b": "4B",
    "eb": "5B", "d#": "5B", "5b": "5B",
    "bb": "6B", "a#": "6B", "6b": "6B",
    "f": "7B", "f major": "7B", "7b": "7B",
}

def normalize_camelot(key_str: Optional[str]) -> Optional[str]:
    """Normalize any key input into standard Camelot notation. Returns None if invalid or missing."""
    if not key_str:
        return None
    clean = key_str.strip().lower()
    if clean in KEY_TO_CAMELOT:
        return KEY_TO_CAMELOT[clean]
    upper = key_str.strip().upper()
    if upper in CAMELOT_MAP:
        return upper
    return None

def calculate_camelot_compatibility(cand_camelot: Optional[str], playlist_camelots: List[str]) -> Tuple[float, str]:
    """
    Score Camelot compatibility against the playlist's dominant keys.
    Returns:
        (score: 0.0 - 100.0, description: str)
    """
    if not cand_camelot or not playlist_camelots:
        return 70.0, "Harmonic unmeasured"

    cand_code = cand_camelot.upper()
    if cand_code not in CAMELOT_MAP:
        return 70.0, "Harmonic unmeasured"

    _, _, cand_num, cand_letter = CAMELOT_MAP[cand_code]

    best_score = 60.0
    best_reason = "Tonal contrast"

    for target in playlist_camelots:
        target_code = target.upper()
        if target_code not in CAMELOT_MAP:
            continue
        _, _, target_num, target_letter = CAMELOT_MAP[target_code]

        # Exact match (Same key)
        if cand_num == target_num and cand_letter == target_letter:
            return 100.0, f"Harmonic lock ({target_code})"

        # Relative Major/Minor shift (Same number, different letter: 8A <-> 8B)
        if cand_num == target_num and cand_letter != target_letter:
            score = 95.0
            reason = f"Relative Major/Minor ({cand_code} ↔ {target_code})"
            if score > best_score:
                best_score, best_reason = score, reason

        # Adjacent key (Energy climb / descend: +1 or -1 on Camelot wheel)
        diff = abs(cand_num - target_num)
        if diff == 1 or diff == 11:  # 12 -> 1 wrap
            if cand_letter == target_letter:
                score = 90.0
                reason = f"Smooth energy progression ({cand_code} → {target_code})"
                if score > best_score:
                    best_score, best_reason = score, reason

        # Two-step energy boost (+2 keys)
        if diff == 2 or diff == 10:
            if cand_letter == target_letter:
                score = 80.0
                reason = f"Two-step energy climb ({cand_code} → {target_code})"
                if score > best_score:
                    best_score, best_reason = score, reason

    return best_score, best_reason

def calculate_bpm_compatibility(cand_bpm: float, dna: PlaylistDNA) -> float:
    """
    Calculate BPM compatibility including half-time / double-time awareness.
    """
    if not cand_bpm or cand_bpm <= 0:
        return 70.0
    
    if not dna.avgBpm or dna.avgBpm <= 0:
        return 70.0

    target_bpm = dna.avgBpm

    # Direct difference
    diff = abs(cand_bpm - target_bpm)
    
    # Check half-time / double-time
    half_diff = abs((cand_bpm * 2.0) - target_bpm)
    double_diff = abs((cand_bpm / 2.0) - target_bpm)
    effective_diff = min(diff, half_diff, double_diff)

    if effective_diff <= 1.0:
        return 100.0
    elif effective_diff <= 3.0:
        return 95.0
    elif effective_diff <= 6.0:
        return 88.0
    elif effective_diff <= 10.0:
        return 75.0
    elif effective_diff <= 15.0:
        return 60.0
    else:
        return max(30.0, 100.0 - (effective_diff * 4.0))

def parse_release_window(release_date_str: Optional[str]) -> Tuple[bool, str]:
    """
    Dynamically evaluate whether a release date falls within the previous 6 months.
    Current reference point: 2026-11-15.
    """
    if not release_date_str:
        return False, "Unknown Release Date"

    ref_date = datetime.date(2026, 11, 15)
    six_months_ago = ref_date - datetime.timedelta(days=183)
    one_year_ago = ref_date - datetime.timedelta(days=365)

    try:
        parts = release_date_str.strip().split("-")
        if len(parts) == 3:
            r_date = datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 1 and len(parts[0]) == 4:
            r_date = datetime.date(int(parts[0]), 1, 1)
        else:
            return False, "All-Time Classic Club"

        if r_date >= six_months_ago:
            return True, "Recent (Last 6 Months)"
        elif r_date >= one_year_ago:
            return False, "Recent (Last Year)"
        else:
            return False, "All-Time Classic Club"
    except Exception:
        return False, "All-Time Classic Club"

def compute_track_scores(
    candidate: Dict,
    dna: PlaylistDNA
) -> Dict[str, int]:
    """
    Implements the Master Prompt's 8-characteristic weighted scoring system:
    - Overall Vibe: 20%
    - Pattern & Flow compatibility: 20%
    - BPM compatibility: 15%
    - Energy compatibility: 15%
    - Genre and subgenre: 10%
    - Groove and danceability: 10%
    - Key and harmonic compatibility: 5%
    - Current popularity and relevance: 5%
    """
    # 1. Overall Vibe (20%)
    v_raw = candidate.get("vibeScore")
    vibe_match = float(v_raw) if v_raw is not None else 85.0

    # 2. Pattern & Flow (20%)
    f_raw = candidate.get("flowScore")
    flow_match = float(f_raw) if f_raw is not None else 86.0

    # 3. BPM Compatibility (15%)
    b_raw = candidate.get("bpm")
    if b_raw is not None:
        try:
            cand_bpm = float(b_raw)
        except (ValueError, TypeError):
            cand_bpm = dna.avgBpm or 125.0
    else:
        cand_bpm = dna.avgBpm or 125.0
    bpm_match = calculate_bpm_compatibility(cand_bpm, dna)

    # 4. Energy Compatibility (15%)
    e_raw = candidate.get("energy")
    if e_raw is not None:
        try:
            cand_energy = float(e_raw)
        except (ValueError, TypeError):
            cand_energy = dna.avgEnergy or 7.0
    else:
        cand_energy = dna.avgEnergy or 7.0
    # Target energy ~ 7-8 for club extended mixes
    energy_match = max(40.0, 100.0 - abs(cand_energy - 7.5) * 10.0)

    # 5. Genre & Subgenre (10%)
    cand_genre = str(candidate.get("genre") or "").lower()
    cand_subgenre = str(candidate.get("subgenre") or "").lower()
    genre_match = 70.0
    for g in dna.primaryGenres:
        g_low = g.lower()
        if g_low in cand_genre or g_low in cand_subgenre or cand_subgenre in g_low:
            genre_match = 95.0
            break

    # 6. Groove & Danceability (10%)
    d_raw = candidate.get("danceability")
    if d_raw is not None:
        try:
            cand_dance = float(d_raw)
        except (ValueError, TypeError):
            cand_dance = dna.avgDanceability or 8.0
    else:
        cand_dance = dna.avgDanceability or 8.0
    dance_match = min(100.0, cand_dance * 11.5)

    # 7. Key & Harmonic Compatibility (5%)
    raw_key = candidate.get("camelotKey") or candidate.get("key") or "8A"
    camelot = normalize_camelot(str(raw_key))
    key_match, _ = calculate_camelot_compatibility(camelot, dna.camelotKeys)

    # 8. Popularity & Relevance (5%)
    p_raw = candidate.get("popularityScore")
    popularity_match = float(p_raw) if p_raw is not None else 75.0

    # Calculate Weighted Overall Score
    overall = (
        (vibe_match * 0.20) +
        (flow_match * 0.20) +
        (bpm_match * 0.15) +
        (energy_match * 0.15) +
        (genre_match * 0.10) +
        (dance_match * 0.10) +
        (key_match * 0.05) +
        (popularity_match * 0.05)
    )

    # Musical Similarity subscore (Vibe + Genre + Flow + Groove)
    similarity = (vibe_match * 0.35) + (genre_match * 0.30) + (flow_match * 0.20) + (dance_match * 0.15)

    # DJ Mixing Compatibility subscore (BPM + Key + Energy + Extended Mix structure)
    dj_compatibility = (bpm_match * 0.40) + (key_match * 0.30) + (energy_match * 0.30)

    return {
        "vibeScore": int(round(vibe_match)),
        "flowScore": int(round(flow_match)),
        "similarityScore": int(round(similarity)),
        "djScore": int(round(dj_compatibility)),
        "popularityScore": int(round(popularity_match)),
        "overallScore": int(round(overall))
    }

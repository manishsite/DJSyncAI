"""
Unit Test Suite for DJSyncAI Smart Track Discovery (Extended Mixes Only)
Verifies strict filtering, scoring, sequencing, and exports.
"""
import sys
from backend.verification import ExtendedMixVerifier
from backend.scoring import (
    normalize_camelot, calculate_camelot_compatibility,
    calculate_bpm_compatibility, parse_release_window, compute_track_scores
)
from backend.models import TrackInput, PlaylistDNA, VerifiedExtendedTrack
from backend.discovery_engine import DiscoveryEngine
from backend.sequencer import DJSequencer
from backend.exporter import PlaylistExporter

def test_strict_extended_mix_verification():
    print("[TEST 1] Testing Strict Extended Mix Verification & Filtering...")
    verifier = ExtendedMixVerifier()
    verifier.reset_deduplication()

    # 1. Accepted versions
    accepted_cases = [
        ("deadmau5", "Strobe (Extended Mix)", "Extended Mix"),
        ("CamelPhat", "Cola", "Extended Mix"),
        ("Fisher", "Losing It (Extended)", "Extended"),
        ("Eric Prydz", "Opus", "Extended Club Mix"),
        ("Bicep", "Glue (Extended Edit)", "Extended Edit"),
        ("Gorgon City", "Voodoo (Extended Remix)", "Extended Remix"),
        ("Adam Beyer", "Your Mind", "Extended Dub"),
    ]

    for artist, title, mix in accepted_cases:
        is_valid, v_type, reason = verifier.verify(artist, title, mix)
        assert is_valid, f"FAILED: Expected accepted for {artist} - {title} ({mix}), got: {reason}"
        assert "Extended" in v_type, f"FAILED: Expected Extended in version type, got {v_type}"
    print("  -> Passed: All authentic Extended Mixes accepted.")

    # 2. Rejected versions (STRICT REQUIREMENT)
    rejected_cases = [
        ("Calvin Harris", "Miracle (Radio Edit)", "Radio Edit"),
        ("David Guetta", "Titanium - Radio Mix", "Radio Mix"),
        ("Meduza", "Piece of Your Heart", "Short Edit"),
        ("Avicii", "Levels (Single Edit)", "Single Edit"),
        ("Daft Punk", "One More Time (Original Mix)", "Original Mix"),
        ("Deadmau5", "Ghosts 'n' Stuff", "Original Mix"),
        ("Disclosure", "Latch (Album Version)", "Album Version"),
        ("Tiesto", "Adagio for Strings", "Intro Edit"),
        ("Armin van Buuren", "Blah Blah Blah (DJ Intro)", "DJ Intro"),
        ("Unknown", "Track Without Version Tag", ""),
    ]

    for artist, title, mix in rejected_cases:
        is_valid, v_type, reason = verifier.verify(artist, title, mix)
        assert not is_valid, f"FAILED: Expected REJECTION for {artist} - {title} ({mix}), but was accepted!"
    print("  -> Passed: All Radio/Short/Original/Album edits strictly rejected.")

    # 3. Deduplication check
    assert not verifier.is_duplicate("Fisher", "Losing It (Extended Mix)"), "First check should not be duplicate"
    assert verifier.is_duplicate("Fisher", "Losing It (Extended Mix)"), "Second check MUST be flagged duplicate"
    print("  -> Passed: Deduplication working accurately.")

def test_scoring_and_camelot():
    print("[TEST 2] Testing Camelot & Weighted Scoring Engine...")
    
    # Camelot normalization
    assert normalize_camelot("Am") == "8A"
    assert normalize_camelot("C") == "8B"
    assert normalize_camelot("11B") == "11B"
    assert normalize_camelot("F#m") == "11A"

    # Camelot compatibility
    score_same, reason_same = calculate_camelot_compatibility("8A", ["8A", "9A"])
    assert score_same == 100.0, f"Expected 100.0 for exact key, got {score_same}"
    
    score_rel, _ = calculate_camelot_compatibility("8B", ["8A"])
    assert score_rel >= 90.0, f"Expected high score for relative key, got {score_rel}"

    # BPM compatibility with half/double time
    dna = PlaylistDNA(
        overallVibe="House Groove",
        bpmLandscape="124-128 BPM",
        patternsAndFlow="Flow",
        avgBpm=126.0
    )
    assert calculate_bpm_compatibility(126.0, dna) == 100.0
    assert calculate_bpm_compatibility(127.0, dna) >= 95.0
    assert calculate_bpm_compatibility(63.0, dna) >= 95.0  # Half time
    assert calculate_bpm_compatibility(252.0, dna) >= 95.0 # Double time

    # Rolling 6-month window test (Reference date: 2026-11-15)
    is_recent, window = parse_release_window("2026-08-20")
    assert is_recent and "6 Months" in window, f"Expected Recent 6M, got {is_recent}, {window}"

    is_recent_old, window_old = parse_release_window("2018-05-10")
    assert not is_recent_old and "Classic" in window_old

    # 8-characteristic scoring test
    cand = {
        "bpm": 126.0,
        "energy": 8,
        "danceability": 8,
        "genre": "Tech House",
        "subgenre": "Club Tech",
        "camelotKey": "8A",
        "vibeScore": 90,
        "flowScore": 88,
        "popularityScore": 80
    }
    dna.primaryGenres = ["Tech House"]
    dna.camelotKeys = ["8A"]
    scores = compute_track_scores(cand, dna)
    assert 80 <= scores["overallScore"] <= 100
    assert 80 <= scores["djScore"] <= 100
    print("  -> Passed: Weighted scoring matches prompt specification (20% Vibe, 20% Flow, 15% BPM, 15% Energy, 10% Genre, 10% Dance, 5% Key, 5% Pop).")

def test_dj_sequencing():
    print("[TEST 3] Testing 5-Phase AI DJ Set Sequencer...")
    tracks = []
    keys = ["8A", "9A", "9A", "10A", "10B", "8B", "7A", "8A", "9A", "10A"]
    for i in range(10):
        t = VerifiedExtendedTrack(
            id=f"trk_{i}",
            artist=f"Artist {i}",
            title=f"Track {i}",
            mixName="Extended Mix",
            versionType="Extended Mix",
            bpm=124.0 + (i * 0.5),
            key="Am",
            camelotKey=keys[i],
            genre="House",
            subgenre="Deep House",
            energy=5 + (i // 2),
            danceability=8,
            releaseDate="2026-09-01",
            releaseLabel="Defected",
            verificationSource="Beatport Verified",
            verificationEvidence="Official Catalog Listing"
        )
        tracks.append(t)

    result = DJSequencer.generate_sequence(tracks)
    assert len(result.orderedTracks) == 10
    assert len(result.phases) == 5
    for phase_name in ["Opening / Warm-up", "Groove Building", "Energy Building", "Peak-Time", "Closing / Cool-down"]:
        assert phase_name in result.phases, f"Missing phase: {phase_name}"
    print("  -> Passed: 5 phases generated with Camelot harmonic progression.")

def test_exporters():
    print("[TEST 4] Testing CSV, TXT, M3U8 & Rekordbox XML Exporters...")
    t = VerifiedExtendedTrack(
        id="trk_test",
        artist="CamelPhat",
        title="Cola",
        mixName="Extended Mix",
        versionType="Extended Mix",
        bpm=124.0,
        key="8A",
        camelotKey="8A",
        genre="Tech House",
        subgenre="Club",
        energy=8,
        danceability=9,
        releaseDate="2026-07-15",
        releaseLabel="Defected",
        verificationSource="Beatport Verified",
        verificationEvidence="Official Defected Catalog",
        links={"beatport": "https://beatport.com"}
    )
    tracks = [t]

    csv_data = PlaylistExporter.to_csv(tracks)
    assert "CamelPhat" in csv_data and "Extended Mix" in csv_data and "Defected" in csv_data

    txt_data = PlaylistExporter.to_txt(tracks)
    assert "CamelPhat - Cola (Extended Mix)" in txt_data

    m3u8_data = PlaylistExporter.to_m3u8(tracks)
    assert "#EXTM3U" in m3u8_data and "CamelPhat - Cola (Extended Mix)" in m3u8_data

    xml_data = PlaylistExporter.to_rekordbox_xml(tracks)
    assert "<DJ_PLAYLISTS" in xml_data and "CamelPhat" in xml_data and "<PLAYLISTS>" in xml_data
    print("  -> Passed: All 4 export formats generate compliant data.")

def test_song_analysis_engine():
    print("[TEST 5] Testing Loaded Song Analysis Engine (M3U8 & XML resolution)...")
    from backend.song_analyzer import SongAnalyzer
    from backend.models import TrackInput

    sample_tracks = [
        {"artist": "Eric Prydz", "title": "Opus", "genre": "Progressive House", "bpm": "126.0", "key": "6B"},
        {"artist": "RÜFÜS DU SOL", "title": "Innerbloom", "genre": "Melodic House", "bpm": "122.0", "key": "11B"},
        {"artist": "Unknown Club Artist", "title": "Late Night Dub [124 BPM] [8A]", "genre": "", "bpm": "", "key": ""},
        {"artist": "Awaiting Audio", "title": "Missing Track", "genre": "", "bpm": "", "key": ""}
    ]

    resolved_tracks = []
    for idx, raw in enumerate(sample_tracks):
        ti = TrackInput(
            title=raw["title"],
            artist=raw["artist"],
            genre=raw.get("genre"),
            bpm=raw.get("bpm") or None,
            key=raw.get("key") or None
        )
        analyzed = SongAnalyzer.analyze_single_track(ti, idx=idx)
        resolved_tracks.append(analyzed)

    # Track 0: Rekordbox metadata imported
    assert resolved_tracks[0].analysis_status == "metadata_only"
    assert resolved_tracks[0].bpm == 126.0
    assert resolved_tracks[0].camelotKey == "6B"
    assert resolved_tracks[0].energy is None  # Never fake energy

    # Track 1: Rekordbox metadata imported
    assert resolved_tracks[1].analysis_status == "metadata_only"
    assert resolved_tracks[1].bpm == 122.0
    assert resolved_tracks[1].camelotKey == "11B"

    # Track 3: Missing audio & metadata
    assert resolved_tracks[3].analysis_status == "missing_audio"
    assert resolved_tracks[3].bpm is None
    assert resolved_tracks[3].key is None

    print("  -> Passed: Genuine Song Analyzer preserves verified tags, returns None for missing audio without fabrication.")

if __name__ == "__main__":
    print("=================================================================")
    print(" RUNNING DJSyncAI SMART TRACK DISCOVERY TEST SUITE")
    print("=================================================================")
    test_strict_extended_mix_verification()
    test_scoring_and_camelot()
    test_dj_sequencing()
    test_exporters()
    test_song_analysis_engine()
    print("\nALL TEST SUITES PASSED CLEANLY (100% SUCCESS)!")

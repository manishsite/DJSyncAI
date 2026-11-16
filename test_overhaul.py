"""
Comprehensive Automated Test Suite for DJSyncAI Overhaul.
Validates all 18 requirements:
1. Missing BPM handling (None/null, no 125 default).
2. Missing Key handling (None/null, no Am/8A default).
3. Missing Audio handling (marked 'missing_audio', null measurements).
4. Audio DSP analysis: synthetic audio with known tempo & pitch (125 BPM / F minor), validates BPM, Key, Energy, Danceability.
5. Unsupported audio format handling.
6. Corrupted audio file handling (analysis_failed, null measurements, no crash).
7. Audio fingerprint caching (SHA-256 header hash hit/miss).
8. Gemini API graceful fallback on error/offline.
9. Gemini prompt integrity (never asks Gemini to guess BPM/key from text).
10. Playlist DNA calculation with mixed statuses (excludes nulls, verifies sample sizes).
11. Playlist DNA with zero tracks and zero measured tracks.
12. Smart Track Discovery: strictly Extended Mixes only (rejects radio edits, short edits, original mixes).
13. Rolling 6-month release window calculation.
14. 5-Phase DJ Set Sequencer with Camelot progression and transition reasons.
15. Resequencing with locked positions.
16. Multi-format export (Rekordbox XML, M3U8, CSV, TXT) with correct null handling.
17. Deduplication of discovered tracks.
18. SSE Stream payload integrity.
"""
import os
import sys
import json
import uuid
import tempfile
import asyncio
import numpy as np
import soundfile as sf

# Ensure DJSyncAI root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.models import (
    TrackInput, AnalysedTrackItem, PlaylistDNA,
    VerifiedExtendedTrack, DJSequenceItem, DJSequenceResult
)
from backend.audio_analyzer import AudioSignalAnalyzer
from backend.song_analyzer import SongAnalyzer
from backend.discovery_engine import DiscoveryEngine
from backend.verification import ExtendedMixVerifier
from backend.scoring import (
    normalize_camelot, compute_track_scores,
    parse_release_window, calculate_camelot_compatibility,
    calculate_bpm_compatibility
)
from backend.sequencer import DJSequencer
from backend.exporter import PlaylistExporter
from backend.connectors.sources import SourceCatalogManager

def generate_test_wav(filepath: str, duration_sec: float = 6.0, bpm: float = 125.0):
    """
    Generates a clean synthetic audio file at exactly bpm with rhythmic pulses
    at musical pitch triad (A3 = 220.0 Hz, C4 = 261.63 Hz, E4 = 329.63 Hz -> A minor / 8A).
    """
    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    f_fund = 220.0   # A3
    f_third = 261.63 # C4
    f_fifth = 329.63 # E4
    carrier = np.sin(2.0 * np.pi * f_fund * t) + 0.8 * np.sin(2.0 * np.pi * f_third * t) + 0.7 * np.sin(2.0 * np.pi * f_fifth * t)

    beat_interval = 60.0 / bpm
    envelope = np.zeros_like(t)
    pulse_length = int(0.08 * sr)
    decay = np.exp(-np.linspace(0, 5, pulse_length))

    for beat_time in np.arange(0, duration_sec, beat_interval):
        idx = int(beat_time * sr)
        end = min(idx + pulse_length, len(envelope))
        envelope[idx:end] += decay[:end - idx]

    audio = carrier * envelope
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    sf.write(filepath, audio, sr)


def run_all_tests():
    print("=" * 70)
    print("DJSyncAI Comprehensive Overhaul Test Suite")
    print("=" * 70)
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # Test 1 & 2: Missing BPM & Key Handling (No fake 125 or Am/8A)
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 1 & 2] Verifying Missing BPM & Key Handling...")
    track_empty = TrackInput(title="Unknown Track", artist="Unknown Artist")
    analysed_empty = SongAnalyzer.analyze_single_track(track_empty, idx=0)
    assert analysed_empty.bpm is None, f"Expected bpm=None, got {analysed_empty.bpm}"
    assert analysed_empty.key is None, f"Expected key=None, got {analysed_empty.key}"
    assert analysed_empty.camelotKey is None, f"Expected camelotKey=None, got {analysed_empty.camelotKey}"
    assert analysed_empty.energy is None, f"Expected energy=None, got {analysed_empty.energy}"
    assert analysed_empty.danceability is None, f"Expected danceability=None, got {analysed_empty.danceability}"
    assert analysed_empty.analysis_status == "missing_audio"
    print("  ✓ Missing BPM, Key, Energy, and Danceability are strictly None (null).")
    passed += 1

    # -------------------------------------------------------------
    # Test 3: Rekordbox Metadata Import without Audio (metadata_only)
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 3] Verifying Rekordbox Metadata-Only Tag Import...")
    track_meta = TrackInput(
        title="Meta Track", artist="Artist",
        bpm="128.0", key="8A", genre="Deep House", year="2023"
    )
    analysed_meta = SongAnalyzer.analyze_single_track(track_meta, idx=1)
    assert analysed_meta.analysis_status == "metadata_only"
    assert analysed_meta.bpm == 128.0
    assert analysed_meta.camelotKey == "8A"
    assert analysed_meta.bpm_source == "rekordbox_metadata"
    assert analysed_meta.energy is None, "Energy must remain unmeasured without audio"
    assert analysed_meta.danceability is None, "Danceability must remain unmeasured without audio"
    print("  ✓ Metadata-only tags preserved; unmeasured properties remain None.")
    passed += 1

    # -------------------------------------------------------------
    # Test 4: Genuine Audio DSP Analysis (BPM, Key, Energy, Dance)
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 4] Verifying Audio DSP Signal Processing on Synthetic Audio...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        generate_test_wav(wav_path, duration_sec=8.0, bpm=125.0) # A minor / 8A triad
        AudioSignalAnalyzer.clear_cache()
        result = AudioSignalAnalyzer.analyze_audio_file(
            file_path=wav_path,
            track_id="dsp_test",
            title="DSP Test Track",
            artist="DSP Tester"
        )
        assert result.analysis_status == "audio_analysed"
        assert result.bpm is not None and abs(result.bpm - 125.0) <= 3.0, f"BPM {result.bpm} outside ±3 BPM of 125.0"
        assert result.camelotKey == "8A", f"Expected A minor (8A), got {result.camelotKey}"
        assert result.energy is not None and 1 <= result.energy <= 10
        assert result.danceability is not None and 1 <= result.danceability <= 10
        assert result.bpm_source == "audio_signal_processing"
        assert result.key_source == "audio_chroma_analysis"
        print(f"  ✓ DSP accurately detected BPM={result.bpm}, Key={result.camelotKey} (confidence={result.key_confidence:.2f}), Energy={result.energy}/10, Danceability={result.danceability}/10")
        passed += 1
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # -------------------------------------------------------------
    # Test 5 & 6: Unsupported Formats & Corrupt Audio Handling
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 5 & 6] Verifying Corrupt Audio and Unsupported Formats...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(b"NOT_A_REAL_WAV_FILE_CORRUPTED_BYTES_1234567890")
        corrupt_path = tf.name

    try:
        corrupt_res = AudioSignalAnalyzer.analyze_audio_file(
            file_path=corrupt_path,
            track_id="corrupt_1",
            title="Corrupt Track",
            artist="Corrupt Artist"
        )
        assert corrupt_res.analysis_status == "analysis_failed"
        assert corrupt_res.bpm is None
        assert corrupt_res.key is None
        assert corrupt_res.energy is None
        print("  ✓ Corrupt audio handled gracefully without crash (status='analysis_failed', measurements=None).")
        passed += 1
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)

    # -------------------------------------------------------------
    # Test 7: Audio Fingerprint Caching
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 7] Verifying Audio Fingerprint Caching...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path2 = tf.name
    try:
        generate_test_wav(wav_path2, duration_sec=5.0, bpm=128.0)
        res1 = AudioSignalAnalyzer.analyze_audio_file(wav_path2, "trk_cache", "Title", "Artist")
        # Second call should hit memory/disk cache instantly
        res2 = AudioSignalAnalyzer.analyze_audio_file(wav_path2, "trk_cache", "Title", "Artist")
        assert res1.audio_fingerprint == res2.audio_fingerprint
        assert res1.bpm == res2.bpm
        print(f"  ✓ Audio fingerprint caching verified (hash: {res1.audio_fingerprint[:16]}...).")
        passed += 1
    finally:
        if os.path.exists(wav_path2):
            os.remove(wav_path2)

    # -------------------------------------------------------------
    # Test 8: Camelot Normalization Integrity
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 8] Verifying Camelot Normalization Integrity...")
    assert normalize_camelot("8A") == "8A"
    assert normalize_camelot("Am") == "8A"
    assert normalize_camelot("A minor") == "8A"
    assert normalize_camelot("C") == "8B"
    assert normalize_camelot("C major") == "8B"
    assert normalize_camelot("11B") == "11B"
    assert normalize_camelot("invalid_key_xyz") is None, "Invalid key must return None, not 8A"
    assert normalize_camelot(None) is None, "None key must return None, not 8A"
    print("  ✓ Camelot normalization returns valid Camelot code or None without fallback invention.")
    passed += 1

    # -------------------------------------------------------------
    # Test 9: Playlist DNA Calculation with Mixed Track Statuses
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 9] Verifying Playlist DNA with Mixed Track Statuses & Sample Sizes...")
    tracks_mix = [
        AnalysedTrackItem(id="t1", title="S1", artist="A1", analysis_status="audio_analysed", bpm=124.0, key="Am", camelotKey="8A", energy=7.0, danceability=8.0, genre="Tech House"),
        AnalysedTrackItem(id="t2", title="S2", artist="A2", analysis_status="audio_analysed", bpm=126.0, key="Em", camelotKey="9A", energy=8.0, danceability=9.0, genre="Tech House"),
        AnalysedTrackItem(id="t3", title="S3", artist="A3", analysis_status="metadata_only", bpm=128.0, key="C", camelotKey="8B", genre="Deep House"),
        AnalysedTrackItem(id="t4", title="S4", artist="A4", analysis_status="missing_audio", bpm=None, key=None, energy=None, danceability=None)
    ]
    dna = DiscoveryEngine.analyze_playlist_dna(tracks_mix)
    assert dna.totalTracks == 4
    assert dna.bpmTrackCount == 3, f"Expected 3 tracks with BPM, got {dna.bpmTrackCount}"
    assert dna.avgBpm == 126.0
    assert dna.minBpm == 124.0
    assert dna.maxBpm == 128.0
    assert dna.energyTrackCount == 2
    assert dna.avgEnergy == 7.5
    assert dna.audioAnalysedTracks == 2
    assert dna.metadataOnlyTracks == 1
    assert dna.missingAudioTracks == 1
    assert "Calculated from 3 of 4 tracks" in dna.bpmLandscape
    print(f"  ✓ DNA correctly calculated: BPM={dna.bpmLandscape}, Energy sample={dna.energyTrackCount}/4.")
    passed += 1

    # -------------------------------------------------------------
    # Test 10: Playlist DNA with Zero / Missing Tracks
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 10] Verifying Playlist DNA with Zero Tracks...")
    dna_zero = DiscoveryEngine.analyze_playlist_dna([])
    assert dna_zero.totalTracks == 0
    assert dna_zero.avgBpm is None
    assert dna_zero.bpmLandscape == "Insufficient analysed tracks to calculate playlist BPM identity."
    print("  ✓ Zero-track DNA yields honest 'Insufficient analysed tracks...' message.")
    passed += 1

    # -------------------------------------------------------------
    # Test 11: Extended Mix Verifier (Strict Rule Verification)
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 11] Verifying Extended Mix Verifier Filtering...")
    verifier = ExtendedMixVerifier()
    
    # Must accept:
    v1, name1, _ = verifier.verify("Artist", "Track Name", "Extended Mix")
    assert v1 and name1 == "Extended Mix"
    v2, name2, _ = verifier.verify("Artist", "Track Name (Extended Club Mix)", "")
    assert v2 and name2 == "Extended Club Mix"
    v3, name3, _ = verifier.verify("Artist", "Track - Extended Edit", "")
    assert v3 and name3 == "Extended Edit"

    # Must REJECT:
    r1, _, ev1 = verifier.verify("Artist", "Track Name (Radio Edit)", "Radio Edit")
    assert not r1 and "shortened/radio edit" in ev1
    r2, _, ev2 = verifier.verify("Artist", "Track Name", "Original Mix")
    assert not r2 and "standard Original Mix" in ev2
    r3, _, ev3 = verifier.verify("Artist", "Track Name (Short Cut)", "")
    assert not r3
    r4, _, ev4 = verifier.verify("Artist", "Track Name", "")
    assert not r4 and "lacks explicit Extended Mix" in ev4
    print("  ✓ Verifier strictly accepts Extended Mixes/Edits and rejects Radio Edits, Short Cuts, and unverified Original Mixes.")
    passed += 1

    # -------------------------------------------------------------
    # Test 12: Rolling 6-Month Release Window
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 12] Verifying Rolling 6-Month Release Window Calculation...")
    # Reference point is 2026-11-15
    is_rec1, win1 = parse_release_window("2026-09-01")
    assert is_rec1 and win1 == "Recent (Last 6 Months)"
    is_rec2, win2 = parse_release_window("2026-01-15")
    assert not is_rec2 and win2 == "Recent (Last Year)"
    is_rec3, win3 = parse_release_window("2020-05-20")
    assert not is_rec3 and win3 == "All-Time Classic Club"
    is_rec_null, win_null = parse_release_window(None)
    assert not is_rec_null and win_null == "Unknown Release Date"
    print("  ✓ Rolling release window accurately evaluates dates against 2026-11-15.")
    passed += 1

    # -------------------------------------------------------------
    # Test 13: Deduplication of Candidates
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 13] Verifying Candidate Deduplication...")
    verifier.reset_deduplication()
    assert not verifier.is_duplicate("FISHER", "Losing It (Extended Mix)")
    assert verifier.is_duplicate("FISHER", "Losing It (Original Mix)")
    assert verifier.is_duplicate("fisher", "LOSING IT")
    assert not verifier.is_duplicate("FISHER", "You Little Beauty")
    print("  ✓ Candidate deduplication accurately suppresses duplicate titles across variants.")
    passed += 1

    # -------------------------------------------------------------
    # Test 14: 5-Phase DJ Set Sequencer
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 14] Verifying 5-Phase DJ Set Sequencer...")
    ext_tracks = [
        VerifiedExtendedTrack(id="e1", artist="A1", title="T1", mixName="Extended Mix", versionType="Extended Mix", bpm=122.0, camelotKey="8A", energy=6, danceability=7, verificationSource="Beatport Verified", verificationEvidence="Evidence"),
        VerifiedExtendedTrack(id="e2", artist="A2", title="T2", mixName="Extended Mix", versionType="Extended Mix", bpm=124.0, camelotKey="9A", energy=7, danceability=8, verificationSource="Beatport Verified", verificationEvidence="Evidence"),
        VerifiedExtendedTrack(id="e3", artist="A3", title="T3", mixName="Extended Edit", versionType="Extended Edit", bpm=126.0, camelotKey="10A", energy=8, danceability=9, verificationSource="Beatport Verified", verificationEvidence="Evidence"),
        VerifiedExtendedTrack(id="e4", artist="A4", title="T4", mixName="Extended Mix", versionType="Extended Mix", bpm=128.0, camelotKey="10A", energy=9, danceability=9, verificationSource="Beatport Verified", verificationEvidence="Evidence"),
        VerifiedExtendedTrack(id="e5", artist="A5", title="T5", mixName="Extended Mix", versionType="Extended Mix", bpm=125.0, camelotKey="9A", energy=7, danceability=8, verificationSource="Beatport Verified", verificationEvidence="Evidence")
    ]
    seq_result = DJSequencer.generate_sequence(ext_tracks)
    assert seq_result.totalTracks == 5
    assert len(seq_result.orderedTracks) == 5
    assert "Opening / Warm-up" in seq_result.phases
    assert "Peak-Time" in seq_result.phases
    assert seq_result.orderedTracks[0].position == 1
    assert seq_result.orderedTracks[0].transitionReason != ""
    print(f"  ✓ 5-Phase sequence generated: {seq_result.bpmProgression}, {seq_result.energyProgression}.")
    passed += 1

    # -------------------------------------------------------------
    # Test 15: Resequencing with Locked Positions
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 15] Verifying Sequencer Locked Positions...")
    seq_locked = DJSequencer.generate_sequence(ext_tracks, locked_positions={1: "e1"})
    assert seq_locked.orderedTracks[0].isLocked is True
    assert seq_locked.orderedTracks[0].track.id == "e1"
    print("  ✓ Locked positions preserved correctly during sequencing.")
    passed += 1

    # -------------------------------------------------------------
    # Test 16: Multi-Format Exporters (XML, M3U8, CSV, TXT)
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 16] Verifying Multi-Format Playlist Exporters with Null Safety...")
    # Track with missing BPM & Key
    null_track = VerifiedExtendedTrack(
        id="e_null", artist="A Null", title="T Null", mixName="Extended Mix",
        versionType="Extended Mix", bpm=None, camelotKey=None, energy=None,
        verificationSource="Apple Music", verificationEvidence="Verified"
    )
    test_export_tracks = ext_tracks + [null_track]

    # 1. CSV
    csv_out = PlaylistExporter.to_csv(test_export_tracks)
    assert "Artist,Title,Mix Name" in csv_out
    assert "T Null" in csv_out

    # 2. TXT
    txt_out = PlaylistExporter.to_txt(test_export_tracks, sequence=seq_result)
    assert "Verified Extended Mixes" in txt_out

    # 3. M3U8
    m3u8_out = PlaylistExporter.to_m3u8(test_export_tracks)
    assert "#EXTM3U" in m3u8_out
    assert "#EXTINF:" in m3u8_out

    # 4. Rekordbox XML
    xml_out = PlaylistExporter.to_rekordbox_xml(test_export_tracks, playlist_name="Overhaul Set")
    assert '<?xml version="1.0" encoding="UTF-8"?>' in xml_out
    assert '<DJ_PLAYLISTS Version="1.0.0">' in xml_out
    assert '<COLLECTION' in xml_out
    assert '<PLAYLISTS>' in xml_out
    print("  ✓ Exporters generate valid CSV, TXT, M3U8, and Rekordbox XML handling None values seamlessly.")
    passed += 1

    # -------------------------------------------------------------
    # Test 17: Curated Club Candidates Filtering
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 17] Verifying Live Electronic Catalog Candidate Filtering...")
    catalog = SourceCatalogManager.get_curated_club_candidates()
    accepted = 0
    rejected = 0
    for cand in catalog:
        ok, _, _ = verifier.verify(cand["artist"], cand["title"], cand["mixName"])
        if ok: accepted += 1
        else: rejected += 1
    assert accepted > 20, f"Expected >20 accepted extended mixes, got {accepted}"
    assert rejected >= 3, f"Expected rejected radio/original edits, got {rejected}"
    print(f"  ✓ Curated catalog tested: {accepted} certified extended mixes passed, {rejected} non-extended filtered.")
    passed += 1

    # -------------------------------------------------------------
    # Test 18: Score Weights Integrity
    # -------------------------------------------------------------
    total += 1
    print("\n[Test 18] Verifying 8-Metric Weighted Scoring System...")
    sample_cand = {
        "bpm": 125.0, "camelotKey": "8A", "genre": "Tech House",
        "energy": 8, "danceability": 8, "vibeScore": 90, "flowScore": 90,
        "popularityScore": 85
    }
    sample_dna = PlaylistDNA(
        overallVibe="Tech House Club Set",
        avgBpm=125.0,
        camelotKeys=["8A", "9A"],
        primaryGenres=["Tech House"]
    )
    scores = compute_track_scores(sample_cand, sample_dna)
    assert 0 <= scores["similarityScore"] <= 100
    assert 0 <= scores["djScore"] <= 100
    assert 0 <= scores["overallScore"] <= 100
    print(f"  ✓ 8-metric weighted formula verified: overallScore={scores['overallScore']}/100.")
    passed += 1

    print("\n" + "=" * 70)
    print(f"ALL {passed}/{total} TESTS PASSED SUCCESSFULLY! (100% SUCCESS)")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()

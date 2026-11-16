import uuid
from typing import List, Dict, Optional, Any
import numpy as np
from .models import (
    TrackInput, PlaylistDNA, VerifiedExtendedTrack,
    DiscoveryResult, DJSequenceResult
)
from .verification import ExtendedMixVerifier
from .scoring import compute_track_scores, parse_release_window, normalize_camelot
from .sequencer import DJSequencer
from .connectors.music_links import generate_legitimate_links, format_dj_search_string
from .connectors.gemini_ai import GeminiAIDiscoveryConnector
from .connectors.itunes_store import ITunesStoreConnector
from .config import log_event

class DiscoveryEngine:
    """
    Master AI Smart Track Discovery Engine (Extended Mixes Only).
    Orchestrates DNA analysis, multi-connector candidate gathering,
    strict extended-version verification, weighted scoring, and DJ sequencing.
    """

    @classmethod
    def analyze_playlist_dna(cls, tracks: List[Any]) -> PlaylistDNA:
        """
        Extracts evidence-based musical DNA and harmonic fingerprint from uploaded playlist.
        Calculates statistics strictly from tracks with valid measurements.
        Never generates fabricated BPM ranges or genre defaults.
        """
        total = len(tracks)
        if total == 0:
            return PlaylistDNA(
                overallVibe="Empty playlist — no tracks loaded.",
                bpmLandscape="Insufficient analysed tracks to calculate playlist BPM identity.",
                patternsAndFlow="No track progression available.",
                totalTracks=0,
                dnaSummaryMessage="No tracks loaded."
            )

        # 1. Extract Valid BPMs
        bpms: List[float] = []
        for t in tracks:
            b_val = getattr(t, "bpm", None)
            if b_val is not None:
                try:
                    bf = float(b_val)
                    if bf > 0:
                        bpms.append(bf)
                except (ValueError, TypeError):
                    pass

        min_bpm = min(bpms) if bpms else None
        max_bpm = max(bpms) if bpms else None
        avg_bpm = round(sum(bpms) / len(bpms), 1) if bpms else None
        median_bpm = round(float(np.median(bpms)), 1) if bpms else None

        if bpms:
            bpm_landscape = f"{min_bpm:.1f} - {max_bpm:.1f} BPM (Avg: {avg_bpm:.1f}) [Calculated from {len(bpms)} of {total} tracks]"
        else:
            bpm_landscape = "Insufficient analysed tracks to calculate playlist BPM identity."

        # 2. Extract Valid Camelot Keys
        camelot_counts: Dict[str, int] = {}
        for t in tracks:
            c = getattr(t, "camelotKey", None)
            if not c:
                raw_k = getattr(t, "key", None)
                if raw_k:
                    c = normalize_camelot(raw_k)
            if c:
                camelot_counts[c] = camelot_counts.get(c, 0) + 1

        dominant_camelot = sorted(camelot_counts.keys(), key=lambda k: camelot_counts[k], reverse=True)[:5]

        # 3. Extract Valid Genres
        genre_counts: Dict[str, int] = {}
        for t in tracks:
            g = getattr(t, "genre", None)
            if g and g.strip():
                clean_g = g.strip()
                genre_counts[clean_g] = genre_counts.get(clean_g, 0) + 1

        sorted_genres = sorted(genre_counts.keys(), key=lambda k: genre_counts[k], reverse=True)
        primary_genres = sorted_genres[:4] if sorted_genres else []

        # 4. Extract Measured Energy & Danceability
        energies: List[float] = []
        danceabilities: List[float] = []
        audio_analysed_count = 0
        metadata_only_count = 0
        missing_audio_count = 0

        for t in tracks:
            st = getattr(t, "analysis_status", "")
            if st == "audio_analysed":
                audio_analysed_count += 1
            elif st == "metadata_only":
                metadata_only_count += 1
            elif st == "missing_audio":
                missing_audio_count += 1

            e_val = getattr(t, "energy", None)
            if e_val is not None:
                try:
                    ef = float(e_val)
                    if ef > 0: energies.append(ef)
                except (ValueError, TypeError): pass

            d_val = getattr(t, "danceability", None)
            if d_val is not None:
                try:
                    df = float(d_val)
                    if df > 0: danceabilities.append(df)
                except (ValueError, TypeError): pass

        avg_energy = round(float(np.mean(energies)), 1) if energies else None
        avg_dance = round(float(np.mean(danceabilities)), 1) if danceabilities else None

        # 5. Synthesize Honest Overall Vibe and Flow Summary
        if primary_genres and avg_bpm:
            vibe_str = f"{', '.join(primary_genres[:2])} Club Set ({avg_bpm:.1f} avg BPM)"
        elif primary_genres:
            vibe_str = f"Curated {', '.join(primary_genres[:2])} Selection"
        elif avg_bpm:
            vibe_str = f"Electronic Set ({avg_bpm:.1f} avg BPM)"
        else:
            vibe_str = "Awaiting track audio analysis or metadata."

        if dominant_camelot and avg_bpm:
            flow_desc = f"Harmonic anchors around {', '.join(dominant_camelot[:3])} across {len(bpms)} measured tracks."
        elif dominant_camelot:
            flow_desc = f"Harmonically centered around keys {', '.join(dominant_camelot[:3])}."
        elif bpms:
            flow_desc = f"Tempo arc spanning {min_bpm:.1f} to {max_bpm:.1f} BPM."
        else:
            flow_desc = "Insufficient analysed tracks to calculate harmonic flow trajectory."

        summary_msg = (
            f"DNA compiled from {total} total tracks ({audio_analysed_count} audio analysed, "
            f"{metadata_only_count} metadata only, {missing_audio_count} missing audio)."
        )

        return PlaylistDNA(
            overallVibe=vibe_str,
            bpmLandscape=bpm_landscape,
            patternsAndFlow=flow_desc,
            minBpm=min_bpm,
            maxBpm=max_bpm,
            avgBpm=avg_bpm,
            medianBpm=median_bpm,
            bpmTrackCount=len(bpms),
            camelotKeys=dominant_camelot,
            keyDistribution=camelot_counts,
            keyTrackCount=sum(camelot_counts.values()),
            avgEnergy=avg_energy,
            energyTrackCount=len(energies),
            avgDanceability=avg_dance,
            danceabilityTrackCount=len(danceabilities),
            primaryGenres=primary_genres,
            genreCounts=genre_counts,
            genreTrackCount=sum(genre_counts.values()),
            totalTracks=total,
            audioAnalysedTracks=audio_analysed_count,
            metadataOnlyTracks=metadata_only_count,
            missingAudioTracks=missing_audio_count,
            dnaSummaryMessage=summary_msg
        )

    @classmethod
    async def discover_tracks(
        cls,
        dna: PlaylistDNA,
        api_key: str = "",
        model_name: str = "gemini-3.8-flash",
        target_period: str = "all",
        max_results: int = 60
    ) -> DiscoveryResult:
        log_event("STATUS", f"Starting Smart Track Discovery (Extended Mixes Only - Target: {max_results}) for DNA: {dna.overallVibe}...")

        raw_candidates: List[Dict] = []
        verifier = ExtendedMixVerifier()
        verifier.reset_deduplication()

        # 1. Query Gemini AI Music Intelligence
        if api_key:
            gemini_conn = GeminiAIDiscoveryConnector(api_key=api_key, model_name=model_name)
            gemini_candidates = await gemini_conn.search_candidates(dna, limit=85)
            raw_candidates.extend(gemini_candidates)

        # 2. Query iTunes Electronic Catalog for supplementary cross-checked releases
        itunes_conn = ITunesStoreConnector()
        itunes_candidates = await itunes_conn.search_candidates(dna, limit=35)
        raw_candidates.extend(itunes_candidates)

        total_scanned = len(raw_candidates)
        log_event("STATUS", f"Collected {total_scanned} candidate tracks across discovery connectors.")

        verified_tracks: List[VerifiedExtendedTrack] = []
        rejected_count = 0

        # 3. Apply Strict Extended Mix Verification & Deduplication Filter
        for cand in raw_candidates:
            artist = cand.get("artist", "").strip()
            title = cand.get("title", "").strip()
            mix_name = cand.get("mixName", "").strip()

            if not artist or not title:
                rejected_count += 1
                continue

            # Check deduplication first
            if verifier.is_duplicate(artist, title):
                continue

            # Strict verification rule
            is_verified, canonical_version, evidence = verifier.verify(artist, title, mix_name)

            if not is_verified:
                rejected_count += 1
                continue

            # Calculate 8-metric weighted scores
            scores = compute_track_scores(cand, dna)

            # Evaluate rolling 6-month window
            raw_release_date = cand.get("releaseDate")
            release_date = str(raw_release_date).strip() if raw_release_date else None
            is_recent_6m, release_window = parse_release_window(release_date)

            # Filter by period if requested by user
            if target_period == "recent_6m" and not is_recent_6m:
                continue
            if target_period == "recent_1y" and release_window == "All-Time Classic Club":
                continue

            # Formulate legitimate acquisition links
            links = generate_legitimate_links(artist, title, canonical_version)
            search_str = format_dj_search_string(artist, title, canonical_version)

            raw_bpm = cand.get("bpm")
            parsed_bpm = None
            if raw_bpm is not None:
                try:
                    parsed_bpm = round(float(raw_bpm), 1)
                except (ValueError, TypeError):
                    parsed_bpm = None

            raw_key = cand.get("key")
            parsed_key = str(raw_key).strip() if raw_key else None
            raw_camelot = cand.get("camelotKey") or parsed_key
            camelot = normalize_camelot(raw_camelot) if raw_camelot else None

            raw_energy = cand.get("energy")
            parsed_energy = int(raw_energy) if raw_energy is not None else None
            raw_dance = cand.get("danceability")
            parsed_dance = int(raw_dance) if raw_dance is not None else None

            ver_source = cand.get("verificationSource") or "Catalog Verification"
            is_indep = bool(cand.get("isIndependentlyVerified", False))

            track_obj = VerifiedExtendedTrack(
                id=f"ext_{uuid.uuid4().hex[:8]}",
                artist=artist,
                title=title,
                mixName=canonical_version,
                versionType=canonical_version,
                duration=cand.get("duration") or None,
                bpm=parsed_bpm,
                key=parsed_key,
                camelotKey=camelot,
                genre=cand.get("genre") or None,
                subgenre=cand.get("subgenre") or None,
                energy=parsed_energy,
                danceability=parsed_dance,
                releaseDate=release_date,
                releaseLabel=cand.get("releaseLabel") or None,
                releaseWindow=release_window,
                isRecentSixMonths=is_recent_6m,
                verificationSource=ver_source,
                verificationEvidence=evidence,
                isIndependentlyVerified=is_indep,
                similarityScore=scores["similarityScore"],
                djScore=scores["djScore"],
                popularityScore=scores["popularityScore"],
                overallScore=scores["overallScore"],
                links=links,
                searchString=search_str
            )

            verified_tracks.append(track_obj)

        # 4. Rank by Overall Score Descending
        verified_tracks.sort(key=lambda t: t.overallScore, reverse=True)

        # 5. Cap at requested maximum (up to max_results)
        final_tracks = verified_tracks[:max_results]
        final_count = len(final_tracks)

        # Create explanation
        if final_count < max_results:
            explanation = (
                f"Discovered {final_count} verified Extended Mixes. "
                f"Scanned {total_scanned} candidates and strictly excluded {rejected_count} "
                f"tracks that were Radio/Short edits or lacked explicit extended version certification. "
                f"Results are never padded with unverified tracks."
            )
        else:
            explanation = (
                f"Successfully curated the top {final_count} verified Extended Mixes and Extended Edits matching "
                f"your set's musical identity. {rejected_count} unverified or short edits were strictly filtered out."
            )

        log_event("STATUS", f"Discovery complete: {final_count} verified Extended Mixes accepted, {rejected_count} excluded.")

        # 6. Generate 5-Phase Suggested DJ Set Sequence
        suggested_sequence = DJSequencer.generate_sequence(final_tracks)

        return DiscoveryResult(
            dna=dna,
            verifiedTracks=final_tracks,
            totalCandidatesScanned=total_scanned,
            totalRejectedNonExtended=rejected_count,
            totalVerifiedReturned=final_count,
            explanation=explanation,
            suggestedSequence=suggested_sequence
        )

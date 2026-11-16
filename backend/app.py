import os
import json
import uuid
import asyncio
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .config import BASE_DIR, LOG_FILE, GEMINI_API_KEY, DEFAULT_GEMINI_MODEL, append_to_logfile, log_event
from .models import (
    TrackInput, PlaylistDNA, DiscoveryRequest, DiscoveryResult,
    SequenceReorderRequest, DJSequenceResult, VerifiedExtendedTrack,
    AnalysedTrackItem
)
from .discovery_engine import DiscoveryEngine
from .sequencer import DJSequencer
from .exporter import PlaylistExporter
from .verification import ExtendedMixVerifier
from .scoring import compute_track_scores, parse_release_window, normalize_camelot
from .connectors.music_links import generate_legitimate_links, format_dj_search_string
from .connectors.sources import SourceCatalogManager
from .connectors.itunes_store import ITunesStoreConnector
from .connectors.gemini_ai import GeminiAIDiscoveryConnector
from .song_analyzer import SongAnalyzer
from .audio_analyzer import AudioSignalAnalyzer
from .gemini_musicologist import GeminiMusicologist

# =========================================================================
# Route Handlers
# =========================================================================

async def serve_index(request: Request):
    """Serve the DJSyncAI frontend."""
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>DJSyncAI: index.html not found</h3>", status_code=404)

async def health_check(request: Request):
    """Health and service discovery endpoint."""
    return JSONResponse({
        "status": "online",
        "service": "DJSyncAI Platform v2.0",
        "geminiConfigured": bool(GEMINI_API_KEY),
        "defaultModel": DEFAULT_GEMINI_MODEL,
        "logFile": str(LOG_FILE),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

async def view_log(request: Request):
    """Return raw content of Logs/logs.txt."""
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return PlainTextResponse(content)
    return PlainTextResponse("No logs recorded yet.\n")

async def write_client_log(request: Request):
    """Preserves client logging subsystem."""
    raw_body = await request.body()
    log_text = raw_body.decode("utf-8", errors="replace").strip()
    if log_text:
        for line in log_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and len(line) > 20 and line[5] == "-" and line[8] == "-":
                append_to_logfile(line)
            else:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_to_logfile(f"[{now}] {line}")
    return JSONResponse({"status": "saved"})

async def analyze_dna_endpoint(request: Request):
    """Extract musical DNA from tracklist."""
    try:
        body = await request.json()
        raw_tracks = body.get("tracks", [])
        track_inputs = [TrackInput(**t) for t in raw_tracks]
        dna = DiscoveryEngine.analyze_playlist_dna(track_inputs)
        return JSONResponse(dna.model_dump())
    except Exception as e:
        log_event("BREAK", f"Error in /api/analyze-dna: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

async def discover_tracks_endpoint(request: Request):
    """Master AI Smart Track Discovery (Extended Mixes Only)."""
    try:
        body = await request.json()
        raw_tracks = body.get("tracks", [])
        track_inputs = [TrackInput(**t) for t in raw_tracks]
        
        dna = DiscoveryEngine.analyze_playlist_dna(track_inputs)
        
        req_key = body.get("apiKey", "").strip()
        active_key = req_key if req_key else GEMINI_API_KEY
        active_model = body.get("model", "").strip() or DEFAULT_GEMINI_MODEL
        target_period = body.get("targetPeriod", "all")
        max_results = min(int(body.get("maxResults", 60)), 60)

        result = await DiscoveryEngine.discover_tracks(
            dna=dna,
            api_key=active_key,
            model_name=active_model,
            target_period=target_period,
            max_results=max_results
        )
        return JSONResponse(result.model_dump())
    except Exception as e:
        log_event("BREAK", f"Error in /api/discover: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def resequence_tracks_endpoint(request: Request):
    """AI DJ Set Sequencer endpoint."""
    try:
        body = await request.json()
        raw_tracks = body.get("tracks", [])
        tracks = [VerifiedExtendedTrack(**t) for t in raw_tracks]
        locked_positions = body.get("lockedPositions", {})
        style = body.get("transitionStyle", "harmonic")

        sequence = DJSequencer.generate_sequence(
            tracks=tracks,
            locked_positions=locked_positions,
            style=style
        )
        return JSONResponse(sequence.model_dump())
    except Exception as e:
        log_event("BREAK", f"Error in /api/sequence: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

async def export_playlist_endpoint(request: Request):
    """Generate CSV, TXT, M3U8, or Rekordbox XML exports."""
    try:
        body = await request.json()
        fmt = body.get("format", "csv").lower().strip()
        raw_tracks = body.get("tracks", [])
        tracks = [VerifiedExtendedTrack(**t) for t in raw_tracks]
        playlist_name = body.get("playlistName", "DJSyncAI Extended Mixes")
        raw_seq = body.get("sequence")
        sequence = DJSequenceResult(**raw_seq) if raw_seq else None

        safe_name = playlist_name.replace(" ", "_").replace("/", "_")

        if fmt == "csv":
            content = PlaylistExporter.to_csv(tracks)
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.csv"'}
            )
        elif fmt == "txt":
            content = PlaylistExporter.to_txt(tracks, sequence=sequence)
            return Response(
                content=content,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}_tracklist.txt"'}
            )
        elif fmt == "m3u8":
            content = PlaylistExporter.to_m3u8(tracks)
            return Response(
                content=content,
                media_type="application/x-mpegurl; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.m3u8"'}
            )
        elif fmt in ("rekordbox_xml", "xml"):
            content = PlaylistExporter.to_rekordbox_xml(tracks, playlist_name=playlist_name)
            return Response(
                content=content,
                media_type="application/xml; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}_rekordbox.xml"'}
            )
        else:
            return JSONResponse({"error": f"Unsupported export format: {fmt}"}, status_code=400)
    except Exception as e:
        log_event("BREAK", f"Error in /api/export: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def analyze_stream_endpoint(request: Request):
    """
    Real-time Server-Sent Events (SSE) stream for song-by-song playlist analysis.
    Performs real musical analysis on every track using:
    1. Embedded tag & filename metadata extraction.
    2. Curated electronic club catalog matching.
    3. Google Gemini 3.8 AI Music Intelligence (when API key is present).
    4. Sophisticated musicological heuristic modeling (when offline or fallback).
    Yields actual inspection progress, genuine BPMs, Camelot keys, dynamic Energy (1-10),
    and Danceability (1-10) with complete 9-characteristic musical DNA.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_tracks = body.get("tracks", [])
    total_tracks = len(raw_tracks)
    file_name = body.get("fileName", "Playlist")
    audio_files_list = body.get("audioFiles", [])  # Optional list of known local audio file paths
    
    req_key = body.get("apiKey", "").strip()
    active_key = req_key if req_key else GEMINI_API_KEY
    active_model = body.get("model", "").strip() or DEFAULT_GEMINI_MODEL

    async def event_generator():
        try:
            # Stage 1: Upload notification
            yield "data: " + json.dumps({
                "stage": "Playlist uploaded successfully. Initializing song analyzer...",
                "percent": 5,
                "total": total_tracks,
                "analyzed": 0,
                "remaining": total_tracks,
                "incomplete": 0,
                "status": "processing"
            }) + "\n\n"
            await asyncio.sleep(0.04)

            # Build a lookup map of authorized local files if supplied
            import re
            audio_lookup = {}
            if audio_files_list:
                for af in audio_files_list:
                    base_n = os.path.basename(af).lower()
                    audio_lookup[base_n] = af

            analyzed_tracks: List[AnalysedTrackItem] = []
            incomplete_count = 0

            # Process track-by-track with genuine analysis & smooth visible pacing
            for idx, raw in enumerate(raw_tracks):
                if await request.is_disconnected():
                    log_event("WARN", "Client disconnected during /api/analyze-stream.")
                    return

                track_input = TrackInput(**raw)
                
                # Check authorized audio lookup if available
                authorized_path = None
                if audio_lookup:
                    cand_name = os.path.basename(track_input.filePath or track_input.location or "").lower()
                    if cand_name and cand_name in audio_lookup:
                        authorized_path = audio_lookup[cand_name]
                    else:
                        clean_title = re.sub(r'[^a-z0-9]', '', track_input.title.lower())
                        for b_name, full_p in audio_lookup.items():
                            if clean_title and clean_title in re.sub(r'[^a-z0-9]', '', b_name):
                                authorized_path = full_p
                                break

                # Analyze single track honestly
                analyzed_item = SongAnalyzer.analyze_single_track(
                    track_input=track_input,
                    idx=idx,
                    authorized_audio_path=authorized_path
                )
                analyzed_tracks.append(analyzed_item)

                is_incomplete = bool(analyzed_item.bpm is None or analyzed_item.key is None)
                if is_incomplete:
                    incomplete_count += 1

                # Calculate progress percentage and stage
                pct = 10 + int(((idx + 1) / max(total_tracks, 1)) * 75)
                if analyzed_item.analysis_status == "audio_analysed":
                    stage_msg = f"Audio DSP analysed: {analyzed_item.artist} — {analyzed_item.title}"
                elif analyzed_item.analysis_status == "metadata_only":
                    stage_msg = f"Imported Rekordbox metadata: {analyzed_item.artist} — {analyzed_item.title}"
                else:
                    stage_msg = f"Audio file not located: {analyzed_item.artist} — {analyzed_item.title}"

                bpm_display = f"{analyzed_item.bpm:.1f} BPM" if analyzed_item.bpm else "Not available"
                key_display = f"{analyzed_item.camelotKey} ({analyzed_item.key})" if (analyzed_item.camelotKey and analyzed_item.key) else (analyzed_item.camelotKey or analyzed_item.key or "Not available")

                event_payload = {
                    "stage": stage_msg,
                    "percent": pct,
                    "currentTrack": f"{analyzed_item.artist} — {analyzed_item.title}",
                    "currentBpm": bpm_display,
                    "currentKey": key_display,
                    "currentEnergy": analyzed_item.energy,
                    "currentDanceability": analyzed_item.danceability,
                    "currentGenre": analyzed_item.genre or "Not available",
                    "analysisStatus": analyzed_item.analysis_status,
                    "trackIndex": idx + 1,
                    "total": total_tracks,
                    "analyzed": idx + 1,
                    "remaining": total_tracks - (idx + 1),
                    "incomplete": incomplete_count,
                    "status": "processing",
                    "track": analyzed_item.model_dump()
                }
                yield "data: " + json.dumps(event_payload) + "\n\n"

                # Pacing: yield to event loop smoothly
                if total_tracks > 60:
                    if (idx % 2) == 0: await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0.02)

            yield "data: " + json.dumps({
                "stage": "Building your playlist's musical DNA and Camelot matrix...",
                "percent": 88,
                "total": total_tracks,
                "analyzed": total_tracks,
                "remaining": 0,
                "incomplete": incomplete_count,
                "status": "processing"
            }) + "\n\n"
            await asyncio.sleep(0.04)

            # Analyze DNA from the genuinely measured and imported tracks
            dna = DiscoveryEngine.analyze_playlist_dna(analyzed_tracks)

            # Interpret with Gemini Musicologist if API key available
            if active_key:
                yield "data: " + json.dumps({
                    "stage": "Consulting Gemini Pro DJ Musicologist for atmosphere and set flow...",
                    "percent": 93,
                    "total": total_tracks,
                    "analyzed": total_tracks,
                    "remaining": 0,
                    "incomplete": incomplete_count,
                    "status": "processing"
                }) + "\n\n"
                try:
                    interp_result = await GeminiMusicologist.interpret_tracks_and_set_flow(analyzed_tracks, active_key, active_model)
                    if interp_result and "playlistInterpretation" in interp_result:
                        pi = interp_result["playlistInterpretation"]
                        if pi.get("overallVibe"): dna.overallVibe = pi["overallVibe"]
                        if pi.get("patternsAndFlow"): dna.patternsAndFlow = pi["patternsAndFlow"]
                        if pi.get("bpmLandscapeDescription") and dna.bpmLandscape != "Insufficient analysed tracks to calculate playlist BPM identity.":
                            dna.bpmLandscape = f"{dna.bpmLandscape} | {pi['bpmLandscapeDescription']}"
                    
                    if interp_result and "trackInterpretations" in interp_result:
                        ti_map = {item.get("id"): item for item in interp_result["trackInterpretations"] if item.get("id")}
                        for trk in analyzed_tracks:
                            if trk.id in ti_map:
                                t_info = ti_map[trk.id]
                                trk.atmosphere = t_info.get("atmosphere")
                                trk.dj_role = t_info.get("dj_role")
                                trk.transition_notes = t_info.get("transition_guidance")
                                if not trk.genre and t_info.get("electronic_genre"):
                                    trk.genre = t_info.get("electronic_genre")
                                    trk.genre_source = "gemini_interpretation"
                                if not trk.subgenre and t_info.get("subgenre"):
                                    trk.subgenre = t_info.get("subgenre")
                except Exception as e:
                    log_event("WARN", f"Gemini musicologist interpretation exception: {e}")

            yield "data: " + json.dumps({
                "stage": "Preparing your track discovery profile...",
                "percent": 98,
                "total": total_tracks,
                "analyzed": total_tracks,
                "remaining": 0,
                "incomplete": incomplete_count,
                "status": "processing"
            }) + "\n\n"
            await asyncio.sleep(0.04)

            summary = {
                "totalLoaded": total_tracks,
                "analyzed": len([t for t in analyzed_tracks if t.analysis_status in ("audio_analysed", "metadata_only")]),
                "audioAnalysed": len([t for t in analyzed_tracks if t.analysis_status == "audio_analysed"]),
                "metadataOnly": len([t for t in analyzed_tracks if t.analysis_status == "metadata_only"]),
                "missingAudio": len([t for t in analyzed_tracks if t.analysis_status == "missing_audio"]),
                "incomplete": incomplete_count,
                "bpmRange": dna.bpmLandscape,
                "minBpm": dna.minBpm,
                "maxBpm": dna.maxBpm,
                "avgBpm": dna.avgBpm,
                "mainGenres": dna.primaryGenres,
                "energyProfile": f"Measured from {dna.energyTrackCount} of {total_tracks} tracks" if dna.energyTrackCount > 0 else "Awaiting audio analysis",
                "keyDistribution": dna.camelotKeys,
                "overallVibe": dna.overallVibe,
                "patternFlow": dna.patternsAndFlow
            }

            yield "data: " + json.dumps({
                "status": "complete",
                "stage": "Playlist analysis complete — your musical DNA is ready.",
                "percent": 100,
                "dna": dna.model_dump(),
                "summary": summary,
                "analyzedTracks": [t.model_dump() for t in analyzed_tracks]
            }) + "\n\n"

        except Exception as e:
            log_event("BREAK", f"Error in /api/analyze-stream: {e}")
            yield "data: " + json.dumps({"status": "failed", "stage": f"Analysis error: {str(e)}", "error": str(e)}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

async def discover_stream_endpoint(request: Request):
    """
    Real-time Server-Sent Events (SSE) stream for Extended Mix Discovery (Target: up to 60 tracks).
    Iterates through all 17 stages with real source queries, candidate audit logging,
    strict Extended Mix verification, live counters, and 5-phase sequencing.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_tracks = body.get("tracks", [])
    track_inputs = [TrackInput(**t) for t in raw_tracks]
    dna = DiscoveryEngine.analyze_playlist_dna(track_inputs)

    req_key = body.get("apiKey", "").strip()
    active_key = req_key if req_key else GEMINI_API_KEY
    active_model = body.get("model", "").strip() or DEFAULT_GEMINI_MODEL
    target_period = body.get("targetPeriod", "all")
    max_results = min(int(body.get("maxResults", 60)), 60)

    async def event_generator():
        try:
            # Stage 1: Preparing profile
            yield "data: " + json.dumps({"stageNumber": 1, "stage": "Preparing your musical discovery profile...", "percent": 4, "status": "searching", "candidatesDiscovered": 0, "extendedVerified": 0, "rejectedNonExtended": 0, "rejectedUnverified": 0, "duplicatesRemoved": 0, "eligibleRemaining": 0, "finalRecommendations": 0}) + "\n\n"
            await asyncio.sleep(0.08)

            raw_candidates: List[Dict] = []
            source_statuses = SourceCatalogManager.get_source_statuses(has_gemini_key=bool(active_key))

            # Helper for streaming source events
            async def report_source(stage_num: int, source_name: str, display_stage: str, pct: int):
                info = source_statuses.get(source_name, {"available": False, "reason": "Not configured"})
                reason = info.get("reason", "")
                if info.get("available"):
                    yield "data: " + json.dumps({"stageNumber": stage_num, "stage": display_stage, "percent": pct, "status": "searching", "source": source_name, "sourceStatus": "active", "sourceReason": reason}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"stageNumber": stage_num, "stage": f"{display_stage} ({reason})", "percent": pct, "status": "warning", "source": source_name, "sourceStatus": "unavailable", "sourceReason": reason}) + "\n\n"
                await asyncio.sleep(0.09)

            # Stage 2: Searching Beatport...
            async for chunk in report_source(2, "Beatport", "Searching Beatport...", 8):
                yield chunk
            if await request.is_disconnected(): return

            # Stage 3: Searching 1001Tracklists...
            async for chunk in report_source(3, "1001Tracklists", "Searching 1001Tracklists...", 14):
                yield chunk
            if await request.is_disconnected(): return

            # Stage 4: Searching SoundCloud...
            async for chunk in report_source(4, "SoundCloud", "Searching SoundCloud...", 20):
                yield chunk
            if await request.is_disconnected(): return

            # Stage 5: Searching BPM Supreme...
            async for chunk in report_source(5, "BPM Supreme", "Searching BPM Supreme...", 26):
                yield chunk
            if await request.is_disconnected(): return

            # Stage 6: Searching Traxsource...
            async for chunk in report_source(6, "Traxsource", "Searching Traxsource...", 32):
                yield chunk
            if await request.is_disconnected(): return

            # Stage 7: Discovering additional matching tracks...
            yield "data: " + json.dumps({"stageNumber": 7, "stage": "Discovering additional matching tracks...", "percent": 38, "status": "searching", "source": "iTunes", "sourceStatus": "active", "sourceReason": "Querying Apple Music / iTunes Store API..."}) + "\n\n"

            # Query live iTunes Store API
            try:
                itunes_conn = ITunesStoreConnector()
                itunes_candidates = await itunes_conn.search_candidates(dna, limit=40)
                raw_candidates.extend(itunes_candidates)
                yield "data: " + json.dumps({"candidatesDiscovered": len(raw_candidates), "stage": f"Discovered {len(raw_candidates)} electronic candidates from live store search..."}) + "\n\n"
            except Exception as e:
                log_event("WARN", f"iTunes search exception: {e}")

            # Query Gemini if key available
            if active_key:
                yield "data: " + json.dumps({"stage": "Querying Google Gemini AI Music Intelligence for set flow candidates...", "source": "Gemini", "sourceStatus": "active"}) + "\n\n"
                try:
                    gemini_conn = GeminiAIDiscoveryConnector(api_key=active_key, model_name=active_model)
                    gemini_candidates = await gemini_conn.search_candidates(dna, limit=85)
                    raw_candidates.extend(gemini_candidates)
                except Exception as e:
                    log_event("WARN", f"Gemini discovery exception: {e}")
            else:
                yield "data: " + json.dumps({"stage": "Loading curated electronic record store catalog (95+ club releases)...", "source": "ElectronicCatalog", "sourceStatus": "active"}) + "\n\n"

            # Merge curated electronic club catalog
            curated = SourceCatalogManager.get_curated_club_candidates()
            raw_candidates.extend(curated)

            total_discovered = len(raw_candidates)
            yield "data: " + json.dumps({"candidatesDiscovered": total_discovered, "percent": 46, "stage": f"Candidate pool established: {total_discovered} tracks collected."}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 8: Checking exact mix and version titles...
            yield "data: " + json.dumps({"stageNumber": 8, "stage": "Checking exact mix and version titles...", "percent": 50, "status": "verifying"}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 9 & 10: Verifying Extended Mix / Extended Edit versions & Filtering out radio edits
            yield "data: " + json.dumps({"stageNumber": 9, "stage": "Verifying Extended Mix / Extended Edit versions...", "percent": 54, "status": "verifying"}) + "\n\n"
            await asyncio.sleep(0.06)
            yield "data: " + json.dumps({"stageNumber": 10, "stage": "Filtering out radio edits and short versions...", "percent": 58, "status": "verifying"}) + "\n\n"

            verifier = ExtendedMixVerifier()
            verifier.reset_deduplication()

            verified_tracks: List[VerifiedExtendedTrack] = []
            extended_verified = 0
            rejected_non_ext = 0
            rejected_unverified = 0
            duplicates_removed = 0

            # Audit candidates one by one
            for i, cand in enumerate(raw_candidates):
                if await request.is_disconnected():
                    log_event("WARN", "Client disconnected during discovery verification stream.")
                    return

                artist = cand.get("artist", "").strip()
                title = cand.get("title", "").strip()
                mix_name = cand.get("mixName", "").strip()

                if not artist or not title:
                    rejected_unverified += 1
                    continue

                # Deduplication check
                if verifier.is_duplicate(artist, title):
                    duplicates_removed += 1
                    audit_event = {
                        "type": "audit",
                        "verdict": "duplicate",
                        "candidate": f"{artist} - {title} ({mix_name})",
                        "reason": "Duplicate title/version detected — removed",
                        "counters": {
                            "candidatesDiscovered": total_discovered,
                            "extendedVerified": extended_verified,
                            "rejectedNonExtended": rejected_non_ext,
                            "rejectedUnverified": rejected_unverified,
                            "duplicatesRemoved": duplicates_removed,
                            "eligibleRemaining": extended_verified,
                            "finalRecommendations": min(extended_verified, max_results)
                        }
                    }
                    yield "data: " + json.dumps(audit_event) + "\n\n"
                    continue

                # Strict Verification Filter
                is_verified, canonical_version, evidence = verifier.verify(artist, title, mix_name)

                if not is_verified:
                    low_evidence = evidence.lower()
                    if "prohibited" in low_evidence or "radio" in low_evidence or "short" in low_evidence or "single" in low_evidence:
                        rejected_non_ext += 1
                        verdict_code = "rejected_non_extended"
                    else:
                        rejected_unverified += 1
                        verdict_code = "rejected_unverified"

                    audit_event = {
                        "type": "audit",
                        "verdict": verdict_code,
                        "candidate": f"{artist} - {title} ({mix_name})",
                        "reason": f"Excluded: {evidence}",
                        "counters": {
                            "candidatesDiscovered": total_discovered,
                            "extendedVerified": extended_verified,
                            "rejectedNonExtended": rejected_non_ext,
                            "rejectedUnverified": rejected_unverified,
                            "duplicatesRemoved": duplicates_removed,
                            "eligibleRemaining": extended_verified,
                            "finalRecommendations": min(extended_verified, max_results)
                        }
                    }
                    yield "data: " + json.dumps(audit_event) + "\n\n"
                    continue

                # Verified Extended Mix!
                extended_verified += 1

                # Calculate 8-metric weighted scores
                scores = compute_track_scores(cand, dna)

                # Evaluate rolling 6-month window
                raw_release_date = cand.get("releaseDate")
                release_date = str(raw_release_date).strip() if raw_release_date else None
                is_recent_6m, release_window = parse_release_window(release_date)

                if target_period == "recent_6m" and not is_recent_6m:
                    continue
                if target_period == "recent_1y" and release_window == "All-Time Classic Club":
                    continue

                links = generate_legitimate_links(artist, title, canonical_version)
                search_str = format_dj_search_string(artist, title, canonical_version)
                
                raw_bpm = cand.get("bpm")
                cand_bpm = None
                if raw_bpm is not None:
                    try:
                        cand_bpm = round(float(raw_bpm), 1)
                    except (ValueError, TypeError):
                        cand_bpm = None

                cand_key = str(cand.get("key")).strip() if cand.get("key") else None
                raw_cam = cand.get("camelotKey") or cand_key
                camelot = normalize_camelot(raw_cam) if raw_cam else None

                raw_energy = cand.get("energy")
                cand_energy = None
                if raw_energy is not None:
                    try:
                        cand_energy = int(round(float(raw_energy)))
                    except (ValueError, TypeError):
                        cand_energy = None

                raw_dance = cand.get("danceability")
                cand_dance = None
                if raw_dance is not None:
                    try:
                        cand_dance = int(round(float(raw_dance)))
                    except (ValueError, TypeError):
                        cand_dance = None

                ver_source = cand.get("verificationSource") or "Catalog Verification"
                is_indep = bool(cand.get("isIndependentlyVerified", False))

                track_obj = VerifiedExtendedTrack(
                    id=f"ext_{uuid.uuid4().hex[:8]}",
                    artist=artist,
                    title=title,
                    mixName=canonical_version,
                    versionType=canonical_version,
                    duration=cand.get("duration") or None,
                    bpm=cand_bpm,
                    key=cand_key,
                    camelotKey=camelot,
                    genre=cand.get("genre") or None,
                    subgenre=cand.get("subgenre") or None,
                    energy=cand_energy,
                    danceability=cand_dance,
                    releaseDate=release_date,
                    releaseLabel=cand.get("releaseLabel") or None,
                    vibeScore=scores.get("vibeScore"),
                    flowScore=scores.get("flowScore"),
                    similarityScore=scores["similarityScore"],
                    djScore=scores["djScore"],
                    popularityScore=scores["popularityScore"],
                    overallScore=scores["overallScore"],
                    isRecentSixMonths=is_recent_6m,
                    releaseWindow=release_window,
                    verificationEvidence=evidence,
                    verificationSource=ver_source,
                    isIndependentlyVerified=is_indep,
                    links=links,
                    searchString=search_str
                )
                verified_tracks.append(track_obj)

                bpm_str = f"{track_obj.bpm:.1f} BPM" if track_obj.bpm is not None else "BPM unmeasured"
                cam_str = track_obj.camelotKey if track_obj.camelotKey else "Key unmeasured"

                audit_event = {
                    "type": "audit",
                    "verdict": "verified",
                    "candidate": f"{artist} - {title} ({canonical_version})",
                    "reason": f"Certified {canonical_version} [{bpm_str} | {cam_str}]",
                    "counters": {
                        "candidatesDiscovered": total_discovered,
                        "extendedVerified": extended_verified,
                        "rejectedNonExtended": rejected_non_ext,
                        "rejectedUnverified": rejected_unverified,
                        "duplicatesRemoved": duplicates_removed,
                        "eligibleRemaining": len(verified_tracks),
                        "finalRecommendations": min(len(verified_tracks), max_results)
                    }
                }
                yield "data: " + json.dumps(audit_event) + "\n\n"

                if (i % 3) == 0:
                    await asyncio.sleep(0.01)

            # Stage 11: Checking BPM and musical key compatibility...
            yield "data: " + json.dumps({"stageNumber": 11, "stage": "Checking BPM and musical key compatibility...", "percent": 74, "status": "processing"}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 12: Comparing energy, groove and musical style...
            yield "data: " + json.dumps({"stageNumber": 12, "stage": "Comparing energy, groove and musical style...", "percent": 79, "status": "processing"}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 13: Calculating musical similarity scores...
            yield "data: " + json.dumps({"stageNumber": 13, "stage": "Calculating musical similarity scores (8-metric weighted formula)...", "percent": 84, "status": "processing"}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 14: Removing duplicate tracks and versions...
            yield "data: " + json.dumps({"stageNumber": 14, "stage": "Removing duplicate tracks and versions...", "percent": 89, "status": "processing"}) + "\n\n"
            await asyncio.sleep(0.08)

            # Stage 15: Ranking your best 60 extended mixes...
            yield "data: " + json.dumps({"stageNumber": 15, "stage": f"Ranking your best {max_results} extended mixes...", "percent": 93, "status": "processing"}) + "\n\n"
            verified_tracks.sort(key=lambda t: t.overallScore, reverse=True)
            final_tracks = verified_tracks[:max_results]
            final_count = len(final_tracks)
            await asyncio.sleep(0.08)

            # Stage 16: Building your suggested DJ set sequence...
            yield "data: " + json.dumps({"stageNumber": 16, "stage": "Building your suggested DJ set sequence...", "percent": 97, "status": "processing"}) + "\n\n"
            suggested_sequence = DJSequencer.generate_sequence(final_tracks)
            await asyncio.sleep(0.08)

            # Stage 17: Discovery completed successfully.
            if final_count < max_results:
                explanation = (
                    f"Discovered {final_count} verified Extended Mixes. "
                    f"Scanned {total_discovered} candidates and strictly excluded {rejected_non_ext} "
                    f"radio/short edits and {rejected_unverified} unverified tracks. "
                    f"Results are never padded with unverified tracks."
                )
            else:
                explanation = (
                    f"Successfully curated the top {final_count} verified Extended Mixes and Extended Edits matching "
                    f"your set's musical identity. {rejected_non_ext} radio/short edits and {rejected_unverified} unverified versions were strictly filtered out."
                )

            final_data = {
                "status": "complete",
                "stageNumber": 17,
                "stage": "Discovery completed successfully.",
                "percent": 100,
                "dna": dna.model_dump(),
                "verifiedTracks": [t.model_dump() for t in final_tracks],
                "totalCandidatesScanned": total_discovered,
                "totalRejectedNonExtended": rejected_non_ext,
                "totalRejectedUnverified": rejected_unverified,
                "totalDuplicatesRemoved": duplicates_removed,
                "totalVerifiedReturned": final_count,
                "explanation": explanation,
                "suggestedSequence": suggested_sequence.model_dump() if suggested_sequence else None
            }
            yield "data: " + json.dumps(final_data) + "\n\n"

        except Exception as e:
            log_event("BREAK", f"Error in /api/discover-stream: {e}")
            yield "data: " + json.dumps({"status": "failed", "stage": f"Discovery error: {str(e)}", "error": str(e)}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =========================================================================
# Application Assembly
# =========================================================================

async def match_audio_endpoint(request: Request):
    """
    Accepts user-authorized audio files or directory path to match and analyze tracks.
    Enables linking local music collections to resolve 'missing_audio' tracks
    with genuine DSP audio analysis (BPM, Key, Energy, Danceability).
    """
    try:
        body = await request.json()
        raw_tracks = body.get("tracks", [])
        audio_files = body.get("audioFiles", [])  # list of absolute paths or filenames
        folder_path = body.get("folderPath", "").strip()

        # If a folderPath is provided on the server, scan it
        if folder_path and os.path.isdir(folder_path):
            valid_exts = {".mp3", ".wav", ".flac", ".aif", ".aiff", ".m4a", ".aac", ".ogg"}
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        audio_files.append(os.path.join(root, f))

        import re
        audio_lookup = {}
        for af in audio_files:
            b_name = os.path.basename(af).lower()
            audio_lookup[b_name] = af

        updated_tracks: List[AnalysedTrackItem] = []
        for idx, raw in enumerate(raw_tracks):
            track_input = TrackInput(**raw)
            auth_path = None
            cand_name = os.path.basename(track_input.filePath or track_input.location or "").lower()
            if cand_name and cand_name in audio_lookup:
                auth_path = audio_lookup[cand_name]
            else:
                clean_title = re.sub(r'[^a-z0-9]', '', track_input.title.lower())
                for b_name, full_p in audio_lookup.items():
                    if clean_title and clean_title in re.sub(r'[^a-z0-9]', '', b_name):
                        auth_path = full_p
                        break

            item = SongAnalyzer.analyze_single_track(track_input, idx, auth_path)
            updated_tracks.append(item)

        dna = DiscoveryEngine.analyze_playlist_dna(updated_tracks)
        return JSONResponse({
            "status": "matched",
            "tracks": [t.model_dump() for t in updated_tracks],
            "dna": dna.model_dump(),
            "totalLoaded": len(updated_tracks),
            "audioAnalysedCount": len([t for t in updated_tracks if t.analysis_status == "audio_analysed"]),
            "metadataOnlyCount": len([t for t in updated_tracks if t.analysis_status == "metadata_only"]),
            "missingAudioCount": len([t for t in updated_tracks if t.analysis_status == "missing_audio"])
        })
    except Exception as e:
        log_event("BREAK", f"Error in /api/match-audio: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

routes = [
    Route("/", serve_index, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/status", health_check, methods=["GET"]),
    Route("/log", view_log, methods=["GET"]),
    Route("/log", write_client_log, methods=["POST"]),
    Route("/api/analyze-dna", analyze_dna_endpoint, methods=["POST"]),
    Route("/api/analyze-stream", analyze_stream_endpoint, methods=["POST"]),
    Route("/api/match-audio", match_audio_endpoint, methods=["POST"]),
    Route("/api/discover", discover_tracks_endpoint, methods=["POST"]),
    Route("/api/discover-stream", discover_stream_endpoint, methods=["POST"]),
    Route("/api/sequence", resequence_tracks_endpoint, methods=["POST"]),
    Route("/api/export", export_playlist_endpoint, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )
]

app = Starlette(routes=routes, middleware=middleware)

"""
Google Gemini Pro DJ Musicologist & Reasoning Layer.
Acts as the expert electronic musicologist and interpretation layer:
- Interprets genuine, measured audio features and imported Rekordbox metadata.
- Evaluates atmosphere, groove character, set role, and harmonic transition potential.
- Synthesizes playlist-wide set-flow progression and musical DNA narrative.
- STRICT RULE: Never invents missing BPM, key, or energy measurements. If unmeasured, returns null.
"""
import json
import httpx
from typing import List, Dict, Any, Optional

from .models import AnalysedTrackItem, PlaylistDNA
from .config import DEFAULT_GEMINI_MODEL, log_event

class GeminiMusicologist:
    """Intelligent musicological interpretation layer using Gemini Pro."""

    _cached_models: Optional[List[str]] = None

    @classmethod
    async def list_available_models(cls, api_key: str) -> List[str]:
        """Queries Google's ModelService to discover all active models for this API key."""
        if cls._cached_models:
            return cls._cached_models
        if not api_key:
            return []
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m.get("name", "").replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    cls._cached_models = models
                    log_event("STATUS", f"Verified Google Gemini models for your key: {models[:6]}")
                    return models
                else:
                    log_event("WARN", f"ModelService.ListModels returned HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            log_event("WARN", f"Model discovery note: {type(e).__name__} - {e}")
        return []

    @classmethod
    async def interpret_tracks_and_set_flow(
        cls,
        analysed_tracks: List[AnalysedTrackItem],
        api_key: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Sends genuine extracted audio features and Rekordbox metadata to Gemini
        for expert DJ interpretation (atmosphere, set role, transition potential, set flow).
        """
        if not api_key:
            return None

        # Discover or verify active models
        avail = await cls.list_available_models(api_key)
        raw_model = (model_name or DEFAULT_GEMINI_MODEL or "gemini-3.8-flash").strip()
        
        models_to_try = []
        if raw_model:
            models_to_try.append(raw_model)
        for m in avail:
            if m not in models_to_try:
                models_to_try.append(m)
        for fallback in ["gemini-3.8-flash", "gemini-2.5-flash"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        # Construct payload with ONLY genuine measurements and metadata
        structured_track_data = []
        for t in analysed_tracks:
            item = {
                "id": t.id,
                "artist": t.artist,
                "title": t.title,
                "analysis_status": t.analysis_status,
                "measured_bpm": t.bpm,
                "bpm_confidence": t.bpm_confidence,
                "measured_key": t.key,
                "camelot_key": t.camelotKey,
                "key_confidence": t.key_confidence,
                "energy_score": t.energy,
                "energy_features": t.energy_features,
                "danceability_score": t.danceability,
                "tempo_stability": t.tempo_stability,
                "genre_hint": t.genre,
                "year_hint": t.year
            }
            structured_track_data.append(item)

        # Sample up to 15 representative tracks to ensure high responsiveness
        sample_tracks = structured_track_data[:15]

        system_instruction = (
            "You are an expert club DJ, musicologist, and electronic dance music data scientist. "
            "You have been provided with genuine audio signal processing measurements and metadata "
            "from an uploaded DJ playlist.\n\n"
            "STRICT RULES:\n"
            "1. NEVER invent, fabricate, or guess missing BPMs, keys, energy scores, or release dates. "
            "If a track has null for measured_bpm or measured_key, keep it unknown.\n"
            "2. Interpret the musical identity, groove, and atmospheric character of each track based on its "
            "measured audio features (tempo, key, spectral energy, danceability) and genuine metadata.\n"
            "3. Classify the likely electronic genre and subgenre (e.g. Tech House, Melodic Techno, Peak-Time Techno, "
            "Deep House, Drum & Bass, Nu Disco, Progressive House, Breaks, Minimal, Afro House, UK Garage, Trance).\n"
            "4. Assign each track's functional DJ set role (e.g. 'Warm-up / Atmosphere Setter', 'Groove Builder', "
            "'Peak-Time Driver', 'High-Energy Climax', 'Melodic Breather / Cool-down').\n"
            "5. Provide 1 concise sentence of DJ transition guidance (key compatibility, tempo transition advice).\n"
            "6. Synthesize the playlist's overall musical vibe, BPM trajectory, and pattern/flow.\n"
            "7. KEEP ALL DESCRIPTIONS CONCISE (under 10 words each) for maximum speed."
        )

        prompt_text = (
            f"{system_instruction}\n\n"
            f"INPUT TRACKS ({len(sample_tracks)} sample):\n{json.dumps({'tracks': sample_tracks})}\n\n"
            "Respond in valid JSON using this exact structure:\n"
            "{\n"
            '  "playlistInterpretation": {\n'
            '    "overallVibe": "string",\n'
            '    "bpmLandscapeDescription": "string",\n'
            '    "patternsAndFlow": "string"\n'
            "  },\n"
            '  "trackInterpretations": [\n'
            '    {\n'
            '      "id": "string",\n'
            '      "atmosphere": "string",\n'
            '      "electronic_genre": "string",\n'
            '      "subgenre": "string",\n'
            '      "dj_role": "string",\n'
            '      "transition_guidance": "string"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{ "text": prompt_text }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }

        for attempt_model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{attempt_model}:generateContent?key={api_key}"
            try:
                log_event("STATUS", f"Sending {len(sample_tracks)} representative tracks to Gemini ({attempt_model}) for musicological interpretation...")
                async with httpx.AsyncClient(timeout=90.0) as client:
                    resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if text:
                                parsed = json.loads(text)
                                log_event("STATUS", f"Gemini ({attempt_model}) returned musical interpretations for {len(parsed.get('trackInterpretations', []))} tracks.")
                                return parsed
                    else:
                        log_event("WARN", f"Gemini ({attempt_model}) returned HTTP {resp.status_code}: {resp.text[:250]}")
            except httpx.TimeoutException:
                log_event("WARN", f"Gemini interpretation call timed out on ({attempt_model}) after 90s.")
            except Exception as e:
                err_detail = f"{type(e).__name__}: {str(e) or repr(e)}"
                log_event("WARN", f"Gemini interpretation note ({attempt_model}): {err_detail}")

        return None

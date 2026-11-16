import json
import httpx
from typing import List, Dict, Any, Optional
from .base import BaseConnector
from ..models import PlaylistDNA
from ..config import DEFAULT_GEMINI_MODEL, log_event
from ..scoring import normalize_camelot

class GeminiAIDiscoveryConnector(BaseConnector):
    """
    Intelligent Electronic Music Discovery Connector powered by Google Gemini API.
    Curates authentic, verified Extended Mixes from electronic dance music labels.
    """

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        raw_model = (model_name or DEFAULT_GEMINI_MODEL or "gemini-3.8-flash").strip()
        model_map = {
            "gemini-flash": "gemini-3.8-flash",
            "gemini-pro": "gemini-3.8-pro",
        }
        self.model = model_map.get(raw_model, raw_model)

    @property
    def source_name(self) -> str:
        return f"Gemini Music Intelligence ({self.model})"

    async def search_candidates(self, dna: PlaylistDNA, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.api_key:
            log_event("WARN", "Gemini API key is not provided. Skipping Gemini candidate search.")
            return []

        from ..gemini_musicologist import GeminiMusicologist
        avail = await GeminiMusicologist.list_available_models(self.api_key)
        active_model = self.model
        if active_model in avail:
            pass
        elif avail:
            flash_c = next((m for m in avail if "flash" in m.lower()), avail[0])
            active_model = flash_c

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={self.api_key}"

        system_instruction = (
            "You are a world-class electronic music researcher, Beatport/Traxsource specialist, "
            "and professional club DJ catalog archivist. "
            "Your objective is to discover REAL, OFFICIALLY RELEASED electronic dance music tracks that "
            "fit the exact musical DNA, vibe, tempo, and Camelot keys of a DJ set.\n\n"
            "MANDATORY REQUIREMENT - EXTENDED MIXES ONLY:\n"
            "1. Every single recommendation MUST be an officially identified Extended Mix, Extended Edit, "
            "Extended Club Mix, or Extended Remix released on respected dance labels (e.g., Defected, "
            "Toolroom, Drumcode, Afterlife, Anjunadeep, Armada, Diynamic, Solid Grooves, Keinemusik, "
            "Innervisions, Crosstown Rebels, Spinnin' Deep, Repopulate Mars, etc.).\n"
            "2. DO NOT return Radio Edits, Short Edits, Single Versions, or unverified Original Mixes.\n"
            "3. DO NOT invent fake tracks or artists. Return only real, cataloged releases.\n"
            "4. Provide realistic BPM, Camelot Key (e.g. '8A', '11B'), authentic release date "
            "(include both hot club releases from the last 6 months and timeless club staples), "
            "and record label.\n"
            "5. The mix name MUST explicitly contain 'Extended Mix', 'Extended Edit', or 'Extended Club Mix'."
        )

        user_prompt = {
            "task": "Discover verified Extended Mixes matching this DJ Playlist DNA",
            "dna": {
                "overallVibe": dna.overallVibe,
                "bpmLandscape": dna.bpmLandscape,
                "averageBpm": dna.avgBpm,
                "primaryGenres": dna.primaryGenres,
                "patternsAndFlow": dna.patternsAndFlow,
                "camelotKeys": dna.camelotKeys
            },
            "candidateTarget": min(limit, 25)
        }

        prompt_text = (
            f"{system_instruction}\n\n"
            f"PLAYLIST DNA & SEARCH PROFILE:\n{json.dumps(user_prompt)}\n\n"
            "Return valid JSON strictly matching this schema:\n"
            "{\n"
            '  "discoveryAnalysis": { "curationVibe": "string", "energyProfile": "string" },\n'
            '  "candidates": [\n'
            '    {\n'
            '      "artist": "string",\n'
            '      "title": "string",\n'
            '      "mixName": "string (MUST contain Extended Mix / Extended Edit / Extended Remix)",\n'
            '      "versionType": "Extended Mix",\n'
            '      "bpm": 126.0,\n'
            '      "key": "Am",\n'
            '      "camelotKey": "8A",\n'
            '      "genre": "Tech House",\n'
            '      "subgenre": "Latin Tech",\n'
            '      "energy": 7,\n'
            '      "danceability": 8,\n'
            '      "releaseDate": "YYYY-MM-DD",\n'
            '      "releaseLabel": "string",\n'
            '      "duration": "06:20",\n'
            '      "verificationSource": "Beatport Top 100",\n'
            '      "vibeScore": 92,\n'
            '      "flowScore": 90,\n'
            '      "popularityScore": 85\n'
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
                "temperature": 0.3
            }
        }

        candidates = []
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                log_event("STATUS", f"Querying Gemini API ({active_model}) for verified Extended Mix candidates...")
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 200:
                    result = resp.json()
                    candidates_data = result.get("candidates", [])
                    if candidates_data:
                        text_payload = candidates_data[0]["content"]["parts"][0]["text"]
                        parsed = json.loads(text_payload)
                        items = parsed.get("candidates", [])
                        log_event("STATUS", f"Gemini returned {len(items)} initial raw candidate recommendations.")
                        for itm in items:
                            raw_bpm = itm.get("bpm")
                            parsed_bpm = None
                            if raw_bpm is not None:
                                try:
                                    parsed_bpm = round(float(raw_bpm), 1)
                                except (ValueError, TypeError):
                                    parsed_bpm = None

                            raw_energy = itm.get("energy")
                            parsed_energy = None
                            if raw_energy is not None:
                                try:
                                    parsed_energy = max(1, min(10, int(raw_energy)))
                                except (ValueError, TypeError):
                                    parsed_energy = None

                            raw_dance = itm.get("danceability")
                            parsed_dance = None
                            if raw_dance is not None:
                                try:
                                    parsed_dance = max(1, min(10, int(raw_dance)))
                                except (ValueError, TypeError):
                                    parsed_dance = None

                            raw_key = itm.get("key")
                            parsed_key = str(raw_key).strip() if raw_key else None
                            raw_camelot = itm.get("camelotKey") or parsed_key
                            parsed_camelot = normalize_camelot(raw_camelot) if raw_camelot else None

                            candidates.append({
                                "artist": itm.get("artist", "").strip(),
                                "title": itm.get("title", "").strip(),
                                "mixName": itm.get("mixName", "").strip(),
                                "versionType": itm.get("versionType", "Extended Mix").strip(),
                                "bpm": parsed_bpm,
                                "key": parsed_key,
                                "camelotKey": parsed_camelot,
                                "genre": itm.get("genre") or "Electronic",
                                "subgenre": itm.get("subgenre") or None,
                                "energy": parsed_energy,
                                "danceability": parsed_dance,
                                "releaseDate": str(itm.get("releaseDate")).strip() if itm.get("releaseDate") else None,
                                "releaseLabel": itm.get("releaseLabel") or None,
                                "duration": itm.get("duration") or None,
                                "verificationSource": "Gemini AI Discovery Suggestion",
                                "isIndependentlyVerified": False,
                                "vibeScore": itm.get("vibeScore"),
                                "flowScore": itm.get("flowScore"),
                                "popularityScore": itm.get("popularityScore")
                            })
                else:
                    log_event("BREAK", f"Gemini candidate search failed with HTTP {resp.status_code}: {resp.text[:300]}")
            except httpx.TimeoutException:
                log_event("WARN", f"Gemini candidate search timed out after 90s. Continuing with certified club catalogue.")
            except Exception as e:
                log_event("BREAK", f"Exception during Gemini candidate search: {type(e).__name__} - {str(e) or repr(e)}")

        return candidates

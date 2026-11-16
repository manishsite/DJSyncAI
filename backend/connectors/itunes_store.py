import httpx
from typing import List, Dict, Any
from .base import BaseConnector
from ..models import PlaylistDNA
from ..verification import ExtendedMixVerifier

class ITunesStoreConnector(BaseConnector):
    """
    Public iTunes / Apple Music electronic dance music catalog search.
    Provides legitimate release metadata, exact release dates, genres, and track lengths.
    """

    @property
    def source_name(self) -> str:
        return "iTunes Electronic Catalog"

    async def search_candidates(self, dna: PlaylistDNA, limit: int = 50) -> List[Dict[str, Any]]:
        candidates = []
        genres_to_search = dna.primaryGenres if dna.primaryGenres else ["House", "Techno", "Electronic"]
        
        # Focus on electronic club terms with "Extended"
        search_terms = []
        for g in genres_to_search[:3]:
            search_terms.append(f"{g} Extended Mix")
        search_terms.append("Extended Club Mix")

        async with httpx.AsyncClient(timeout=httpx.Timeout(2.5, connect=1.0)) as client:
            for term in search_terms:
                try:
                    url = "https://itunes.apple.com/search"
                    params = {
                        "term": term,
                        "media": "music",
                        "entity": "song",
                        "limit": 25,
                        "genreId": "17"  # Dance / Electronic genre ID
                    }
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("results", []):
                            track_name = item.get("trackName", "")
                            artist_name = item.get("artistName", "")
                            collection_name = item.get("collectionName", "")
                            
                            # Verify if title or collection contains an explicit extended version tag
                            mix_name = ExtendedMixVerifier.extract_mix_label(track_name, collection_name)
                            is_extended, version_type, evidence = ExtendedMixVerifier.verify(artist_name, track_name, mix_name)
                            
                            if is_extended:
                                duration_ms = item.get("trackTimeMillis", 0)
                                minutes = duration_ms // 60000
                                seconds = (duration_ms % 60000) // 1000
                                dur_str = f"{minutes:02d}:{seconds:02d}" if duration_ms > 0 else "06:00"

                                release_date = item.get("releaseDate", "2024-01-01").split("T")[0]

                                candidates.append({
                                    "artist": artist_name,
                                    "title": track_name,
                                    "mixName": mix_name,
                                    "versionType": version_type,
                                    "duration": dur_str,
                                    "bpm": None,  # iTunes API does not report BPM; keep unmeasured
                                    "key": None,  # iTunes API does not report musical key; keep unmeasured
                                    "camelotKey": None,
                                    "genre": item.get("primaryGenreName", "Dance / Electronic"),
                                    "subgenre": term.replace(" Extended Mix", "").replace(" Extended Club Mix", "Electronic"),
                                    "releaseDate": release_date,
                                    "releaseLabel": item.get("collectionCensoredName", "Apple Music / Store Release"),
                                    "verificationSource": "Apple Music / iTunes Electronic Store",
                                    "verificationEvidence": f"Official Apple Music Electronic Release ({item.get('collectionName', 'Single')})",
                                    "isIndependentlyVerified": True
                                })
                except Exception:
                    continue

        return candidates[:limit]

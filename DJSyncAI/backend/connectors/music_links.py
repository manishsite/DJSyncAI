import urllib.parse
from typing import Dict

def generate_legitimate_links(artist: str, title: str, mix_name: str) -> Dict[str, str]:
    """
    Generate authentic search links for legitimate electronic music storefronts & platforms.
    Does not scrape or automate transfers; provides permitted discovery URLs.
    """
    # Clean query for search
    full_query = f"{artist} {title} {mix_name}".strip()
    encoded_full = urllib.parse.quote_plus(full_query)
    
    track_query = f"{artist} {title}".strip()
    encoded_track = urllib.parse.quote_plus(track_query)

    return {
        "beatport": f"https://www.beatport.com/search?q={encoded_full}",
        "traxsource": f"https://www.traxsource.com/search?term={encoded_full}",
        "tracklists1001": f"https://www.1001tracklists.com/search/result.php?search={encoded_track}",
        "soundcloud": f"https://soundcloud.com/search/sounds?q={encoded_full}"
    }

def format_dj_search_string(artist: str, title: str, mix_name: str) -> str:
    """Format plain string for manual DJ acquisition / crate workflow."""
    return f"{artist} - {title} ({mix_name})"

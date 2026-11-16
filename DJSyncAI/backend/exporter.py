import csv
import io
import html
from typing import List
from .models import VerifiedExtendedTrack, DJSequenceResult

class PlaylistExporter:
    """
    Export generator for DJSyncAI verified Extended Mixes.
    Supports CSV, TXT, M3U8, and Rekordbox XML.
    """

    @classmethod
    def to_csv(cls, tracks: List[VerifiedExtendedTrack]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Position", "Artist", "Title", "Mix Name", "Version Type",
            "BPM", "Musical Key", "Camelot Key", "Genre", "Subgenre",
            "Energy (1-10)", "Danceability (1-10)", "Release Date", "Release Label",
            "Release Period", "Similarity Score", "DJ Score", "Overall Score",
            "Verification Source", "Beatport Search Link", "Traxsource Search Link"
        ])

        for idx, t in enumerate(tracks, start=1):
            writer.writerow([
                idx,
                t.artist,
                t.title,
                t.mixName,
                t.versionType,
                f"{t.bpm:.1f}" if t.bpm is not None else "",
                t.key or "",
                t.camelotKey or "",
                t.genre or "",
                t.subgenre or "",
                t.energy if t.energy is not None else "",
                t.danceability if t.danceability is not None else "",
                t.releaseDate or "",
                t.releaseLabel or "",
                t.releaseWindow or "",
                f"{t.similarityScore}%" if t.similarityScore is not None else "",
                f"{t.djScore}%" if t.djScore is not None else "",
                f"{t.overallScore}%" if t.overallScore is not None else "",
                t.verificationSource,
                t.links.get("beatport", "") if t.links else "",
                t.links.get("traxsource", "") if t.links else ""
            ])

        return output.getvalue()

    @classmethod
    def to_txt(cls, tracks: List[VerifiedExtendedTrack], sequence: DJSequenceResult = None) -> str:
        lines = []
        lines.append("================================================================================")
        lines.append(f" DJSyncAI: AI Smart Track Discovery — {len(tracks)} Verified Extended Mixes")
        lines.append(" Strict Filter: Official Extended Mixes & Extended Edits Only")
        lines.append(" Generated for Professional DJ Performance & Rekordbox Sets")
        lines.append("================================================================================\n")

        if sequence and sequence.orderedTracks:
            current_phase = None
            for item in sequence.orderedTracks:
                if item.phase != current_phase:
                    current_phase = item.phase
                    lines.append(f"\n--- [{current_phase.upper()}] ---")
                
                t = item.track
                bpm_str = f"{t.bpm:.1f} BPM" if t.bpm is not None else "BPM unmeasured"
                cam_str = t.camelotKey if t.camelotKey else "Key unmeasured"
                energy_str = f"Energy: {t.energy}/10" if t.energy is not None else "Energy unmeasured"
                lines.append(
                    f"{item.position:02d}. {t.artist} - {t.title} ({t.mixName}) "
                    f"[{bpm_str} | {cam_str} | {energy_str}] "
                    f"-> {item.transitionQuality} ({item.transitionReason})"
                )
        else:
            for idx, t in enumerate(tracks, start=1):
                bpm_str = f"{t.bpm:.1f} BPM" if t.bpm is not None else "BPM unmeasured"
                cam_str = t.camelotKey if t.camelotKey else "Key unmeasured"
                genre_str = t.genre or "Electronic"
                date_str = t.releaseDate or "Release date unknown"
                lines.append(
                    f"{idx:02d}. {t.artist} - {t.title} ({t.mixName}) "
                    f"[{bpm_str} | {cam_str} | {genre_str} | {date_str}]"
                )

        lines.append("\n================================================================================")
        lines.append("Note: Audio files must be legally acquired/downloaded from authorized stores.")
        lines.append("Links to Beatport, Traxsource, and SoundCloud are included in the CSV manifest.")
        lines.append("================================================================================")

        return "\n".join(lines)

    @classmethod
    def to_m3u8(cls, tracks: List[VerifiedExtendedTrack]) -> str:
        lines = ["#EXTM3U", "#EXTENC:UTF-8", "#PLAYLIST:DJSyncAI Verified Extended Mixes"]
        
        for t in tracks:
            duration_sec = 360  # Default 6-minute club estimate if exact duration not provided
            lines.append(f"#EXTINF:{duration_sec},{t.artist} - {t.title} ({t.mixName})")
            if t.genre:
                lines.append(f"#EXTGENRE:{t.genre}")
            if t.bpm is not None:
                lines.append(f"#EXTBPM:{t.bpm:.1f}")
            if t.camelotKey:
                lines.append(f"#EXTKEY:{t.camelotKey}")
            # Placeholder comment for path resolution
            search_query = f"{t.artist} - {t.title} ({t.mixName})".replace(" ", "+")
            lines.append(f"https://www.beatport.com/search?q={search_query}")

        return "\n".join(lines)

    @classmethod
    def to_rekordbox_xml(cls, tracks: List[VerifiedExtendedTrack], playlist_name: str = "DJSyncAI Extended Mixes") -> str:
        """
        Produce authentic Pioneer Rekordbox XML with <COLLECTION> and <PLAYLISTS><NODE>.
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<DJ_PLAYLISTS Version="1.0.0">',
            '  <PRODUCT Name="rekordbox" Version="7.0.0" Company="AlphaTheta"/>',
            f'  <COLLECTION Entries="{len(tracks)}">'
        ]

        for idx, t in enumerate(tracks, start=1):
            esc_title = html.escape(f"{t.title} ({t.mixName})")
            esc_artist = html.escape(t.artist)
            esc_genre = html.escape(t.genre or "")
            esc_key = html.escape(t.camelotKey or t.key or "")
            esc_label = html.escape(t.releaseLabel or "")
            year = (t.releaseDate.split("-")[0] if "-" in t.releaseDate else t.releaseDate) if t.releaseDate else ""
            bpm_attr = f'AverageBpm="{t.bpm:.2f}" ' if t.bpm is not None else ''
            score_str = f" | Score: {t.overallScore}%" if t.overallScore is not None else ""

            lines.append(
                f'    <TRACK TrackID="{idx}" Name="{esc_title}" Artist="{esc_artist}" '
                f'Genre="{esc_genre}" {bpm_attr}Tonality="{esc_key}" '
                f'Label="{esc_label}" Year="{year}" Comments="DJSyncAI Extended Mix{score_str}"/>'
            )

        lines.append('  </COLLECTION>')
        lines.append('  <PLAYLISTS>')
        lines.append('    <NODE Type="0" Name="ROOT">')
        lines.append(f'      <NODE Name="{html.escape(playlist_name)}" Type="1" KeyType="0" Entries="{len(tracks)}">')

        for idx in range(1, len(tracks) + 1):
            lines.append(f'        <TRACK Key="{idx}"/>')

        lines.append('      </NODE>')
        lines.append('    </NODE>')
        lines.append('  </PLAYLISTS>')
        lines.append('</DJ_PLAYLISTS>')

        return "\n".join(lines)

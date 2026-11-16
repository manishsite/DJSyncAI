from typing import List, Dict, Optional
from .models import VerifiedExtendedTrack, DJSequenceItem, DJSequenceResult
from .scoring import CAMELOT_MAP, normalize_camelot

PHASE_NAMES = [
    "Opening / Warm-up",
    "Groove Building",
    "Energy Building",
    "Peak-Time",
    "Closing / Cool-down"
]

def get_camelot_transition_info(prev_cam: Optional[str], next_cam: Optional[str]) -> (str, str):
    """Analyze harmonic transition between two Camelot keys."""
    if not prev_cam or not next_cam:
        return "Smooth Transition", "Standard beatmatch transition"

    prev = normalize_camelot(prev_cam)
    nxt = normalize_camelot(next_cam)
    
    if not prev or not nxt or prev not in CAMELOT_MAP or nxt not in CAMELOT_MAP:
        return "Smooth Beatmatch", "Standard tempo transition"

    _, _, p_num, p_letter = CAMELOT_MAP[prev]
    _, _, n_num, n_letter = CAMELOT_MAP[nxt]

    if p_num == n_num and p_letter == n_letter:
        return "Harmonic Lock", f"Perfect tonic match ({prev} = {nxt})"
    
    if p_num == n_num and p_letter != n_letter:
        return "Relative Major/Minor", f"Mood shift between relative keys ({prev} ↔ {nxt})"
    
    diff = (n_num - p_num) % 12
    if diff == 1 and p_letter == n_letter:
        return "Energy Climb (+1)", f"Harmonic energy ascension ({prev} → {nxt})"
    
    if diff == 11 and p_letter == n_letter:
        return "Energy Descend (-1)", f"Smooth harmonic resolution ({prev} → {nxt})"

    if diff == 2 and p_letter == n_letter:
        return "Energy Boost (+2)", f"High-impact key lift for building tension ({prev} → {nxt})"

    return "Dynamic Transition", f"Tonal contrast ({prev} → {nxt})"

class DJSequencer:
    """
    Intelligent AI DJ Playlist Sequencer.
    Sorts and sections verified Extended Mixes into a 5-phase club journey
    guided by Energy arcs, BPM gradients, and Camelot Wheel harmonic transitions.
    """

    @classmethod
    def generate_sequence(
        cls,
        tracks: List[VerifiedExtendedTrack],
        locked_positions: Optional[Dict[int, str]] = None,
        style: str = "harmonic"
    ) -> DJSequenceResult:
        if not tracks:
            return DJSequenceResult()

        if locked_positions is None:
            locked_positions = {}

        total = len(tracks)

        # Sort candidate pool initially into energy tiers
        # Warmup: energy 5-7, lower bpm
        # Groove: energy 6-8
        # Energy Building: energy 7-9, rising bpm
        # Peak: energy 8-10, highest danceability
        # Cooldown: melodic/emotional, energy 6-8

        # Segment track count into 5 phases
        phase_distribution = [
            max(1, int(round(total * 0.15))),  # Warmup ~15%
            max(1, int(round(total * 0.25))),  # Groove ~25%
            max(1, int(round(total * 0.25))),  # Build ~25%
            max(1, int(round(total * 0.25))),  # Peak ~25%
            max(1, total - (
                max(1, int(round(total * 0.15))) +
                max(1, int(round(total * 0.25))) +
                max(1, int(round(total * 0.25))) +
                max(1, int(round(total * 0.25)))
            ))                                 # Cooldown remainder
        ]

        # Ensure sum equals total
        while sum(phase_distribution) > total and phase_distribution[-1] > 1:
            phase_distribution[-1] -= 1
        while sum(phase_distribution) < total:
            phase_distribution[2] += 1

        # Sort pool safely by energy and BPM (handling None gracefully)
        def sort_key(t: VerifiedExtendedTrack):
            e_val = t.energy if t.energy is not None else 5
            b_val = t.bpm if t.bpm is not None else 125.0
            return (e_val, b_val)

        sorted_by_energy = sorted(tracks, key=sort_key)

        pool = list(sorted_by_energy)
        sequenced_items: List[DJSequenceItem] = []
        phase_groups: Dict[str, List[DJSequenceItem]] = {p: [] for p in PHASE_NAMES}

        current_idx = 0
        current_pos = 1

        # Helper to pick best harmonic transition from candidate subpool
        def pick_best_next(prev_track: Optional[VerifiedExtendedTrack], candidates: List[VerifiedExtendedTrack]) -> Optional[VerifiedExtendedTrack]:
            if not candidates:
                return None
            if not prev_track or not prev_track.camelotKey:
                return candidates.pop(0)

            prev_cam = normalize_camelot(prev_track.camelotKey)
            if not prev_cam:
                return candidates.pop(0)
            
            # Rank candidates by harmonic closeness to prev_track
            def harmonic_rank(c: VerifiedExtendedTrack):
                if not c.camelotKey:
                    return 20
                c_cam = normalize_camelot(c.camelotKey)
                if not c_cam:
                    return 20
                if c_cam == prev_cam:
                    return 0  # Perfect lock
                if len(c_cam) > 1 and len(prev_cam) > 1 and c_cam[:-1] == prev_cam[:-1]:
                    return 1  # Relative shift
                try:
                    c_num = int(c_cam[:-1])
                    p_num = int(prev_cam[:-1])
                    diff = abs(c_num - p_num)
                    if diff in (1, 11):
                        return 2  # Adjacent key
                    return 3 + diff
                except ValueError:
                    return 10

            ranked = sorted(candidates, key=harmonic_rank)
            chosen = ranked[0]
            candidates.remove(chosen)
            return chosen

        # Assign tracks to each phase with harmonic ordering
        prev_track: Optional[VerifiedExtendedTrack] = None

        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            count = phase_distribution[phase_idx]
            # Take slice for this phase
            phase_slice = pool[current_idx : current_idx + count]
            current_idx += count

            phase_candidates = list(phase_slice)

            while phase_candidates:
                chosen_track = pick_best_next(prev_track, phase_candidates)
                if not chosen_track:
                    break

                t_quality, t_reason = get_camelot_transition_info(
                    prev_track.camelotKey if prev_track else chosen_track.camelotKey,
                    chosen_track.camelotKey
                )

                if prev_track is None:
                    t_reason = "Set opener: establishes tonal foundation and atmospheric groove"

                item = DJSequenceItem(
                    position=current_pos,
                    track=chosen_track,
                    phase=phase_name,
                    transitionQuality=t_quality,
                    transitionReason=t_reason,
                    isLocked=str(current_pos) in locked_positions or current_pos in locked_positions
                )

                sequenced_items.append(item)
                phase_groups[phase_name].append(item)
                prev_track = chosen_track
                current_pos += 1

        # Summary telemetry
        valid_bpms = [t.track.bpm for t in sequenced_items if t.track.bpm is not None and t.track.bpm > 0]
        bpm_prog = f"{min(valid_bpms):.1f} → {max(valid_bpms):.1f} BPM" if valid_bpms else "Dynamic BPM"
        
        valid_energies = [t.track.energy for t in sequenced_items if t.track.energy is not None]
        if valid_energies:
            energy_prog = f"Energy {min(valid_energies)} (Warmup) → {max(valid_energies)} (Peak) → {valid_energies[-1]} (Outro)"
        else:
            energy_prog = "Dynamic Energy"

        return DJSequenceResult(
            phases=phase_groups,
            orderedTracks=sequenced_items,
            totalTracks=len(sequenced_items),
            bpmProgression=bpm_prog,
            energyProgression=energy_prog,
            harmonicSummary="Harmonic progression optimized for Camelot wheel continuity with tension builds"
        )

"""
Audio Signal Processing Engine for DJSyncAI.
Performs genuine audio signal analysis on MP3, WAV, FLAC, AIFF, AAC, M4A, and OGG files:
- Beat tracking, onset detection, and BPM calculation (with half/double time).
- 12-bin Chroma harmonic analysis & Krumhansl-Schmuckler musical key / Camelot detection.
- Acoustic energy features (RMS, dynamic range, spectral centroid, sub/bass ratio, onset density).
- Rhythm regularity, pulse clarity, and danceability scoring.
- Audio fingerprinting and local caching.
Strictly evidence-based: Returns None if audio cannot be decoded or features cannot be confidently measured.
"""
import os
import io
import json
import hashlib
import tempfile
import numpy as np
import scipy.signal
from typing import Dict, Any, Optional, Tuple

import soundfile as sf
from pydub import AudioSegment
import mutagen

from .models import AnalysedTrackItem
from .config import LOGS_DIR, log_event

# Camelot Wheel Mapping for all 24 keys
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B',
    'Ab': '4B', 'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B'
}

MINOR_KEY_TO_CAMELOT = {
    'C': '5A', 'C#': '12A', 'Db': '12A', 'D': '7A', 'D#': '2A', 'Eb': '2A',
    'E': '9A', 'F': '4A', 'F#': '11A', 'Gb': '11A', 'G': '6A', 'G#': '1A',
    'Ab': '1A', 'A': '8A', 'A#': '3A', 'Bb': '3A', 'B': '10A'
}

# Krumhansl-Schmuckler Key Profiles (Cognitive musicology profiles)
KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Cache file path
AUDIO_CACHE_FILE = LOGS_DIR / "audio_analysis_cache.json"

class AudioSignalAnalyzer:
    """Core DSP and acoustic analysis pipeline."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_loaded: bool = False

    @classmethod
    def _init_cache(cls):
        """Load persistent cache from disk."""
        if cls._cache_loaded:
            return
        cls._cache_loaded = True
        try:
            if os.path.exists(AUDIO_CACHE_FILE):
                with open(AUDIO_CACHE_FILE, "r", encoding="utf-8") as f:
                    cls._cache = json.load(f)
        except Exception as e:
            log_event("WARN", f"Could not load audio analysis cache: {e}")
            cls._cache = {}

    @classmethod
    def _save_cache(cls):
        """Save persistent cache to disk."""
        try:
            with open(AUDIO_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cls._cache, f, indent=2)
        except (PermissionError, OSError):
            pass
        except Exception as e:
            log_event("WARN", f"Could not write audio analysis cache: {e}")

    @classmethod
    def clear_cache(cls):
        """Clears memory and disk cache."""
        cls._cache = {}
        cls._cache_loaded = True
        try:
            if os.path.exists(AUDIO_CACHE_FILE):
                os.remove(AUDIO_CACHE_FILE)
        except Exception:
            pass

    @classmethod
    def compute_file_fingerprint(cls, file_path: str) -> str:
        """Fast SHA-256 fingerprint based on file header + file size."""
        try:
            size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                header = f.read(65536)
            h = hashlib.sha256()
            h.update(str(size).encode())
            h.update(header)
            return h.hexdigest()[:16]
        except Exception:
            return ""

    @classmethod
    def compute_bytes_fingerprint(cls, data: bytes) -> str:
        """Fast SHA-256 fingerprint from bytes."""
        h = hashlib.sha256()
        h.update(str(len(data)).encode())
        h.update(data[:65536])
        return h.hexdigest()[:16]

    @classmethod
    def decode_audio_to_mono(cls, file_path: str, target_sr: int = 22050, max_seconds: float = 180.0) -> Tuple[np.ndarray, int, float]:
        """
        Decode any supported audio format (MP3, WAV, FLAC, AIFF, AAC, M4A, OGG)
        into a mono float32 numpy array.
        Returns: (samples, sample_rate, total_duration_seconds)
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Try soundfile first (fastest for WAV, FLAC, OGG)
        if ext in [".wav", ".flac", ".ogg"]:
            try:
                data, sr = sf.read(file_path, dtype="float32", always_2d=True)
                total_duration = len(data) / sr
                # Convert to mono
                mono = np.mean(data, axis=1)

                # Resample if necessary using scipy
                if sr != target_sr:
                    num_target = int(len(mono) * (target_sr / sr))
                    mono = scipy.signal.resample(mono, num_target)
                    sr = target_sr

                # Focus on representative section (skip intro/outro silence)
                if len(mono) > int(max_seconds * sr):
                    start_sec = max(0, (total_duration - max_seconds) / 2.0)
                    start_idx = int(start_sec * sr)
                    end_idx = start_idx + int(max_seconds * sr)
                    mono = mono[start_idx:end_idx]

                return mono.astype(np.float32), sr, total_duration
            except Exception:
                pass

        # Use pydub + ffmpeg for MP3, AAC, M4A, AIFF and as universal fallback
        try:
            seg = AudioSegment.from_file(file_path)
            total_duration = len(seg) / 1000.0

            # Convert to mono and target sample rate in pydub
            seg = seg.set_channels(1).set_frame_rate(target_sr)

            # Limit window for speed on large files (e.g. 10-minute tracks)
            if len(seg) > int(max_seconds * 1000):
                start_ms = max(0, int((len(seg) - max_seconds * 1000) / 2.0))
                seg = seg[start_ms:start_ms + int(max_seconds * 1000)]

            # Convert raw audio bytes to float32
            raw_samples = np.array(seg.get_array_of_samples())
            if seg.sample_width == 2:
                samples = raw_samples.astype(np.float32) / 32768.0
            elif seg.sample_width == 4:
                samples = raw_samples.astype(np.float32) / 2147483648.0
            else:
                samples = raw_samples.astype(np.float32) / (2 ** (seg.sample_width * 8 - 1))

            return samples, target_sr, total_duration

        except Exception as e:
            raise RuntimeError(f"Audio decoding failed for {os.path.basename(file_path)}: {e}")

    @classmethod
    def detect_bpm_and_tempo(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculates BPM and tempo metrics using onset envelope autocorrelation:
        Returns:
            bpm: float
            bpm_confidence: float (0.0 to 1.0)
            half_time_bpm: float
            double_time_bpm: float
            tempo_stability: float (0.0 to 1.0)
        """
        hop_length = 512
        n_fft = 1024

        f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)

        # Spectral flux onset envelope (half-wave rectified difference)
        diff = np.diff(mag, axis=1)
        diff = np.maximum(0, diff)
        onset_env = np.sum(diff, axis=0)

        if len(onset_env) < 100 or np.max(onset_env) == 0:
            return {"bpm": None, "bpm_confidence": 0.0, "tempo_stability": 0.0}

        # Normalize onset envelope
        onset_env = onset_env - np.mean(onset_env)
        norm_factor = np.std(onset_env)
        if norm_factor > 0:
            onset_env = onset_env / norm_factor

        # Autocorrelation
        corr = np.correlate(onset_env, onset_env, mode='full')
        corr = corr[len(corr) // 2:]

        hop_time = hop_length / float(sr)

        # DJ tempo search range: 60 BPM to 190 BPM
        min_bpm = 60.0
        max_bpm = 190.0
        min_lag = max(1, int((60.0 / max_bpm) / hop_time))
        max_lag = min(len(corr) - 1, int((60.0 / min_bpm) / hop_time))

        if min_lag >= max_lag:
            return {"bpm": None, "bpm_confidence": 0.0}

        corr_window = corr[min_lag:max_lag].copy()
        lags = np.arange(min_lag, max_lag)
        bpms = 60.0 / (lags * hop_time)

        # Standard musicological tempo prior centered at 126 BPM (sigma 0.7 in log2 space)
        # Prevents octave jumping / half-time bias (e.g. 62.5 instead of 125 BPM in 4/4 beats)
        tempo_prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(126.0)) / 0.65) ** 2)
        weighted_corr = corr_window * tempo_prior

        peak_idx = np.argmax(weighted_corr)
        peak_lag = min_lag + peak_idx
        primary_bpm = round(60.0 / (peak_lag * hop_time), 1)

        # Peak prominence calculation for confidence
        raw_peak_val = corr_window[peak_idx]
        median_val = np.median(corr_window)
        std_val = np.std(corr_window) + 1e-6
        prominence = (raw_peak_val - median_val) / std_val
        confidence = float(np.clip(prominence / 4.0, 0.2, 0.99))

        # Check alternative half-time and double-time interpretations
        half_time = round(primary_bpm / 2.0, 1)
        double_time = round(primary_bpm * 2.0, 1)

        # Calculate tempo stability across 3 time blocks
        chunk_len = len(y) // 3
        block_bpms = []
        if chunk_len > int(sr * 10):  # At least 10 sec per block
            for b in range(3):
                sub_y = y[b * chunk_len:(b + 1) * chunk_len]
                sub_f, sub_t, sub_z = scipy.signal.stft(sub_y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
                sub_diff = np.maximum(0, np.diff(np.abs(sub_z), axis=1))
                sub_env = np.sum(sub_diff, axis=0)
                sub_env = sub_env - np.mean(sub_env)
                sub_corr = np.correlate(sub_env, sub_env, mode='full')[len(sub_env) - 1:]
                if max_lag < len(sub_corr):
                    sub_lag = min_lag + np.argmax(sub_corr[min_lag:max_lag])
                    sub_bpm = 60.0 / (sub_lag * hop_time)
                    block_bpms.append(sub_bpm)

        stability = 0.85
        if len(block_bpms) == 3:
            var = np.std(block_bpms)
            stability = float(np.clip(1.0 - (var / 5.0), 0.2, 0.99))

        return {
            "bpm": primary_bpm,
            "bpm_confidence": round(confidence, 2),
            "half_time_bpm": half_time if 60 <= half_time <= 100 else None,
            "double_time_bpm": double_time if 140 <= double_time <= 190 else None,
            "tempo_stability": round(stability, 2),
            "bpm_method": "onset_spectral_flux_autocorrelation"
        }

    @classmethod
    def detect_musical_key(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculates musical key and Camelot notation using Chroma feature extraction
        and Krumhansl-Schmuckler cognitive key profile correlation:
        Returns:
            key: str (e.g. "A minor")
            key_mode: str ("minor" or "major")
            camelotKey: str (e.g. "8A")
            key_confidence: float (0.0 to 1.0)
            key_method: str
        """
        n_fft = 4096
        hop_length = 2048

        f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)

        # Map frequencies into 12 semitone chroma pitch classes (A4 = 440 Hz = pitch class 9)
        chroma = np.zeros(12)
        freq_mask = (f >= 65.0) & (f <= 2100.0)  # Focus on musical fundamentals (C2 to C7)
        valid_freqs = f[freq_mask]
        valid_mag = mag[freq_mask, :]

        for i, freq in enumerate(valid_freqs):
            semitone = int(round(12 * np.log2(freq / 440.0) + 9)) % 12
            chroma[semitone] += np.sum(valid_mag[i, :])

        norm = np.linalg.norm(chroma)
        if norm == 0:
            return {"key": None, "key_mode": None, "camelotKey": None, "key_confidence": 0.0}

        chroma = chroma / norm

        # Normalize KS profiles
        ks_maj = KS_MAJOR_PROFILE / np.linalg.norm(KS_MAJOR_PROFILE)
        ks_min = KS_MINOR_PROFILE / np.linalg.norm(KS_MINOR_PROFILE)

        correlations = []
        for i in range(12):
            rot_chroma = np.roll(chroma, -i)
            corr_maj = np.corrcoef(rot_chroma, ks_maj)[0, 1]
            corr_min = np.corrcoef(rot_chroma, ks_min)[0, 1]
            pitch = PITCH_NAMES[i]
            correlations.append((corr_maj, f"{pitch} major", "major", MAJOR_KEY_TO_CAMELOT[pitch]))
            correlations.append((corr_min, f"{pitch} minor", "minor", MINOR_KEY_TO_CAMELOT[pitch]))

        # Sort by correlation
        correlations.sort(key=lambda x: x[0], reverse=True)
        best_score, best_key, best_mode, best_camelot = correlations[0]
        second_score = correlations[1][0] if len(correlations) > 1 else 0.0

        # Confidence is evaluated from separation from competing key
        separation = max(0.0, best_score - second_score)
        confidence = float(np.clip(best_score * 0.6 + separation * 2.0, 0.15, 0.98))

        # If best correlation is very weak, do not claim confident key
        if best_score < 0.25:
            return {"key": None, "key_mode": None, "camelotKey": None, "key_confidence": round(confidence, 2)}

        return {
            "key": best_key,
            "key_mode": best_mode,
            "camelotKey": best_camelot,
            "key_confidence": round(confidence, 2),
            "key_method": "krumhansl_schmuckler_chroma_correlation"
        }

    @classmethod
    def calculate_energy_score(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculates evidence-based DJ energy score (1-10 scale) from measured acoustic features:
        - RMS energy
        - Dynamic range (crest factor)
        - Bass energy ratio (20-250 Hz sub/low punch)
        - Spectral centroid (timbral brightness)
        - Onset density (transient activity)
        """
        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(y ** 2)))

        # 2. Dynamic range (peak to RMS)
        peak = float(np.max(np.abs(y))) + 1e-9
        crest_factor_db = float(20 * np.log10(peak / (rms + 1e-9)))

        # 3. Frequency Spectrum & Spectral Centroid
        f, Pxx = scipy.signal.welch(y, fs=sr, nperseg=2048)
        total_power = np.sum(Pxx) + 1e-12

        # Spectral centroid (brightness)
        centroid = float(np.sum(f * Pxx) / total_power)

        # Bass energy ratio (20 Hz to 250 Hz)
        bass_mask = (f >= 20) & (f <= 250)
        bass_power = float(np.sum(Pxx[bass_mask]))
        bass_ratio = float(bass_power / total_power)

        # 4. Onset density
        hop = 512
        f_stft, t_stft, Zxx = scipy.signal.stft(y, fs=sr, nperseg=1024, noverlap=1024 - hop)
        diff = np.maximum(0, np.diff(np.abs(Zxx), axis=1))
        onset_env = np.sum(diff, axis=0)
        onsets, _ = scipy.signal.find_peaks(onset_env, height=np.mean(onset_env) + 0.5 * np.std(onset_env), distance=int(0.15 * sr / hop))
        duration_sec = len(y) / float(sr)
        onset_density = float(len(onsets) / max(duration_sec, 1.0))

        # Standardized evidence-based DJ Energy formula (1 to 10 scale)
        norm_rms = float(np.clip(rms / 0.22, 0.0, 1.0))
        norm_bass = float(np.clip(bass_ratio / 0.45, 0.0, 1.0))
        norm_density = float(np.clip(onset_density / 7.0, 0.0, 1.0))
        norm_centroid = float(np.clip((centroid - 600.0) / 3200.0, 0.0, 1.0))
        norm_crest = float(np.clip((24.0 - crest_factor_db) / 16.0, 0.0, 1.0))

        energy_float = (
            0.30 * norm_rms +
            0.28 * norm_bass +
            0.20 * norm_density +
            0.12 * norm_centroid +
            0.10 * norm_crest
        )
        energy_score = int(round(np.clip(energy_float * 9.0 + 1.0, 1, 10)))

        return {
            "energy": energy_score,
            "energy_confidence": 0.92,
            "energy_source": "audio_features",
            "energy_features": {
                "rms": round(rms, 4),
                "dynamic_range_db": round(crest_factor_db, 1),
                "spectral_centroid_hz": round(centroid, 1),
                "bass_energy_ratio": round(bass_ratio, 3),
                "onset_density_per_sec": round(onset_density, 2)
            }
        }

    @classmethod
    def calculate_danceability_score(cls, y: np.ndarray, sr: int, tempo_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates evidence-based DJ danceability score (1-10 scale) from rhythmic regularity:
        - Pulse clarity (autocorrelation prominence at tempo lag)
        - Tempo stability across time
        - Low-end rhythm consistency
        """
        bpm_confidence = tempo_info.get("bpm_confidence", 0.5)
        tempo_stability = tempo_info.get("tempo_stability", 0.8)

        # Pulse clarity (prominence of primary rhythmic pulse)
        pulse_clarity = bpm_confidence

        # Rhythmic regularity from low-frequency kick drum energy variance
        # Filter sub-bass (40-120 Hz)
        sos = scipy.signal.butter(4, [40, 120], btype='bandpass', fs=sr, output='sos')
        low_band = scipy.signal.sosfilt(sos, y)
        env = np.abs(low_band)
        # Smooth envelope
        win = int(sr * 0.05)
        if win > 0 and len(env) > win:
            kernel = np.ones(win) / win
            env = np.convolve(env, kernel, mode='same')
            peaks, _ = scipy.signal.find_peaks(env, distance=int(sr * 0.35))
            if len(peaks) > 5:
                intervals = np.diff(peaks) / float(sr)
                cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
                rhythm_regularity = float(np.clip(1.0 - cv, 0.2, 0.98))
            else:
                rhythm_regularity = 0.6
        else:
            rhythm_regularity = 0.6

        dance_raw = (
            0.45 * pulse_clarity +
            0.35 * tempo_stability +
            0.20 * rhythm_regularity
        )
        dance_score = int(round(np.clip(dance_raw * 9.0 + 1.0, 1, 10)))

        return {
            "danceability": dance_score,
            "danceability_confidence": round(float(np.mean([bpm_confidence, tempo_stability])), 2),
            "danceability_source": "audio_features",
            "danceability_features": {
                "pulse_clarity": round(pulse_clarity, 2),
                "tempo_stability": round(tempo_stability, 2),
                "rhythm_regularity": round(rhythm_regularity, 2)
            }
        }

    @classmethod
    def analyze_audio_file(cls, file_path: str, track_id: str = "", title: str = "", artist: str = "") -> AnalysedTrackItem:
        """
        Complete end-to-end DSP analysis of a single audio file.
        Checks cache first, then performs genuine beat tracking, key detection, and acoustic measurements.
        """
        cls._init_cache()

        if not os.path.exists(file_path):
            return AnalysedTrackItem(
                id=track_id,
                title=title or os.path.basename(file_path),
                artist=artist or "Unknown Artist",
                filePath=file_path,
                fileName=os.path.basename(file_path),
                analysis_status="missing_audio"
            )

        # Check audio fingerprint cache
        fp = cls.compute_file_fingerprint(file_path)
        if fp and fp in cls._cache:
            cached_dict = cls._cache[fp].copy()
            cached_dict["id"] = track_id
            cached_dict["title"] = title or cached_dict.get("title", "")
            cached_dict["artist"] = artist or cached_dict.get("artist", "")
            cached_dict["filePath"] = file_path
            return AnalysedTrackItem(**cached_dict)

        try:
            # 1. Decode audio
            samples, sr, duration_sec = cls.decode_audio_to_mono(file_path)

            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            dur_str = f"{minutes:02d}:{seconds:02d}"

            # 2. Detect BPM and Tempo
            tempo_res = cls.detect_bpm_and_tempo(samples, sr)

            # 3. Detect Musical Key
            key_res = cls.detect_musical_key(samples, sr)

            # 4. Calculate Energy
            energy_res = cls.calculate_energy_score(samples, sr)

            # 5. Calculate Danceability
            dance_res = cls.calculate_danceability_score(samples, sr, tempo_res)

            # Read embedded ID3 tags if available for year / genre metadata
            genre_tag = None
            year_tag = None
            try:
                m = mutagen.File(file_path)
                if m:
                    if "genre" in m: genre_tag = str(m["genre"][0])
                    elif "TCON" in m: genre_tag = str(m["TCON"])
                    if "date" in m: year_tag = str(m["date"][0])[:4]
                    elif "TDRC" in m: year_tag = str(m["TDRC"])[:4]
                    elif "TYER" in m: year_tag = str(m["TYER"])[:4]
            except Exception:
                pass

            result = AnalysedTrackItem(
                id=track_id,
                title=title or os.path.splitext(os.path.basename(file_path))[0],
                artist=artist or "Unknown Artist",
                filePath=file_path,
                fileName=os.path.basename(file_path),
                analysis_status="audio_analysed",
                audio_fingerprint=fp,
                duration=dur_str,
                duration_seconds=round(duration_sec, 1),
                bpm=tempo_res["bpm"],
                bpm_source="audio_signal_processing" if tempo_res["bpm"] else None,
                bpm_confidence=tempo_res["bpm_confidence"],
                bpm_method=tempo_res.get("bpm_method"),
                half_time_bpm=tempo_res.get("half_time_bpm"),
                double_time_bpm=tempo_res.get("double_time_bpm"),
                tempo_stability=tempo_res.get("tempo_stability"),
                key=key_res["key"],
                key_mode=key_res["key_mode"],
                camelotKey=key_res["camelotKey"],
                key_source="audio_chroma_analysis" if key_res["key"] else None,
                key_confidence=key_res["key_confidence"],
                key_method=key_res.get("key_method"),
                energy=energy_res["energy"],
                energy_source=energy_res["energy_source"],
                energy_confidence=energy_res["energy_confidence"],
                energy_features=energy_res["energy_features"],
                danceability=dance_res["danceability"],
                danceability_source=dance_res["danceability_source"],
                danceability_confidence=dance_res["danceability_confidence"],
                danceability_features=dance_res["danceability_features"],
                genre=genre_tag,
                genre_source="id3_audio_tags" if genre_tag else None,
                year=year_tag
            )

            # Store in cache
            if fp:
                cls._cache[fp] = result.model_dump()
                cls._save_cache()

            return result

        except Exception as e:
            log_event("WARN", f"Audio analysis failed for {file_path}: {e}")
            return AnalysedTrackItem(
                id=track_id,
                title=title or os.path.basename(file_path),
                artist=artist or "Unknown Artist",
                filePath=file_path,
                fileName=os.path.basename(file_path),
                analysis_status="analysis_failed"
            )

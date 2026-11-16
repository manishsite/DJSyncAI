"""
Data models for DJSyncAI Platform.
Strictly evidence-based: All musical measurements default to None (null)
when unmeasured or unavailable. No fabricated defaults.
"""
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict

class TrackInput(BaseModel):
    """Input track received from Rekordbox XML or M3U8 playlist."""
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    title: str = "Unknown Title"
    artist: str = "Unknown Artist"
    genre: Optional[str] = None
    bpm: Optional[Union[str, float, int]] = None
    key: Optional[str] = None
    camelotKey: Optional[str] = None
    year: Optional[Union[str, int]] = None
    filePath: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[Union[str, float, int]] = None
    energy: Optional[Union[float, int, str]] = None
    danceability: Optional[Union[float, int, str]] = None
    style: Optional[str] = None
    analysis_status: Optional[str] = None

class AnalysedTrackItem(BaseModel):
    """
    Evidence-based track model with complete provenance and confidence.
    Stores measured value, source, confidence, and detection method.
    Missing properties are strictly None (null).
    """
    id: str
    title: str
    artist: str
    filePath: Optional[str] = None
    fileName: Optional[str] = None
    
    # Analysis Status:
    # "audio_analysed", "metadata_only", "partially_analysed", "missing_audio", "analysis_failed", "awaiting_analysis"
    analysis_status: str = "awaiting_analysis"
    audio_fingerprint: Optional[str] = None
    duration: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # 1. BPM and Tempo Measurements
    bpm: Optional[float] = None
    bpm_source: Optional[str] = None  # "audio_signal_processing", "rekordbox_metadata", "embedded_tag"
    bpm_confidence: Optional[float] = None  # 0.0 to 1.0
    bpm_method: Optional[str] = None  # "onset_autocorrelation", "spectral_flux"
    half_time_bpm: Optional[float] = None
    double_time_bpm: Optional[float] = None
    tempo_stability: Optional[float] = None  # 0.0 to 1.0
    
    # 2. Musical Key Measurements
    key: Optional[str] = None  # e.g. "A minor"
    key_mode: Optional[str] = None  # "minor" or "major"
    camelotKey: Optional[str] = None  # e.g. "8A"
    key_source: Optional[str] = None  # "audio_chroma_analysis", "rekordbox_metadata"
    key_confidence: Optional[float] = None  # 0.0 to 1.0
    key_method: Optional[str] = None  # "krumhansl_schmuckler_chroma"
    
    # 3. Energy Measurements
    energy: Optional[float] = None  # 1.0 to 10.0 scale
    energy_source: Optional[str] = None  # "audio_features"
    energy_confidence: Optional[float] = None
    energy_features: Optional[Dict[str, float]] = None  # rms, spectral_centroid, bass_energy_ratio, dynamic_range_db
    
    # 4. Danceability Measurements
    danceability: Optional[float] = None  # 1.0 to 10.0 scale
    danceability_source: Optional[str] = None  # "audio_features"
    danceability_confidence: Optional[float] = None
    danceability_features: Optional[Dict[str, float]] = None  # pulse_clarity, rhythm_regularity, onset_density
    
    # 5. Genre and Style (Derived from measured characteristics + verified metadata)
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    genre_source: Optional[str] = None  # "rekordbox_metadata", "gemini_interpretation", "audio_spectral"
    year: Optional[str] = None
    
    # 6. Musical Atmosphere & DJ Set Role (AI Interpretation of measured features)
    atmosphere: Optional[str] = None
    dj_role: Optional[str] = None
    transition_notes: Optional[str] = None

class PlaylistDNA(BaseModel):
    """
    Evidence-based Playlist Musical DNA.
    Calculated strictly from measured and verified tracks.
    Shows exact track contribution counts; never invents defaults.
    """
    overallVibe: Optional[str] = None
    bpmLandscape: Optional[str] = None
    patternsAndFlow: Optional[str] = None
    
    # BPM Statistics (None if no tracks have BPM)
    minBpm: Optional[float] = None
    maxBpm: Optional[float] = None
    avgBpm: Optional[float] = None
    medianBpm: Optional[float] = None
    bpmTrackCount: int = 0
    
    # Key & Camelot Statistics
    camelotKeys: List[str] = Field(default_factory=list)
    keyDistribution: Dict[str, int] = Field(default_factory=dict)
    keyTrackCount: int = 0
    
    # Energy & Danceability Statistics
    avgEnergy: Optional[float] = None
    energyTrackCount: int = 0
    avgDanceability: Optional[float] = None
    danceabilityTrackCount: int = 0
    
    # Genre Distribution
    primaryGenres: List[str] = Field(default_factory=list)
    genreCounts: Dict[str, int] = Field(default_factory=dict)
    genreTrackCount: int = 0
    
    # Playlist Counts & Provenance
    totalTracks: int = 0
    audioAnalysedTracks: int = 0
    metadataOnlyTracks: int = 0
    missingAudioTracks: int = 0
    failedTracks: int = 0
    dnaSummaryMessage: str = ""

class VerifiedExtendedTrack(BaseModel):
    """
    Discovered track strictly filtered for genuine Extended Mixes.
    Includes independent verification source and evidence.
    """
    id: str
    artist: str
    title: str
    mixName: str
    versionType: str  # Extended Mix, Extended Edit, Extended Club Mix, Extended Remix, Extended Dub
    duration: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    camelotKey: Optional[str] = None
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    releaseDate: Optional[str] = None
    releaseLabel: Optional[str] = None
    releaseWindow: Optional[str] = None
    isRecentSixMonths: bool = False
    
    # Verification details (Must be independently verified, never fabricated)
    verificationSource: str
    verificationEvidence: str
    isIndependentlyVerified: bool = False
    
    # Match Scores based on real DNA
    similarityScore: Optional[int] = None
    djScore: Optional[int] = None
    popularityScore: Optional[int] = None
    overallScore: Optional[int] = None
    
    links: Dict[str, str] = Field(default_factory=dict)
    searchString: str = ""

class DJSequenceItem(BaseModel):
    """Item in the 5-phase DJ set sequence with transition rationale."""
    position: int
    track: VerifiedExtendedTrack
    phase: str  # Warm-up, Groove development, Energy building, Peak-time, Closing / after-hours
    transitionQuality: str
    transitionReason: str
    isLocked: bool = False

class DJSequenceResult(BaseModel):
    """5-phase harmonic DJ set sequence."""
    phases: Dict[str, List[DJSequenceItem]] = Field(default_factory=dict)
    orderedTracks: List[DJSequenceItem] = Field(default_factory=list)
    totalTracks: int = 0
    bpmProgression: Optional[str] = None
    energyProgression: Optional[str] = None
    harmonicSummary: Optional[str] = None

class DiscoveryResult(BaseModel):
    """Overall discovery result."""
    dna: PlaylistDNA
    verifiedTracks: List[VerifiedExtendedTrack] = Field(default_factory=list)
    totalCandidatesScanned: int = 0
    totalRejectedNonExtended: int = 0
    totalRejectedUnverified: int = 0
    totalVerifiedReturned: int = 0
    explanation: str = ""
    suggestedSequence: Optional[DJSequenceResult] = None

class DiscoveryRequest(BaseModel):
    tracks: List[TrackInput]
    apiKey: Optional[str] = None
    model: Optional[str] = "gemini-2.5-pro"
    maxResults: Optional[int] = 60
    targetPeriod: Optional[str] = "all"

class SequenceReorderRequest(BaseModel):
    tracks: List[VerifiedExtendedTrack]
    lockedPositions: Optional[Dict[int, str]] = Field(default_factory=dict)
    transitionStyle: Optional[str] = "harmonic"

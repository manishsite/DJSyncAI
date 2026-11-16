# DJSyncAI — Intelligent DJ Track Discovery & Rekordbox Playlist Analyzer

DJSyncAI is an advanced DJ workflow platform and Rekordbox playlist analyzer designed for club and touring DJs. It ingests Rekordbox XML exports and M3U8 playlists, extracts complete musical DNA, and autonomously curates **up to 60 verified Extended Mixes and Extended Edits** tailored to the exact vibe, tempo, and harmonic structure of your set.

---

## ⚡ Core Capabilities

### 1. Ingestion & Real-Time Song Analysis
- **Supported Formats**: Pioneer Rekordbox XML (`.xml`) collections and M3U8 (`.m3u8`) playlists.
- **Live Analysis Station**:
  - Real-time Server-Sent Events (SSE) streaming directly from the backend.
  - Live track inspection showing artist, title, BPM, Camelot harmonic key, and energy score.
  - Live progress counters: *Total Loaded*, *Analysed*, *Remaining*, and *Incomplete Metadata*.
  - Full cancellation and activity logging.

### 2. Complete Musical DNA Extraction
Generates a comprehensive 9-point musical profile:
- **Total & Analysed Track Count** with incomplete metadata tracking.
- **BPM Landscape**: Minimum, maximum, and weighted average BPM.
- **Camelot Harmonic Wheel Mapping**: Standard musical keys automatically normalized into Camelot notation (e.g., `8A`, `11B`).
- **Energy & Groove Profile**: Dynamic energy scoring (1–10) evaluated across tempo transitions.
- **Main Genres & Subgenres**: Multi-genre detection across house, techno, melodic, and breakbeat styles.
- **Overall Vibe & Atmosphere**: Set-flow character analysis.
- **Pattern & Flow Summary**: Dynamic sequencing cues.

### 3. AI Smart Track Discovery — Extended Mixes Only (Target: 60 Tracks)
Strict verification guarantees that **only certified extended versions** reach your crate:
- **Accepted Version Types**:
  - `Extended Mix`
  - `Extended Edit`
  - `Extended Club Mix`
  - `Extended Version`
  - `Extended Remix`
  - `Extended Dub` (club versions)
- **Strictly Excluded**:
  - Radio Edits, Radio Mixes, Short Edits, Single Cuts, Intro/Outro Edits.
  - Unverified standard Original Mixes or Album cuts.
- **Zero Padding**: If only 38 verified extended versions meet the musical DNA criteria, only 38 are returned with a transparent explanation. Never fills results with unqualified music.
- **Rolling 6-Month Filter**: Select between *All-Time Club Releases*, *Past 6 Months*, or *Past 1 Year*.

### 4. 17-Stage Visual Discovery Pipeline & Live Audit Stream
- **Honest Source Status**: Transparently checks Beatport, 1001Tracklists, SoundCloud, BPM Supreme, Traxsource, iTunes, and Gemini 3.8 AI Music Intelligence.
- **7 Live Discovery Counters**:
  - *Candidates Discovered*
  - *Extended Mixes Verified*
  - *Rejected Non-Extended (Radio/Short Edits)*
  - *Rejected Unverified*
  - *Duplicates Removed*
  - *Eligible Tracks Remaining*
  - *Final Recommendations (N / 60)*
- **Real-Time Telemetry Log**: Streams every candidate audit decision with instant color-coded badges (`VERIFIED`, `REJECTED`, `DUPLICATE`).

### 5. 5-Phase Harmonic DJ Set Sequencer
Arranges accepted Extended Mixes into a performance-ready arc:
1. **Opening / Warm-up**: Low-to-moderate energy, establishing tempo and tone.
2. **Groove Building**: Rhythmic progression and rising energy.
3. **Energy Building**: Peak-transition tracks with high danceability.
4. **Peak-Time**: Highest energy, driving basslines, peak BPM.
5. **Closing / Cool-down**: Melodic resolution and memorable closing tracks.

### 6. Multi-Format DJ Export
- **Rekordbox XML (`.xml`)**: Drag-and-drop playlist directly into Pioneer Rekordbox.
- **M3U8 Playlist (`.m3u8`)**: Universal playlist compatible with Engine DJ, Traktor, VirtualDJ, and media players.
- **DJ Set Cue Sheet (`.txt`)**: Text breakdown with phase markers, BPM, and Camelot keys.
- **Crate Spreadsheet (`.csv`)**: Full metadata sheet including verification source and direct purchase links.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ installed on Windows, macOS, or Linux.

### Installation
1. Clone or extract this repository to your workspace.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment configuration:
   ```bash
   copy .env.example .env
   ```
4. (Optional) Add your Google Gemini API key to `.env` for AI set analysis and discovery:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application
- **Windows (One-Click Launcher)**: Double-click `Run_Analyzer.bat`.
- **Command Line**:
   ```bash
   python server.py
   ```
- Open your browser to: **`http://localhost:8000`**

---

## 📂 Project Architecture

```
DJSyncAI/
├── backend/
│   ├── app.py                   # Starlette ASGI web application & SSE streaming endpoints
│   ├── config.py                # Environment, file paths, and thread-safe logging
│   ├── discovery_engine.py      # Core musical DNA analysis & candidate coordination
│   ├── exporter.py              # Rekordbox XML, M3U8, CSV, and TXT exporters
│   ├── models.py                # Pydantic data schemas
│   ├── scoring.py               # 8-metric weighted scoring & Camelot normalization
│   ├── sequencer.py             # 5-phase harmonic DJ set sequencer
│   ├── verification.py          # Strict Extended Mix regex verification & deduplication
│   └── connectors/
│       ├── base.py              # Connector base interface
│       ├── gemini_ai.py         # Google Gemini AI discovery connector
│       ├── itunes_store.py      # Public iTunes dance/electronic catalog connector
│       ├── music_links.py       # Beatport, Traxsource, and SoundCloud direct search links
│       └── sources.py           # Source status manager & curated 90+ club candidate catalog
├── Logs/
│   └── logs.txt                 # Persistent diagnostic and activity logs
├── index.html                   # Glassmorphic single-page frontend (HTML5/Tailwind/ES6)
├── requirements.txt             # Python dependencies
├── Run_Analyzer.bat             # Windows one-click launcher
├── server.py                    # Uvicorn production server runner
└── test_discovery.py            # Automated test suite
```

---

## 🛡️ Extended Mix Filtering Rules

| Version Tag | Verification Status | Action |
| :--- | :---: | :--- |
| `Extended Mix` | ✅ Valid | Accepted into scoring engine |
| `Extended Edit` | ✅ Valid | Accepted into scoring engine |
| `Extended Club Mix` | ✅ Valid | Accepted into scoring engine |
| `Extended Remix` | ✅ Valid | Accepted into scoring engine |
| `Extended Dub` | ✅ Valid | Accepted into scoring engine |
| `Radio Edit` / `Radio Mix` | ❌ Prohibited | Strictly rejected & logged |
| `Short Edit` / `Single Edit`| ❌ Prohibited | Strictly rejected & logged |
| `Original Mix` (unverified) | ❌ Prohibited | Strictly rejected unless proven extended |
| `Album Version` | ❌ Prohibited | Strictly rejected |
| `Intro Edit` / `Outro Edit` | ❌ Prohibited | Strictly rejected |

---

## 📝 Diagnostic Logs
All operational and audit events are automatically written to `Logs/logs.txt` with microsecond-accurate timestamps and logging levels:
- `[STATUS]`: File uploads, parsing milestones, and sequence completions.
- `[AUDIT]`: Candidate verification decisions (accepted extended versions, rejected radio cuts).
- `[WARN]`: Missing metadata or offline API fallbacks.
- `[ERROR]` / `[BREAK]`: Exceptions with stack traces.

Access logs directly through the web UI via **View Activity Log** or by inspecting `Logs/logs.txt`.

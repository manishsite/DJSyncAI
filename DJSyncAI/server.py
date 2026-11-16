"""
DJSyncAI Platform Server Entry Point
Runs the asynchronous Starlette/Uvicorn backend on port 8000.
"""
import sys
import uvicorn
from backend.config import PORT, HOST, LOG_FILE, log_event

def run():
    log_event("SERVER", f"DJSyncAI Platform v2.0 started on http://localhost:{PORT}")
    log_event("SERVER", f"Ready: Web interface at http://localhost:{PORT}")
    log_event("SERVER", f"Logger active: appending to {LOG_FILE}")
    try:
        uvicorn.run("backend.app:app", host="0.0.0.0", port=PORT, log_level="warning")
    except KeyboardInterrupt:
        log_event("SERVER", "DJSyncAI server stopped cleanly.")

if __name__ == "__main__":
    run()

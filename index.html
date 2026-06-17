#!/usr/bin/env python3
"""
Sox Daily Recap — automated morning podcast pipeline.

Fetches last night's Red Sox result from the MLB Stats API, writes a two-host
script with Claude, synthesises the voices with ElevenLabs, stitches them into
one MP3, and saves it to ~/Desktop/Sox Daily Recap/.

Usage:
    python pipeline.py              # recap for yesterday
    python pipeline.py 2025-04-01  # recap for a specific date
"""

import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from pydub import AudioSegment

# Load .env sitting next to this script
load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
HENRY_VOICE_ID     = os.environ.get("HENRY_VOICE_ID",     "")
DAD_VOICE_ID       = os.environ.get("DAD_VOICE_ID",       "")
ELEVENLABS_MODEL   = os.environ.get("ELEVENLABS_MODEL",   "eleven_turbo_v2_5")

# Where episodes are saved. Override with OUTPUT_DIR env var.
_raw_out   = os.environ.get("OUTPUT_DIR", str(Path.home() / "Desktop" / "Sox Daily Recap"))
OUTPUT_DIR = Path(_raw_out).expanduser()

RED_SOX_TEAM_ID   = 111   # MLB Stats API team ID for Boston Red Sox
ELEVENLABS_BASE   = "https://api.elevenlabs.io/v1"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Config validation ──────────────────────────────────────────────────────────
def _check_config() -> None:
    missing = [
        name for name, val in {
            "ANTHROPIC_API_KEY":  ANTHROPIC_API_KEY,
            "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
            "HENRY_VOICE_ID":     HENRY_VOICE_ID,
            "DAD_VOICE_ID":       DAD_VOICE_ID,
        }.items() if not val
    ]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        log.error("Copy .env.example → .env and fill in your credentials.")
        sys.exit(1)


# ── MLB Stats API ──────────────────────────────────────────────────────────────
def fetch_game(date_str: str) -> Optional[dict]:
    """
    Fetch the Red Sox game summary for date_str (YYYY-MM-DD).
    Returns a dict of key stats, or None if the Sox had an off day.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={RED_SOX_TEAM_ID}&date={date_str}"
        "&hydrate=linescore,decisions"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.error("MLB Stats API error: %s", exc)
        raise

    dates = r.json().get("dates", [])
    if not dates or not dates[0].get("games"):
        return None   # off day or postponed

    return _parse_game(dates[0]["games"][0])


def _parse_game(raw: dict) -> dict:
    state = raw.get("status", {}).get("abstractGameState", "Unknown")
    if state != "Final":
        return {"state": state}

    away = raw["teams"]["away"]
    home = raw["teams"]["home"]
    sox_side = "away" if away["team"]["id"] == RED_SOX_TEAM_ID else "home"
    opp_side = "home" if sox_side == "away" else "away"

    sox = raw["teams"][sox_side]
    opp = raw["teams"][opp_side]
    dec = raw.get("decisions", {})

    return {
        "state":            "Final",
        "result":           "Win" if sox.get("isWinner") else "Loss",
        "opponent":         opp["team"]["name"],
        "sox_score":        sox.get("score", 0),
        "opp_score":        opp.get("score", 0),
        "home_away":        sox_side,
        "winning_pitcher":  dec.get("winner", {}).get("fullName", "Unknown"),
        "losing_pitcher":   dec.get("loser",  {}).get("fullName", "Unknown"),
        "save_pitcher":     dec.get("save",   {}).get("fullName"),
    }


# ── Script generation (Claude) ─────────────────────────────────────────────────
_SYSTEM = """\
You write short, punchy morning podcast scripts for "Sox Daily Recap."

Two hosts:
  HENRY — the son, early 20s, high energy. Uses phrases like "absolute missile",
           "he ate", "that slider was FILTHY", "cooked", "no-doubt homer".
  DAD   — his father, mid-50s, old-school Red Sox lifer. Dry wit, mentions 2004
           constantly, grumbles about launch angles and spin rates but secretly
           respects the numbers.

Rules:
- Output ONLY the dialogue. No stage directions, no markdown, no narration.
- Format every line exactly as:
    HENRY: [line]
    DAD: [line]
- Write 10–14 exchanges total.
- Keep lines punchy and conversational — this is audio, not print.
- End with a quick sign-off from both hosts.
"""


def generate_script(game: Optional[dict], date_str: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    if game is None:
        user_msg = (
            f"The Red Sox had an off day on {date_str}. "
            "Write an episode where Henry and Dad talk about the current standings, "
            "an upcoming series, and argue about one roster or lineup decision."
        )
    elif game.get("state") != "Final":
        user_msg = (
            f"The Red Sox game on {date_str} has status: '{game['state']}'. "
            "Write a short hold episode discussing the situation and recent Sox news."
        )
    else:
        sv = game["save_pitcher"] or "none"
        user_msg = (
            f"Write today's episode based on last night's game.\n\n"
            f"  Result:           Red Sox {game['result']}\n"
            f"  Opponent:         {game['opponent']}\n"
            f"  Final score:      Red Sox {game['sox_score']}, "
            f"{game['opponent']} {game['opp_score']}\n"
            f"  Sox played:       {game['home_away']}\n"
            f"  Winning pitcher:  {game['winning_pitcher']}\n"
            f"  Losing pitcher:   {game['losing_pitcher']}\n"
            f"  Save:             {sv}\n\n"
            "React naturally. Reference specific details. Have them disagree about something."
        )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ── Script parser ──────────────────────────────────────────────────────────────
def parse_script(script: str) -> list[tuple[str, str]]:
    """Return [(speaker, line), ...] where speaker is 'henry' or 'dad'."""
    lines = []
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        upper = raw.upper()
        if upper.startswith("HENRY:"):
            lines.append(("henry", raw[6:].strip()))
        elif upper.startswith("DAD:"):
            lines.append(("dad", raw[4:].strip()))
        else:
            log.debug("Skipping unrecognised line: %r", raw)
    return lines


# ── ElevenLabs TTS ─────────────────────────────────────────────────────────────
def _tts(text: str, voice_id: str) -> bytes:
    r = requests.post(
        f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
        headers={
            "Accept":       "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key":   ELEVENLABS_API_KEY,
        },
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability":         0.45,
                "similarity_boost":  0.80,
                "style":             0.00,
                "use_speaker_boost": True,
            },
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.content


def build_audio(lines: list[tuple[str, str]]) -> AudioSegment:
    """Synthesise every dialogue line and stitch with natural pauses."""
    episode    = AudioSegment.empty()
    gap_same   = AudioSegment.silent(duration=350)   # same-speaker follow-up
    gap_switch = AudioSegment.silent(duration=650)   # speaker change
    tmp_paths: list[str] = []
    prev_spk: Optional[str] = None

    try:
        for i, (spk, text) in enumerate(lines, 1):
            voice = HENRY_VOICE_ID if spk == "henry" else DAD_VOICE_ID
            log.info("  [%d/%d] %-5s  synthesising…", i, len(lines), spk.upper())
            data = _tts(text, voice)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(data)
                tmp_paths.append(f.name)

            seg = AudioSegment.from_mp3(tmp_paths[-1])

            if prev_spk is not None:
                episode += gap_switch if prev_spk != spk else gap_same

            episode  += seg
            prev_spk  = spk
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return episode


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run(date_str: Optional[str] = None) -> Path:
    _check_config()

    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("═══  Sox Daily Recap  ·  %s  ═══", date_str)

    # 1. Game data
    log.info("Step 1/4  Fetching game data from MLB Stats API…")
    game = fetch_game(date_str)
    if game and game.get("state") == "Final":
        log.info("         %s  vs %s  —  Sox %s–%s  (%s)",
                 game["result"].upper(), game["opponent"],
                 game["sox_score"], game["opp_score"], game["home_away"])
    elif game is None:
        log.info("         Off day.")
    else:
        log.info("         Game state: %s", game.get("state"))

    # 2. Script
    log.info("Step 2/4  Generating script with Claude…")
    script      = generate_script(game, date_str)
    script_lines = parse_script(script)
    log.info("         %d dialogue lines parsed.", len(script_lines))
    if not script_lines:
        raise RuntimeError(
            "Script produced 0 parseable lines.\n"
            "Raw script output:\n" + script
        )

    # 3. Audio
    log.info("Step 3/4  Synthesising voices with ElevenLabs…")
    episode      = build_audio(script_lines)
    duration_sec = len(episode) / 1000
    log.info("         Episode length: %.1f s", duration_sec)

    # 4. Save
    log.info("Step 4/4  Saving MP3…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"sox-recap-{date_str}.mp3"
    episode.export(
        str(out_path),
        format="mp3",
        bitrate="128k",
        tags={
            "title":  f"Sox Daily Recap – {date_str}",
            "artist": "Henry & Dad",
            "album":  "Sox Daily Recap",
        },
    )
    log.info("         Saved → %s", out_path)
    return out_path


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(date_arg)

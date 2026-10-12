# Voice Moderation Bot

A Discord bot that listens to voice channels, transcribes speech in real-time using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and automatically mutes members who use prohibited language.

Supports running multiple bot tokens in parallel so each token can monitor a separate voice channel simultaneously.

---

## Features

- Real-time speech-to-text via Whisper (GPU or CPU)
- Arabic and English profanity detection with two severity levels
- Automatic mute with configurable duration
- Multi-bot support — one token per voice channel
- Persistent mutes that survive bot restarts
- Inline unmute button on log messages
- Optional auto-distribution across populated channels
- Dry-run mode for testing without applying mutes
- Member exemption list (`/immune`)

---

## Requirements

- Python 3.10+
- `ffmpeg` installed and on `PATH`
- CUDA-capable GPU recommended (falls back to CPU automatically)

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration install the CUDA runtime wheels:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Required — comma-separated list of bot tokens.
# First token = primary bot (commands + muting).
# Additional tokens = monitor-only bots.
DISCORD_TOKENS=TOKEN_ONE,TOKEN_TWO,TOKEN_THREE

# Channel ID where mute/violation logs are sent.
LOG_CHANNEL_ID=123456789012345678

# Duration of an automatic mute, in seconds. Default: 300 (5 min).
MUTE_SECONDS=300

# Whisper model size: tiny / base / small / medium / large-v3
# Larger = more accurate but slower and more RAM. Default: small
WHISPER_MODEL=small

# Device: auto / cpu / cuda. Default: auto (tries GPU first)
WHISPER_DEVICE=auto

# Fix the transcription language for better accuracy (e.g. ar, en).
# Leave empty for automatic language detection.
WHISPER_LANGUAGE=ar

# Beam size for Whisper decoding. 0 = auto (5 on GPU, 3 on CPU).
BEAM_SIZE=0

# Number of mild-language hits within MILD_WINDOW seconds before muting.
MILD_STRIKES=2
MILD_WINDOW=120

# Comma-separated voice channel IDs to never monitor.
IGNORE_CHANNEL_IDS=

# Comma-separated role IDs that are always exempt from muting.
IMMUNE_ROLE_IDS=

# Set to true to exempt server administrators from muting.
IMMUNE_ADMINS=false

# true  = log violations but never apply mutes (for testing).
DRY_RUN=false

# true  = post a public message in the voice channel when someone is muted.
ANNOUNCE=true

# true  = automatically distribute bots across populated channels every N seconds.
AUTO_WATCH=false
AUTO_WATCH_INTERVAL=10

# Print every transcribed segment to the console.
DEBUG=false
```

---

## Running

```bash
python bot.py
```

---

## Slash Commands

All commands require **Mute Members** unless noted.

| Command | Description |
|---|---|
| `/join` | Send an available bot to your current voice channel |
| `/leave` | Remove the bot from your voice channel |
| `/leaveall` | Disconnect all monitoring bots |
| `/watchall` | Auto-distribute bots across the busiest channels |
| `/mute <member> [minutes]` | Manually mute a member |
| `/unmute <member>` | Manually unmute a member |
| `/unmuteall` | Unmute everyone currently muted by the bot |
| `/mutes` | List active mutes and remaining time |
| `/status` | Show bot status, Whisper info, and queue depth |
| `/addword <word> [level]` | Add a word to the profanity list *(requires Manage Server)* |
| `/immune <member> [enabled]` | Exempt or un-exempt a member *(requires Manage Server)* |
| `/dryrun <enabled>` | Toggle dry-run mode *(requires Manage Server)* |
| `/muteduration <minutes>` | Change auto-mute duration *(requires Manage Server)* |

---

## Profanity Detection

Detection runs in two passes:

| Level | Behaviour |
|---|---|
| `strong` | Mutes immediately on first occurrence |
| `mild` | Mutes only after `MILD_STRIKES` hits within `MILD_WINDOW` seconds |

**Word lists** (one word per line, lines starting with `#` are ignored):

| File | Level |
|---|---|
| `badwords.txt` | strong |
| `mildwords.txt` | mild |

Built-in defaults are always active. Files are optional and extend the defaults. Use `/addword` to append words at runtime without restarting.

---

## Multi-Bot Setup

Each additional token in `DISCORD_TOKENS` creates a monitor-only bot. The first token handles all commands and executes mutes; the rest only listen.

All bots must be invited to the server with these permissions:
- Connect
- Speak *(required by py-cord voice)*
- Mute Members *(primary bot only)*

---

## Persistent State

| File | Contents |
|---|---|
| `mutes.json` | Active mutes (user ID → expiry timestamp) |
| `immune.json` | Exempt user IDs |
| `badwords.txt` | Custom strong-level words |
| `mildwords.txt` | Custom mild-level words |

---

## Architecture

```
Discord voice channel
        │  PCM audio (48 kHz stereo)
        ▼
   LiveSink.write()
        │  per-user audio chunks
        ▼
  SegmentBuffer  ──(silence detected)──▶  segment_queue
                                               │
                                     transcribe_worker()
                                               │  Whisper STT
                                               ▼
                                      ProfanityDetector.check()
                                               │  hit found
                                               ▼
                                          _punish()
                                    (mute + log + announce)
```

---

## License

MIT

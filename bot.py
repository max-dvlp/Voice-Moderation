import asyncio
import json
import os
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import discord
import numpy as np
from discord.ext import commands, tasks
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from profanity import ProfanityDetector, normalize

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _add_nvidia_dll_dirs() -> None:
    if os.name != "nt":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        bases = list(spec.submodule_search_locations or []) if spec else []
    except Exception:
        return
    for base in bases:
        for sub in Path(base).glob("*/bin"):
            try:
                os.add_dll_directory(str(sub))
            except Exception:
                pass
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_ids(name: str) -> set[int]:
    return {int(x) for x in os.getenv(name, "").replace(" ", "").split(",") if x.isdigit()}


TOKENS: list[str] = [
    t.strip()
    for t in (os.getenv("DISCORD_TOKENS") or os.getenv("DISCORD_TOKEN", "")).split(",")
    if t.strip()
]
if not TOKENS:
    raise SystemExit("Set DISCORD_TOKENS (or DISCORD_TOKEN) in your .env file.")

AUTO_WATCH          = _env_bool("AUTO_WATCH", False)
AUTO_WATCH_INTERVAL = int(os.getenv("AUTO_WATCH_INTERVAL", "10"))
IGNORE_CHANNEL_IDS  = _env_ids("IGNORE_CHANNEL_IDS")
DEBUG               = _env_bool("DEBUG", False)
MUTE_SECONDS        = int(os.getenv("MUTE_SECONDS", "300"))
WHISPER_MODEL       = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE      = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE    = os.getenv("WHISPER_LANGUAGE", "").strip() or None
BEAM_SIZE           = int(os.getenv("BEAM_SIZE", "0"))
LOG_CHANNEL_ID      = int(os.getenv("LOG_CHANNEL_ID", "0"))
DRY_RUN             = _env_bool("DRY_RUN", False)
ANNOUNCE            = _env_bool("ANNOUNCE", True)
IMMUNE_ADMINS       = _env_bool("IMMUNE_ADMINS", False)
IMMUNE_ROLE_IDS     = _env_ids("IMMUNE_ROLE_IDS")
MILD_STRIKES        = int(os.getenv("MILD_STRIKES", "2"))
MILD_WINDOW         = int(os.getenv("MILD_WINDOW", "120"))

SAMPLE_RATE      = 48000
SILENCE_GAP      = 0.8
MAX_SEGMENT      = 12.0
MIN_SEGMENT      = 0.4
ENERGY_THRESHOLD = 300
STALE_SECONDS    = 25

MUTES_FILE = Path(__file__).with_name("mutes.json")
detector   = ProfanityDetector()

HALLUCINATIONS = [normalize(x) for x in (
    "اشتركوا في القناة", "اشترك في القناة", "لا تنسوا الاشتراك", "شكرا للمشاهدة",
    "شكرا على المشاهدة", "ترجمة نانسي قنقر", "thanks for watching",
    "thank you for watching", "please subscribe", "subtitles by",
)]


def _load_whisper() -> tuple[WhisperModel, str]:
    if WHISPER_DEVICE in ("auto", "cuda"):
        for compute in ("float16", "int8_float16"):
            try:
                model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type=compute)
                segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32))
                list(segs)
                return model, f"GPU ({compute})"
            except Exception as e:
                print(f"Whisper: GPU/{compute} unavailable ({e})")
        print("Whisper: falling back to CPU")
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8"), "CPU"


whisper, WHISPER_RUNTIME = _load_whisper()
if BEAM_SIZE <= 0:
    BEAM_SIZE = 5 if WHISPER_RUNTIME.startswith("GPU") else 3

print(
    f"Whisper [{WHISPER_MODEL}] running on {WHISPER_RUNTIME}"
    + (f" (fixed language: {WHISPER_LANGUAGE})" if WHISPER_LANGUAGE else " (auto language detection)")
)


def pcm_to_float16k(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a = a[: (a.size // 6) * 6].reshape(-1, 6).mean(axis=1)
    a /= 32768.0
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak > 0:
        a *= min(0.9 / peak, 8.0)
    return a


def is_hallucination(text: str) -> bool:
    n = normalize(text)
    return len(n) < 2 or any(h in n for h in HALLUCINATIONS)


async def transcribe(pcm: bytes) -> str:
    audio = pcm_to_float16k(pcm)

    def _run() -> str:
        segments, _ = whisper.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_speech_duration_ms=100,
                min_silence_duration_ms=600,
                speech_pad_ms=400,
            ),
            beam_size=BEAM_SIZE,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        parts = []
        for seg in segments:
            if seg.avg_logprob < -1.5:
                if DEBUG:
                    print(f"[SKIP] low-confidence segment: {seg.text!r}")
                continue
            parts.append(seg.text)
        return " ".join(parts).strip()

    return await asyncio.get_running_loop().run_in_executor(None, _run)


intents = discord.Intents.default()
intents.voice_states = True
intents.members = True

bot        = commands.Bot(command_prefix="!", intents=intents)
extra_bots = [discord.Bot(intents=intents) for _ in TOKENS[1:]]
bots: list = [bot] + extra_bots

segment_queue: asyncio.Queue | None = None
sinks: dict[tuple[int, int], "LiveSink"] = {}
mild_hits: dict[int, deque] = {}


def _load_mutes() -> dict[int, dict]:
    try:
        raw = json.loads(MUTES_FILE.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_mutes() -> None:
    try:
        MUTES_FILE.write_text(json.dumps(mutes), encoding="utf-8")
    except Exception as e:
        print(f"Failed to save mutes.json: {e}")


mutes: dict[int, dict] = _load_mutes()


def fmt_duration(seconds: int) -> str:
    return f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds}s"


def fmt_remaining(seconds: float) -> str:
    s = max(int(seconds), 0)
    return f"{s // 60}:{s % 60:02d}"


IMMUNE_FILE = Path(__file__).with_name("immune.json")


def _load_immune() -> set[int]:
    try:
        return {int(x) for x in json.loads(IMMUNE_FILE.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def _save_immune() -> None:
    try:
        IMMUNE_FILE.write_text(json.dumps(sorted(immune_users)), encoding="utf-8")
    except Exception as e:
        print(f"Failed to save immune.json: {e}")


immune_users: set[int] = _load_immune()


class SegmentBuffer:
    def __init__(self) -> None:
        self.chunks:     list[bytes] = []
        self.last_voice: float       = 0.0
        self.start:      float       = 0.0
        self.lock                    = threading.Lock()
        self.preroll:    deque       = deque(maxlen=15)

    def add(self, pcm: bytes, now: float) -> None:
        arr  = np.frombuffer(pcm, dtype=np.int16)
        loud = arr.size > 0 and np.abs(arr).mean() > ENERGY_THRESHOLD
        with self.lock:
            if loud:
                if not self.chunks:
                    self.start = now
                    self.chunks.extend(self.preroll)
                    self.preroll.clear()
                self.last_voice = now
            if self.chunks or loud:
                self.chunks.append(pcm)
            else:
                self.preroll.append(pcm)

    def ready(self, now: float) -> bool:
        with self.lock:
            if not self.chunks:
                return False
            return (now - self.last_voice > SILENCE_GAP) or (now - self.start > MAX_SEGMENT)

    def pop(self) -> bytes:
        with self.lock:
            data = b"".join(self.chunks)
            self.chunks = []
        return data


class LiveSink(discord.sinks.Sink):
    def __init__(self, loop: asyncio.AbstractEventLoop, guild: discord.Guild) -> None:
        super().__init__()
        self.loop        = loop
        self.guild       = guild
        self.buffers:    dict[int, SegmentBuffer] = {}
        self.task:       asyncio.Task | None      = None
        self.last_packet: float                   = 0.0
        self.packets:    int                      = 0

    def is_opus(self) -> bool:
        return False

    def write(self, *args) -> None:
        try:
            data = next((a for a in args if hasattr(a, "pcm")), None)
            if data is None or not data.pcm:
                return
            user = next((a for a in args if a is not data), None)
            if user is None:
                user = getattr(data, "source", None)
            uid = user if isinstance(user, int) else getattr(user, "id", None)
            if uid is None:
                return
            pcm = data.pcm[: len(data.pcm) - (len(data.pcm) % 2)]
            now = time.monotonic()
            self.last_packet = now
            self.packets += 1
            self.buffers.setdefault(uid, SegmentBuffer()).add(pcm, now)
        except Exception:
            traceback.print_exc()

    def start_worker(self) -> None:
        self.task = self.loop.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            await asyncio.sleep(0.3)
            now = time.monotonic()
            for uid, buf in list(self.buffers.items()):
                if buf.ready(now):
                    pcm = buf.pop()
                    dur = len(pcm) / (SAMPLE_RATE * 4)
                    if DEBUG:
                        print(f"[SEGMENT] {dur:.1f}s from {uid}" + ("" if dur >= MIN_SEGMENT else " (too short, skipped)"))
                    if dur >= MIN_SEGMENT and segment_queue is not None:
                        try:
                            segment_queue.put_nowait((self.guild.id, uid, pcm, now))
                        except asyncio.QueueFull:
                            print("[WARN] Segment queue full — consider a smaller model")

    def cleanup(self) -> None:
        if self.task:
            self.task.cancel()
            self.task = None


async def transcribe_worker() -> None:
    while True:
        guild_id, uid, pcm, ts = await segment_queue.get()
        try:
            if time.monotonic() - ts > STALE_SECONDS:
                continue
            guild = bot.get_guild(guild_id)
            if guild:
                await process_segment(guild, uid, pcm)
        except Exception:
            traceback.print_exc()


def is_immune(member: discord.Member) -> bool:
    if member.id in immune_users:
        return True
    if IMMUNE_ADMINS and member.guild_permissions.administrator:
        return True
    return any(r.id in IMMUNE_ROLE_IDS for r in member.roles)


async def process_segment(guild: discord.Guild, uid: int, pcm: bytes) -> None:
    member = guild.get_member(uid)
    if not member or member.bot or is_immune(member):
        return
    if uid in mutes and mutes[uid]["until"] > time.time():
        return

    text = await transcribe(pcm)
    if DEBUG:
        print(f"[STT] {member.display_name}: {text!r}")
    if not text or is_hallucination(text):
        return

    hit = detector.check(text)
    if not hit:
        return

    level, word = hit

    if level == "mild":
        now = time.time()
        dq  = mild_hits.setdefault(uid, deque())
        while dq and now - dq[0] > MILD_WINDOW:
            dq.popleft()
        dq.append(now)
        if len(dq) < MILD_STRIKES:
            await _log(guild, f"⚠️ Mild language from {member.mention}: `{word}` ({len(dq)}/{MILD_STRIKES})\nText: ||{text}||")
            return
        dq.clear()

    await _punish(guild, member, text, word)


async def _log(guild: discord.Guild, msg: str, view=None) -> discord.Message | None:
    print(msg.replace("\n", " | "))
    if LOG_CHANNEL_ID:
        ch = guild.get_channel(LOG_CHANNEL_ID)
        if ch:
            try:
                return await ch.send(msg, view=view)
            except discord.HTTPException:
                pass
    return None


async def _apply_mute(guild: discord.Guild, member: discord.Member, seconds: int, reason: str) -> str | None:
    try:
        await member.edit(mute=True, reason=reason)
    except discord.Forbidden:
        return "Mute failed: check Mute Members permission and role hierarchy"
    except discord.HTTPException as e:
        return f"Mute failed: {e}"
    mutes[member.id] = {"guild": guild.id, "until": time.time() + seconds}
    _save_mutes()
    return None


async def _do_unmute(member: discord.Member, reason: str) -> bool:
    info = mutes.get(member.id)
    if member.voice and member.voice.channel:
        if member.voice.mute:
            await member.edit(mute=False, reason=reason)
        mutes.pop(member.id, None)
        _save_mutes()
        return True
    if info:
        info["until"] = 0
        _save_mutes()
    return False


class UnmuteView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=None)

    @discord.ui.button(
        label="🔊 Unmute",
        style=discord.ButtonStyle.success,
        custom_id="unmute_btn",
    )
    async def unmute_btn(
        self,
        button: discord.ui.Button,
        interaction: discord.Interaction,
    ) -> None:
        if not interaction.user.guild_permissions.mute_members:
            return await interaction.response.send_message(
                "You need Mute Members permission.", ephemeral=True
            )
        mid    = interaction.message.id
        uid    = next((u for u, i in mutes.items() if i.get("log_msg") == mid), None)
        member = interaction.guild.get_member(uid) if uid else None
        if member is None:
            return await interaction.response.send_message(
                "Mute already expired or removed.", ephemeral=True
            )
        try:
            await _do_unmute(member, f"Unmuted via button by {interaction.user}")
        except discord.HTTPException as e:
            return await interaction.response.send_message(f"❌ Failed: `{e}`", ephemeral=True)

        done = discord.ui.View(timeout=None)
        done.add_item(discord.ui.Button(
            label=f"Unmuted by {interaction.user.display_name} ✅",
            style=discord.ButtonStyle.secondary,
            disabled=True,
        ))
        await interaction.response.edit_message(view=done)


async def _punish(
    guild: discord.Guild,
    member: discord.Member,
    text: str,
    word: str,
) -> None:
    if not member.voice or not member.voice.channel:
        return
    if member.voice.mute:
        return

    channel = member.voice.channel

    if DRY_RUN:
        return await _log(
            guild,
            f"🧪 [DRY RUN] Would have muted {member.mention} for `{word}`\nText: ||{text}||",
        )

    err = await _apply_mute(guild, member, MUTE_SECONDS, f"Voice profanity: {word}")
    if err:
        return await _log(guild, f"❌ {err}")

    sent = await _log(
        guild,
        f"🔇 {member.mention} muted for {fmt_duration(MUTE_SECONDS)}\n"
        f"Word: `{word}`\nText: ||{text}||\n"
        f"Use the button below or `/unmute` to reverse if this was a false positive.",
        view=UnmuteView(),
    )
    if sent and member.id in mutes:
        mutes[member.id]["log_msg"] = sent.id
        _save_mutes()

    if ANNOUNCE:
        try:
            await channel.send(
                f"🔇 {member.mention} has been muted for {fmt_duration(MUTE_SECONDS)} due to prohibited language."
            )
        except Exception:
            pass


@tasks.loop(seconds=5)
async def _expire_mutes() -> None:
    now     = time.time()
    changed = False
    for uid, info in list(mutes.items()):
        if info["until"] > now:
            continue
        guild = bot.get_guild(info["guild"])
        if guild is None:
            continue
        member = guild.get_member(uid)
        if member is None:
            mutes.pop(uid, None)
            changed = True
            continue
        if member.voice is None:
            continue
        try:
            if member.voice.mute:
                await member.edit(mute=False, reason="Mute duration expired")
        except discord.HTTPException:
            continue
        mutes.pop(uid, None)
        changed = True
    if changed:
        _save_mutes()


@_expire_mutes.before_loop
async def _wait_primary_ready() -> None:
    await bot.wait_until_ready()


@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
) -> None:
    info = mutes.get(member.id)
    if info and after.channel and before.channel != after.channel:
        if info["until"] > time.time() and not after.mute:
            try:
                await member.edit(mute=True, reason="Mute duration not yet expired")
            except discord.HTTPException:
                pass


async def _after_recording(sink, *args) -> None:
    pass


watch_lock  = asyncio.Lock()
failed_until: dict[tuple[int, int], float] = {}


def _bot_name(b) -> str:
    return b.user.name if b.user else "unknown"


def _voice_client(b, guild_id: int) -> discord.VoiceClient | None:
    g = b.get_guild(guild_id)
    return g.voice_client if g else None


def _watched_channels(guild_id: int) -> dict[int, object]:
    result = {}
    for b in bots:
        vc = _voice_client(b, guild_id)
        if vc and vc.channel:
            result[vc.channel.id] = b
    return result


def _free_bots(guild_id: int) -> list:
    return [
        b for b in bots
        if b.is_ready() and b.get_guild(guild_id) and not _voice_client(b, guild_id)
    ]


async def _start_watch(b, guild_id: int, channel_id: int) -> None:
    guild   = b.get_guild(guild_id)
    channel = guild.get_channel(channel_id)
    vc      = await channel.connect()
    sink    = LiveSink(asyncio.get_running_loop(), guild)
    try:
        sink.start_worker()
        vc.start_recording(sink, _after_recording)
    except Exception:
        sink.cleanup()
        await vc.disconnect(force=True)
        raise
    sinks[(b.user.id, guild_id)] = sink


async def _stop_watch(b, guild_id: int) -> None:
    vc = _voice_client(b, guild_id)
    if vc:
        try:
            vc.stop_recording()
        except Exception:
            pass
    sink = sinks.pop((b.user.id, guild_id), None)
    if sink:
        sink.cleanup()
    if vc:
        await vc.disconnect(force=True)


async def _rebalance(guild: discord.Guild, release: bool) -> list:
    humans: dict[int, int] = {}
    for ch in guild.voice_channels:
        if ch.id in IGNORE_CHANNEL_IDS:
            continue
        if guild.afk_channel and ch.id == guild.afk_channel.id:
            continue
        count = sum(1 for m in ch.members if not m.bot)
        if count:
            humans[ch.id] = count

    if release:
        for cid, b in list(_watched_channels(guild.id).items()):
            if cid not in humans:
                try:
                    await _stop_watch(b, guild.id)
                except Exception:
                    traceback.print_exc()

    watched = _watched_channels(guild.id)
    free    = _free_bots(guild.id)
    started = []
    now     = time.time()

    for cid, _ in sorted(humans.items(), key=lambda x: -x[1]):
        if cid in watched:
            continue
        if not free:
            break
        if failed_until.get((guild.id, cid), 0) > now:
            continue
        b = free.pop(0)
        try:
            await _start_watch(b, guild.id, cid)
            started.append((b, cid))
        except Exception as e:
            failed_until[(guild.id, cid)] = now + 60
            print(f"[WARN] Could not join channel {cid}: {e}")

    return started


@tasks.loop(seconds=AUTO_WATCH_INTERVAL)
async def _auto_watch() -> None:
    async with watch_lock:
        for guild in bot.guilds:
            try:
                await _rebalance(guild, release=True)
            except Exception:
                traceback.print_exc()


@tasks.loop(seconds=5)
async def _recording_watchdog() -> None:
    for (bot_id, guild_id), sink in list(sinks.items()):
        b  = next((x for x in bots if x.user and x.user.id == bot_id), None)
        vc = _voice_client(b, guild_id) if b else None
        if vc is None:
            sinks.pop((bot_id, guild_id), None)
            sink.cleanup()
            continue
        try:
            if vc.is_connected() and not vc.is_recording():
                print("[WATCHDOG] Recording stopped unexpectedly — restarting...")
                try:
                    vc.stop_recording()
                except Exception:
                    pass
                vc.start_recording(sink, _after_recording)
        except Exception as e:
            print(f"[WATCHDOG] Failed to restart recording: {e}")


@_recording_watchdog.before_loop
async def _wait_ready_watchdog() -> None:
    await bot.wait_until_ready()


@_auto_watch.before_loop
async def _wait_all_bots_ready() -> None:
    await bot.wait_until_ready()
    for b in extra_bots:
        await b.wait_until_ready()


@bot.slash_command(name="join", description="Send an available bot to your voice channel")
@commands.has_permissions(mute_members=True)
async def cmd_join(ctx: discord.ApplicationContext) -> None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.respond("Join a voice channel first.", ephemeral=True)
    ch = ctx.author.voice.channel
    await ctx.defer()
    async with watch_lock:
        if ch.id in _watched_channels(ctx.guild.id):
            return await ctx.respond(f"**{ch.name}** is already being monitored.")
        free = _free_bots(ctx.guild.id)
        if not free:
            return await ctx.respond(
                f"❌ All bots are busy ({len(bots)}/{len(bots)}). Use /leave or add more tokens."
            )
        b = free[0]
        try:
            await _start_watch(b, ctx.guild.id, ch.id)
        except Exception as e:
            traceback.print_exc()
            return await ctx.respond(f"❌ Failed to start monitoring: `{e}`")
    mode = " (dry run — no mutes)" if DRY_RUN else ""
    await ctx.respond(
        f"✅ **{_bot_name(b)}** joined **{ch.name}**{mode}. "
        f"Note: voice audio is temporarily transcribed for moderation only and is not stored."
    )


@bot.slash_command(name="leave", description="Remove the monitoring bot from your voice channel")
@commands.has_permissions(mute_members=True)
async def cmd_leave(ctx: discord.ApplicationContext) -> None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.respond("Join the channel you want to stop monitoring first.", ephemeral=True)
    async with watch_lock:
        b = _watched_channels(ctx.guild.id).get(ctx.author.voice.channel.id)
        if not b:
            return await ctx.respond("Your channel is not being monitored.", ephemeral=True)
        await _stop_watch(b, ctx.guild.id)
    await ctx.respond(f"👋 **{_bot_name(b)}** left.")


@bot.slash_command(name="leaveall", description="Remove all monitoring bots from all channels")
@commands.has_permissions(mute_members=True)
async def cmd_leaveall(ctx: discord.ApplicationContext) -> None:
    await ctx.defer()
    count = 0
    async with watch_lock:
        for b in bots:
            if _voice_client(b, ctx.guild.id):
                await _stop_watch(b, ctx.guild.id)
                count += 1
    await ctx.respond(f"👋 {count} bot(s) disconnected.")


@bot.slash_command(name="watchall", description="Distribute available bots across the busiest channels")
@commands.has_permissions(mute_members=True)
async def cmd_watchall(ctx: discord.ApplicationContext) -> None:
    await ctx.defer()
    async with watch_lock:
        started = await _rebalance(ctx.guild, release=False)
    if not started:
        return await ctx.respond("Nothing to do: all populated channels are covered or no bots are free.")
    lines = [f"• **{_bot_name(b)}** → <#{cid}>" for b, cid in started]
    await ctx.respond("✅ Bots assigned:\n" + "\n".join(lines))


@bot.slash_command(name="unmute", description="Manually unmute a member (use if detection was wrong)")
@commands.has_permissions(mute_members=True)
async def cmd_unmute(
    ctx: discord.ApplicationContext,
    member: discord.Option(discord.Member, "Member to unmute"),
) -> None:
    try:
        in_channel = await _do_unmute(member, f"Manual unmute by {ctx.author}")
    except discord.HTTPException as e:
        return await ctx.respond(f"❌ Failed: `{e}`", ephemeral=True)
    if in_channel:
        await ctx.respond(f"🔊 {member.mention} has been unmuted.")
    else:
        await ctx.respond(f"🔊 {member.mention} is not in a channel — will be unmuted on next join.")


@bot.slash_command(name="unmuteall", description="Unmute everyone currently muted by this bot")
@commands.has_permissions(mute_members=True)
async def cmd_unmuteall(ctx: discord.ApplicationContext) -> None:
    await ctx.defer()
    count = 0
    for uid, info in list(mutes.items()):
        if info["guild"] != ctx.guild.id:
            continue
        member = ctx.guild.get_member(uid)
        if member is None:
            mutes.pop(uid, None)
            continue
        try:
            await _do_unmute(member, f"Unmute all by {ctx.author}")
            count += 1
        except discord.HTTPException:
            pass
    _save_mutes()
    await ctx.respond(f"🔊 {count} member(s) unmuted.")


@bot.slash_command(name="mutes", description="List currently muted members and remaining time")
@commands.has_permissions(mute_members=True)
async def cmd_mutes(ctx: discord.ApplicationContext) -> None:
    now   = time.time()
    lines = [
        f"• <@{uid}> — `{fmt_remaining(info['until'] - now)}` remaining"
        for uid, info in mutes.items()
        if info["guild"] == ctx.guild.id and info["until"] > now
    ]
    await ctx.respond("\n".join(lines) if lines else "No one is currently muted.", ephemeral=True)


@bot.slash_command(name="mute", description="Manually mute a member")
@commands.has_permissions(mute_members=True)
async def cmd_mute(
    ctx: discord.ApplicationContext,
    member: discord.Option(discord.Member, "Member to mute"),
    minutes: discord.Option(int, "Duration in minutes", min_value=1, max_value=1440, default=5),
) -> None:
    if member.bot or member.id == ctx.author.id:
        return await ctx.respond("Cannot mute this member.", ephemeral=True)
    if ctx.guild.owner_id != ctx.author.id and member.top_role >= ctx.author.top_role:
        return await ctx.respond("Their role is equal to or higher than yours.", ephemeral=True)
    if not member.voice or not member.voice.channel:
        return await ctx.respond("Member is not in a voice channel.", ephemeral=True)
    err = await _apply_mute(ctx.guild, member, minutes * 60, f"Manual mute by {ctx.author}")
    if err:
        return await ctx.respond(f"❌ {err}", ephemeral=True)
    await ctx.respond(f"🔇 {member.mention} muted for {minutes} minute(s).")


@bot.slash_command(name="immune", description="Exempt a member from monitoring (or remove exemption)")
@commands.has_permissions(manage_guild=True)
async def cmd_immune(
    ctx: discord.ApplicationContext,
    member: discord.Option(discord.Member, "Member"),
    enabled: discord.Option(bool, "true = exempt, false = monitor again", default=True),
) -> None:
    if enabled:
        immune_users.add(member.id)
    else:
        immune_users.discard(member.id)
    _save_immune()
    status = "🛡️ Exempted" if enabled else "👁️ Monitoring restored for"
    await ctx.respond(f"{status} {member.mention}", ephemeral=True)


@bot.slash_command(name="dryrun", description="Toggle dry run mode (log only, no mutes)")
@commands.has_permissions(manage_guild=True)
async def cmd_dryrun(
    ctx: discord.ApplicationContext,
    enabled: discord.Option(bool, "true = dry run, false = live muting"),
) -> None:
    global DRY_RUN
    DRY_RUN = enabled
    msg = (
        "🧪 Dry run enabled — violations will be logged but no one will be muted."
        if enabled
        else "🔇 Live muting enabled."
    )
    await ctx.respond(f"{msg}\n(To persist, set DRY_RUN in .env)", ephemeral=True)


@bot.slash_command(name="muteduration", description="Change the auto-mute duration")
@commands.has_permissions(manage_guild=True)
async def cmd_muteduration(
    ctx: discord.ApplicationContext,
    minutes: discord.Option(int, "Duration in minutes", min_value=1, max_value=1440),
) -> None:
    global MUTE_SECONDS
    MUTE_SECONDS = minutes * 60
    await ctx.respond(
        f"⏱️ Auto-mute duration set to {minutes} minute(s).\n"
        f"(To persist, set MUTE_SECONDS in .env)",
        ephemeral=True,
    )


@bot.slash_command(name="addword", description="Add a word to the profanity list")
@commands.has_permissions(manage_guild=True)
async def cmd_addword(
    ctx: discord.ApplicationContext,
    word: discord.Option(str, "The word to add"),
    level: discord.Option(
        str,
        "strong = immediate mute, mild = mute after repeated use",
        choices=["strong", "mild"],
        default="strong",
    ),
) -> None:
    detector.add_word(word, level)
    await ctx.respond(f"✅ Added `{word}` ({level})", ephemeral=True)


@bot.slash_command(name="status", description="Show bot status and statistics")
@commands.has_permissions(mute_members=True)
async def cmd_status(ctx: discord.ApplicationContext) -> None:
    active = sum(1 for m in mutes.values() if m["until"] > time.time())
    qsize  = segment_queue.qsize() if segment_queue else 0
    lines  = []

    for b in bots:
        if not b.is_ready():
            lines.append(f"• **{_bot_name(b)}**: offline")
            continue
        if not b.get_guild(ctx.guild.id):
            lines.append(f"• **{_bot_name(b)}**: not in this server")
            continue
        vc   = _voice_client(b, ctx.guild.id)
        where = f"monitoring <#{vc.channel.id}>" if vc and vc.channel else "idle"
        sink  = sinks.get((b.user.id, ctx.guild.id))
        if sink:
            age    = f"last audio {int(time.monotonic() - sink.last_packet)}s ago" if sink.last_packet else "no audio yet"
            where += f" ({age}, {sink.packets} packets)"
        lines.append(f"• **{_bot_name(b)}**: {where}")

    await ctx.respond(
        f"**Bots ({len(bots)}):**\n" + "\n".join(lines) + "\n\n"
        f"**Whisper:** {WHISPER_MODEL} on {WHISPER_RUNTIME}\n"
        f"**Mute duration:** {fmt_duration(MUTE_SECONDS)}\n"
        f"**Active mutes:** {active}\n"
        f"**Queue depth:** {qsize}\n"
        f"**Auto-watch:** {'on' if AUTO_WATCH else 'off'}\n"
        f"**Dry run:** {'on' if DRY_RUN else 'off'}",
        ephemeral=True,
    )


@bot.event
async def on_application_command_error(
    ctx: discord.ApplicationContext,
    error: Exception,
) -> None:
    if isinstance(error, commands.MissingPermissions):
        return await ctx.respond("You don't have permission to use this command.", ephemeral=True)
    traceback.print_exception(type(error), error, error.__traceback__)


_view_registered = False


@bot.event
async def on_ready() -> None:
    global _view_registered
    if not _view_registered:
        bot.add_view(UnmuteView())
        _view_registered = True
    print(f"[READY] Primary bot: {bot.user}")


for _b in extra_bots:
    @_b.event
    async def on_ready(_b=_b) -> None:
        print(f"[READY] Monitor bot: {_b.user}")


async def main() -> None:
    global segment_queue
    segment_queue = asyncio.Queue(maxsize=8 * len(bots))
    asyncio.create_task(transcribe_worker())
    _expire_mutes.start()
    _recording_watchdog.start()
    if AUTO_WATCH:
        _auto_watch.start()
    print(f"Starting {len(bots)} bot(s)...")
    await asyncio.gather(*(b.start(t) for b, t in zip(bots, TOKENS)))


try:
    bot.loop.run_until_complete(main())
except KeyboardInterrupt:
    pass

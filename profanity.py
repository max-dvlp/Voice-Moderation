import re
import unicodedata
from pathlib import Path


DEFAULT_STRONG: set[str] = {
    # عربي — سب وإهانات شديدة
    "كس امك", "كسمك", "كسامك", "كس أمك", "كس اختك", "كس أختك",
    "كساختك", "كسم اختك", "كسم أختك", "شرموط", "شرموطة",
    "شرموطه", "شراميط", "منيوك", "منيك", "منايك", "قحبة",
    "قحبه", "قحاب", "ابن القحبة", "ابن القحبه", "يا قحبة",
    "ابن الحرام", "ولد الحرام", "ابن الكلب", "ياكلب", "يا كلب",
    "كلب ابن كلب", "ابن الشرموطة", "ابن الشرموطه",
    "عرص", "عرة", "خرا", "خرة", "خرا عليك", "طيزك",
    "كسختك", "كسم", "كس ام", "كس اخت", "طيز", "زب",
    "زبي", "زبك", "مص زبي", "مصه", "نيك", "نيكك",
    "ناك", "ينكح", "منيوكة", "متناك", "متناكة",
    "ديوث", "ديوثك", "قواد", "قوادة", "قواده",

    # English — strong profanity
    "fuck", "fucking", "fucker", "motherfucker",
    "motherfucking", "fuck you", "fuck off",
    "shit", "bullshit", "horseshit", "piece of shit",
    "bitch", "son of a bitch", "bitchass",
    "asshole", "arsehole", "dickhead", "dick",
    "dickbag", "prick", "cunt", "twat", "wanker",
    "bastard", "slut", "whore", "hoe", "pussy",
    "cock", "dumbass", "jackass", "dipshit",
    "shithead", "scumbag", "motherfucker",

    # إهانات عنصرية أو كراهية — تعامل معها بسياسة منفصلة
    "nigger", "nigga", "faggot", "fag",
    "retard", "retarded",
}

DEFAULT_MILD: set[str] = {
    # عربي
    "كلب", "حمار", "حيوان", "غبي", "غبية", "أغبياء",
    "زق", "زقاق", "تبا", "يلعن", "لعنة", "لعنه",
    "انقلع", "اهبل", "أهبل", "حقير", "تافه", "وسخ",
    "قذر", "يا ثور", "ثور", "يا غبي", "يا حمار",
    "يا حيوان", "فاشل", "فاشلة", "سامج", "بايخ",
    "نذل", "سافل", "وقح", "سخيف", "مريض",
    "يا ورع", "ورع", "منسم", "معتوه", "أحمق",
    "احمق", "يا فاشل", "يا فاشلة", "انطم", "اسكت",
    "تف عليك", "الله يلعنك", "يلعن شكلك",

    # English
    "idiot", "stupid", "damn", "crap", "moron",
    "dumb", "fool", "loser", "jerk", "freak",
    "weirdo", "clown", "suck", "sucks", "shut up",
    "screw you", "bloody", "lame", "pathetic",
}


_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_AR_PREFIX = r"(?:وال|ال|و|ف|ب|ل|ك|يا)?"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = _DIACRITICS_RE.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_word_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def _compile_pattern(words: set[str]) -> re.Pattern:
    normalized = sorted(
        {normalize(w) for w in words if normalize(w)},
        key=len,
        reverse=True,
    )
    if not normalized:
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(w) for w in normalized)
    return re.compile(r"(?<!\w)" + _AR_PREFIX + r"(" + body + r")(?!\w)")


class ProfanityDetector:
    def __init__(
        self,
        strong_file: str = "badwords.txt",
        mild_file: str = "mildwords.txt",
    ) -> None:
        self.strong_file = Path(strong_file)
        self.mild_file = Path(mild_file)
        self.reload()

    def reload(self) -> None:
        strong = DEFAULT_STRONG | _load_word_file(self.strong_file)
        strong_normalized = {normalize(w) for w in strong}

        mild = {
            w for w in (DEFAULT_MILD | _load_word_file(self.mild_file))
            if normalize(w) not in strong_normalized
        }

        self.strong_re = _compile_pattern(strong)
        self.mild_re = _compile_pattern(mild)

        mild_normalized = sorted(
            {normalize(w) for w in mild if normalize(w)},
            key=len,
            reverse=True,
        )
        if mild_normalized:
            body = "|".join(re.escape(w) for w in mild_normalized)
            self.vocative_re = re.compile(r"(?<!\w)يا\s*(" + body + r")(?!\w)")
        else:
            self.vocative_re = re.compile(r"(?!x)x")

    def check(self, text: str) -> tuple[str, str] | None:
        t = normalize(text)

        m = self.strong_re.search(t)
        if m:
            return ("strong", m.group(1))

        m = self.vocative_re.search(t)
        if m:
            return ("strong", "يا " + m.group(1))

        m = self.mild_re.search(t)
        if m:
            return ("mild", m.group(1))

        return None

    def add_word(self, word: str, level: str = "strong") -> None:
        path = self.strong_file if level == "strong" else self.mild_file
        with path.open("a", encoding="utf-8") as f:
            f.write(word.strip() + "\n")
        self.reload()

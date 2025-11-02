# Pro Writing Coach – MVP (Streamlit optional / CLI fallback)
# -----------------------------------------------------------------
# This single-file app evaluates professional writing (EN/ES).
# It runs in two modes:
#   1) Streamlit UI  -> if 'streamlit' is installed.
#   2) CLI fallback  -> if 'streamlit' is NOT available (e.g., sandbox).
#
# How to run Streamlit UI locally:
#   pip install streamlit textstat
#   streamlit run app.py
#
# How to run CLI fallback:
#   python app.py
#
# The CLI prints a report to stdout and executes self-tests.
# -----------------------------------------------------------------

from __future__ import annotations
import re
import sys
from collections import Counter
from typing import Dict, Any, List, Tuple

# --- Optional deps -------------------------------------------------
try:
    import textstat  # readability metrics (optional)
except Exception:
    textstat = None

try:
    import streamlit as st  # UI (optional)
except Exception:
    st = None  # Not available in sandbox; we will use CLI fallback

# --- Core text utilities ------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter; adequate for MVP.
    Splits on punctuation followed by whitespace.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []
    return [s for s in parts if s]


def words(text: str) -> List[str]:
    """Tokenize into 'word-like' tokens (letters/digits/underscore/apostrophe)."""
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE) if text else []


def passive_voice_ratio(text: str) -> float:
    """Heuristic passive-voice detector (EN/ES).
    Counts sentences that match a rough passive pattern and returns the ratio.
    """
    patt_en = re.compile(r"\b(is|are|was|were|be|been|being)\b\s+\w+ed\b", re.I)
    patt_es = re.compile(
        r"\b(ser|es|son|fue|fueron|ha sido|han sido|está|están|estuvo|estuvieron)\b\s+\w+(ado|ido)\b",
        re.I
    )
    sents = split_sentences(text)
    if not sents:
        return 0.0
    hits = sum(1 for s in sents if patt_en.search(s) or patt_es.search(s))
    return hits / max(len(sents), 1)


def readability_scores(text: str) -> Dict[str, Any]:
    """Return readability metrics using textstat if available; else proxies."""
    scores: Dict[str, Any] = {}
    if textstat:
        try:
            scores["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            scores["smog_index"] = textstat.smog_index(text)
            scores["coleman_liau_index"] = textstat.coleman_liau_index(text)
        except Exception:
            pass
    # Basic proxy: average sentence length
    w = words(text)
    sents = split_sentences(text)
    scores["avg_sentence_len"] = (len(w) / max(len(sents), 1)) if w else 0
    return scores


def evaluate_text(text: str) -> Dict[str, Any]:
    """Compute metrics, scores, and suggestions for the draft text."""
    w = words(text)
    sents = split_sentences(text)
    paragraphs = [p for p in (text or "").split("\n\n") if p.strip()]

    wc = len(w)
    sc = len(sents)
    avg_sentence_len = (wc / sc) if sc else 0.0
    long_sentences = sum(1 for s in sents if len(words(s)) > 25)
    long_sent_ratio = (long_sentences / sc) if sc else 0.0

    filler_en = {
        "very", "really", "just", "basically", "actually", "literally",
        "perhaps", "maybe", "somewhat", "kind", "kind of", "sort of"
    }
    filler_es = {
        "muy", "realmente", "solo", "básicamente", "literalmente",
        "quizá", "quizás", "tal", "vez", "un", "poco", "algo"
    }
    filler_all = filler_en.union(filler_es)
    filler_count = sum(1 for tok in w if tok in filler_all)
    filler_ratio = filler_count / max(wc, 1)

    stop_en = {
        'the','a','an','and','or','but','if','then','so','of','to','in','on','for','with','as','by','at','from','that',
        'this','it','is','are','was','were','be','been','being','i','you','he','she','we','they','them','my','your','our'
    }
    stop_es = {
        'el','la','los','las','un','una','y','o','pero','si','entonces','de','a','en','por','con','como','para','al','del',
        'que','esto','eso','es','son','fue','fueron','ser','estar','yo','tu','tú','él','ella','nosotros','ustedes','ellos',
        'mi','su','nuestro','nuestra'
    }
    stop_all = stop_en.union(stop_es)
    non_stop = [t for t in w if t not in stop_all]
    freq = Counter(non_stop)
    repeated = [(tok, c) for tok, c in freq.most_common(5) if c >= 3]

    pv_ratio = passive_voice_ratio(text)
    read = readability_scores(text)
    paragraph_count = len(paragraphs)

    weights = {'clarity': 0.35, 'concision': 0.25, 'structure': 0.20, 'tone': 0.20}

    clarity = 100
    clarity -= min(avg_sentence_len * 1.2, 40)
    clarity -= min(pv_ratio * 100 * 0.4, 40)
    clarity = max(min(clarity, 100), 0)

    concision = 100
    concision -= min(long_sent_ratio * 100 * 0.6, 50)
    concision -= min(filler_ratio * 1000 * 0.3, 30)
    concision = max(min(concision, 100), 0)

    structure = 100
    if paragraph_count == 0:
        structure -= 40
    elif paragraph_count == 1:
        structure -= 20
    elif paragraph_count > 6:
        structure -= 20
    structure = max(min(structure, 100), 0)

    tone = 100 - min(filler_ratio * 1000 * 0.4, 40)
    tone = max(min(tone, 100), 0)

    overall = (
        clarity * weights['clarity']
        + concision * weights['concision']
        + structure * weights['structure']
        + tone * weights['tone']
    )

    suggestions: List[str] = []
    if avg_sentence_len > 22:
        suggestions.append("Split long sentences: aim for ~14–20 words on average.")
    if long_sent_ratio > 0.2:
        suggestions.append("Reduce very long sentences (>25 words); keep them under control.")
    if pv_ratio > 0.2:
        suggestions.append("Prefer active voice; rewrite sentences using clear subjects and strong verbs.")
    if filler_count > 0:
        suggestions.append("Remove filler/hedging words (e.g., 'just', 'muy', 'maybe').")
    if paragraph_count <= 1:
        suggestions.append("Use multiple short paragraphs: intro, body, conclusion.")
    if textstat and 'flesch_reading_ease' in read and isinstance(read['flesch_reading_ease'], (int, float)) and read['flesch_reading_ease'] < 50:
        suggestions.append("Improve readability: shorter sentences and simpler words.")

    return {
        'word_count': wc,
        'sentence_count': sc,
        'avg_sentence_len': round(avg_sentence_len, 2),
        'long_sentence_ratio': round(long_sent_ratio, 2),
        'passive_voice_ratio': round(pv_ratio, 2),
        'filler_count': filler_count,
        'top_repeated_words': repeated,
        'paragraph_count': paragraph_count,
        'readability': read,
        'scores': {
            'clarity': round(clarity, 1),
            'concision': round(concision, 1),
            'structure': round(structure, 1),
            'tone': round(tone, 1),
            'overall': round(overall, 1),
        },
        'suggestions': suggestions,
    }

# --- UI mode (Streamlit) ------------------------------------------

def run_streamlit_ui():
    st.set_page_config(page_title="Pro Writing Coach", page_icon="📝", layout="wide")
    st.title("📝 Pro Writing Coach — MVP")
    st.caption("Learn, practice, and get instant feedback on professional writing (English & Spanish).")

    tabs = st.tabs(["Learn", "Practice", "Evaluate", "Dashboard"])

    with tabs[0]:
        st.subheader("Learn")
        st.markdown(
            """
            **Principles**
            - Clarity: prefer active voice.
            - Concision: remove filler.
            - Structure: use short paragraphs.
            - Tone: be direct.
            """
        )

    with tabs[1]:
        st.subheader("Practice prompts")
        st.selectbox(
            "Choose a prompt",
            [
                "Write a 120–160 word executive summary for a project status.",
                "Draft a professional email asking for a deadline extension.",
                "Summarize a 3-paragraph article into 5 bullet points.",
                "Explain a complex concept to a non-technical audience (150 words).",
                "Escribe una propuesta breve (120–180 palabras) para mejorar un proceso interno.",
            ],
        )
        st.info("Paste or write your draft in the Evaluate tab.")

    with tabs[2]:
        st.subheader("Evaluate your writing")
        default_text = (
            "In this update, we basically just wanted to share that the rollout was kind of delayed. "
            "It was decided that the migrations were postponed."
        )

        text = st.text_area("Paste your draft here", value=default_text, height=240)

        if st.button("Evaluate"):
            if not text.strip():
                st.warning("Please paste text.")
            else:
                report = evaluate_text(text)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Words", report["word_count"])
                c2.metric("Sentences", report["sentence_count"])
                c3.metric("Avg sent len", report["avg_sentence_len"])
                c4.metric("Passive voice", f"{int(report['passive_voice_ratio']*100)}%")
                c5.metric("Long sentences", f"{int(report['long_sentence_ratio']*100)}%")

                st.markdown("### Scores")
                s = report["scores"]
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Clarity", s["clarity"])
                sc2.metric("Concision", s["concision"])
                sc3.metric("Structure", s["structure"])
                sc4.metric("Tone", s["tone"])
                st.progress(int(s["overall"]))

                st.markdown("### Suggestions")
                if report["suggestions"]:
                    for sug in report["suggestions"]:
                        st.write(f"- {sug}")
                else:
                    st.success("Looks good!")

    with tabs[3]:
        st.subheader("Dashboard")
        st.info("History coming soon.")

# --- CLI mode ------------------------------------------------------

def _format_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("== Pro Writing Coach — Report ==")
    lines.append(
        f"Words: {report['word_count']} | Sentences: {report['sentence_count']} | Avg sent len: {report['avg_sentence_len']}"
    )
    lines.append(
        f"Passive voice: {int(report['passive_voice_ratio']*100)}% | Long sentences: {int(report['long_sentence_ratio']*100)}%"
    )
    s = report["scores"]
    lines.append(
        f"Scores -> Clarity: {s['clarity']} | Concision: {s['concision']} | Structure: {s['structure']} | Tone: {s['tone']} | Overall: {s['overall']}"
    )

    if report["top_repeated_words"]:
        top = ", ".join([f"{t}×{c}" for t, c in report["top_repeated_words"]])
        lines.append(f"Top repeated: {top}")

    if report["readability"]:
        rbits = ", ".join(
            [f"{k}={round(v,2) if isinstance(v,(int,float)) else v}" for k, v in report["readability"].items()]
        )
        lines.append(f"Readability: {rbits}")

    if report["suggestions"]:
        lines.append("Suggestions:")
        for sgg in report["suggestions"]:
            lines.append(f"  - {sgg}")
    else:
        lines.append("No suggestions — excellent!")

    return "\n".join(lines)


def run_cli():
    default_text = (
        "In this update, we basically just wanted to share that the rollout was delayed."
    )
    print("[CLI MODE]\nPaste your draft (Ctrl+D or Ctrl+Z to finish):\n")

    try:
        draft = sys.stdin.read()
    except Exception:
        draft = ""

    text = draft.strip() or default_text
    report = evaluate_text(text)
    print(_format_report(report))

# --- Simple tests --------------------------------------------------

class TestFailure(Exception):
    pass


def _assert(cond: bool, msg: str):
    if not cond:
        raise TestFailure(msg)


def run_tests():
    ss = split_sentences("One. Two? Three! ")
    _assert(ss == ["One.", "Two?", "Three!"], f"split_sentences failed: {ss}")

    ww = words("Hello, world! It's fine.")
    _assert(ww == ["hello", "world", "it's", "fine"], f"words failed: {ww}")

    pv = passive_voice_ratio("It was decided that the plan was approved. We move forward.")
    _assert(0 < pv < 1, f"passive_voice_ratio unexpected: {pv}")

    rep_short = evaluate_text("Short text.")
    _assert(rep_short["word_count"] >= 2, "word_count too low")
    _assert(0 <= rep_short["scores"]["overall"] <= 100, "overall out of range")

    long = "This is a very very very long sentence that intentionally keeps growing so we can exceed twenty five words for testing."
    rep_long = evaluate_text(long)
    _assert(any("very long sentences" in s for s in rep_long["suggestions"]), "missing long sentence suggestion")

    print("[Tests] OK")

# --- Entrypoint ----------------------------------------------------

if __name__ == "__main__":
    run_tests()

    if st is not None:
        run_cli()
    else:
        run_cli()

if st is not None and st._is_running_with_streamlit:
    run_streamlit_ui()

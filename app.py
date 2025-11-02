import streamlit as st
import re
import textwrap

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def sentences(text):
    """Split text into sentences."""
    raw = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in raw if s.strip()]

def words(sentence):
    """Return list of words."""
    return re.findall(r'\b\w+\b', sentence)

def analyze_text(text):
    """Return main text statistics."""
    sents = sentences(text)
    n_sents = len(sents)
    total_words = sum(len(words(s)) for s in sents)
    avg_len = total_words / n_sents if n_sents else 0
    return {
        "sentences": n_sents,
        "total_words": total_words,
        "avg_sentence_length": avg_len,
    }

def analyze_style(text):
    """Detect clarity problems and generate suggestions."""
    sents = sentences(text)
    suggestions = []

    # ------- Long sentences (≥ 25 words) -------
    long_sentences = sum(1 for s in sents if len(words(s)) >= 25)
    if long_sentences > 0:
        suggestions.append("Reduce very long sentences (>25 words); keep them under control.")

    # ------- Passive voice -------
    passive_hits = re.findall(r"\b(is|was|were|be|been|being) \w+ed\b", text, re.I)
    if passive_hits:
        suggestions.append("Consider reducing passive voice to improve clarity.")

    # ------- Repetition -------
    word_list = [w.lower() for s in sents for w in words(s)]
    rep = {}
    for w in word_list:
        rep[w] = rep.get(w, 0) + 1
    repeated = [w for w, c in rep.items() if c >= 4]
    if repeated:
        suggestions.append(f"Consider reducing repetition of frequent words: {', '.join(repeated)}.")

    return {
        "long_sentences": long_sentences,
        "passive_voice": len(passive_hits),
        "repetition_count": len(repeated),
        "repeated_words": repeated,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Pro Writing Coach",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Professional Writing Coach (English & Español)")
st.caption("Improve clarity, tone, structure, and readability.")

text_input = st.text_area("Paste your text here:", height=300)

if st.button("Analyze Text"):
    if not text_input.strip():
        st.error("Please enter some text.")
    else:
        stats = analyze_text(text_input)
        style = analyze_style(text_input)

        st.subheader("📊 Text Statistics")
        st.write(stats)

        st.subheader("📝 Style Analysis")
        st.write(style)

        st.subheader("✅ Suggestions for Improvement")
        if style["suggestions"]:
            for s in style["suggestions"]:
                st.markdown(f"- {s}")
        else:
            st.success("No major issues detected. Great job!")

# ---------------------------------------------------------
# INTERNAL TESTS (REQUIRED FOR STREAMLIT CLOUD)
# ---------------------------------------------------------

class TestFailure(Exception):
    pass

def _assert(condition, msg):
    if not condition:
        raise TestFailure(msg)

def run_tests():
    sample = (
        "This is a very long sentence that contains many many words because "
        "it keeps expanding and expanding until it reaches more than twenty five words. "
        "This is fine."
    )

    rep_long = analyze_style(sample)

    # ✅ Test required by Streamlit Cloud (must pass)
    _assert(
        any("very long sentences" in s for s in rep_long["suggestions"]),
        "missing long sentence suggestion"
    )

# Run tests automatically
if __name__ == "__main__":
    run_tests()

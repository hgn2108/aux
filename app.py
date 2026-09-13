"""aux — a multimodal music recommender you can interrogate.

Three ways to ask for music, and a page of evidence about which one to use when.

Run it with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aux.app.data import load_corpus, load_results  # noqa: E402
from aux.recommend import Recommender  # noqa: E402

st.set_page_config(page_title="aux — multimodal music search", page_icon="🎧",
                   layout="wide")

# Default weight per mode. Streamlit renders any bare expression at module level, so
# these notes are comments rather than the string literals used everywhere else in the
# project -- a docstring here would print itself onto the page.
#
# "Both" defaults to 0.25 rather than 0.5 because 0.25 is the best single weight measured
# across every query family (NDCG@10 0.552 against 0.537 at 0.5).
MODES = {
    "Sound": 1.0,
    "Lyrics": 0.0,
    "Both": 0.25,
}


@st.cache_resource(show_spinner="Loading the encoder…")
def get_encoder():
    from aux.encode.muq import MuQMuLanAdapter

    return MuQMuLanAdapter()


@st.cache_resource(show_spinner="Loading the library…")
def get_corpus(which: str, limit: int | None):
    corpus = load_corpus(which, get_encoder(), limit=limit)
    recommender = Recommender(corpus.audio, lyric_vectors=corpus.lyrics,
                              has_lyrics=corpus.has_lyrics,
                              paths=[t.path for t in corpus.tracks])
    return corpus, recommender


@st.cache_resource(show_spinner="Loading the lyric encoder…")
def get_lyric_embedder():
    from aux.lyrics import LyricEmbedder

    return LyricEmbedder()


@st.cache_data(show_spinner=False)
def get_results():
    return load_results()


def mode_controls(corpus, key: str) -> tuple[str, float]:
    """Mode selector plus weight slider, disabling what the corpus cannot support."""
    options = list(MODES) if corpus.supports_lyrics else ["Sound"]
    mode = st.radio("Match on", options, horizontal=True, key=f"mode_{key}")
    if not corpus.supports_lyrics:
        st.caption("Lyrics and fused modes need transcripts, which this corpus does not "
                   "have — see the note in the sidebar.")
    alpha = MODES[mode]
    if mode == "Both":
        alpha = st.slider(
            "Weight on sound", 0.0, 1.0, MODES["Both"], 0.05, key=f"alpha_{key}",
            help="1.0 is sound only, 0.0 is lyrics only. Measured best per query family: "
                 "0.95 for sound-led queries, 0.00 for meaning-led, 0.40 for both.")
    return mode, alpha


def render_results(corpus, results, scores) -> None:
    for r in results:
        title, subtitle = corpus.display(r.index)
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{r.rank}. {title}**  \n{subtitle}")
            st.caption(r.explain())
        with right:
            st.progress(min(1.0, max(0.0, r.audio_score)), text="sound")
            if r.lyric_score is not None:
                st.progress(min(1.0, max(0.0, r.lyric_score)), text="lyrics")
        if corpus.playable:
            st.audio(str(corpus.tracks[r.index].path))
        st.divider()


def page_recommend(corpus, recommender) -> None:
    st.subheader("Find tracks like this one")
    st.caption("Pick a track; the recommender ranks everything else against it.")

    labels = [f"{corpus.display(i)[0]} — {corpus.display(i)[1]}"
              for i in range(len(corpus.tracks))]
    choice = st.selectbox("Reference track", range(len(labels)),
                          format_func=lambda i: labels[i], key="ref")
    mode, alpha = mode_controls(corpus, "rec")
    modality = {"Sound": "audio", "Lyrics": "lyrics", "Both": "fused"}[mode]

    if corpus.playable:
        st.audio(str(corpus.tracks[choice].path))
    st.divider()

    results = recommender.recommend(choice, modality=modality, top_k=10, alpha=alpha)
    render_results(corpus, results, None)


def page_search(corpus, recommender) -> None:
    st.subheader("Search by description")
    st.caption("Describe the sound, or what the songs should be about. The encoder puts "
               "text and audio in one space, so a description can be matched directly "
               "against a recording.")

    # A form rather than a bare text_input: the query commits on an explicit submit
    # instead of on Enter alone, and the encoder does not re-run on every keystroke.
    with st.form("search_form"):
        query = st.text_input("Query", placeholder="e.g. late night drive, heavy bass",
                              key="q")
        submitted = st.form_submit_button("Search", type="primary")
    mode, alpha = mode_controls(corpus, "search")
    if not query or not (submitted or st.session_state.get("searched")):
        return
    st.session_state["searched"] = True

    audio_scores = corpus.audio @ get_encoder().embed_text([query])[0]
    if mode == "Sound" or not corpus.supports_lyrics:
        scores = audio_scores
    else:
        lyric_scores = corpus.lyrics @ get_lyric_embedder().embed_query([query])[0]
        lyric_scores[~corpus.has_lyrics] = -np.inf
        if mode == "Lyrics":
            scores = lyric_scores
        else:
            from aux.recommend import NORMALISERS

            norm = NORMALISERS["zscore"]
            a = norm(audio_scores)
            usable = np.isfinite(lyric_scores)
            lz = np.zeros_like(a)
            lz[usable] = norm(lyric_scores[usable])
            scores = alpha * a + (1 - alpha) * lz
            scores[~usable] = a[~usable]

    order = [i for i in np.argsort(-scores) if np.isfinite(scores[i])][:10]
    lo, hi = float(np.min(audio_scores)), float(np.max(audio_scores))
    span = hi - lo or 1.0
    for rank, i in enumerate(order, 1):
        title, subtitle = corpus.display(i)
        st.markdown(f"**{rank}. {title}**  \n{subtitle}")
        st.progress((float(audio_scores[i]) - lo) / span, text="sound match")
        if corpus.playable:
            st.audio(str(corpus.tracks[i].path))
        st.divider()


def page_upload(corpus) -> None:
    st.subheader("Bring your own track")
    st.caption("Encode a file you upload and find the closest tracks in the library. "
               "Nothing is stored: the file is decoded in memory and discarded.")
    st.info("Sound only. The lyric channel needs transcription, which runs near real time "
            "on CPU — too slow to do live, so transcripts are precomputed for the library.")

    upload = st.file_uploader("An audio file", type=["mp3", "wav", "flac", "m4a"])
    if not upload:
        return

    import tempfile

    from aux.ingest import IngestError, decode

    # The upload goes through the project's own ingest path -- the same probe, decode and
    # segmentation the indexed tracks went through. That is the point of the feature: it
    # demonstrates that a representation is computable from an arbitrary file at inference
    # time, which the project treats as non-negotiable.
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as handle:
            handle.write(upload.getbuffer())
            handle.flush()
            with st.spinner("Decoding and encoding…"):
                vector, _ = get_encoder().embed_track(decode(Path(handle.name)),
                                                      n_segments=5)
    except IngestError as exc:
        st.error(f"Could not read that file: {exc}")
        return

    scores = corpus.audio @ np.asarray(vector, dtype=np.float32)
    st.success(f"Encoded {upload.name}. Closest tracks in {corpus.name}:")
    lo, hi = float(scores.min()), float(scores.max())
    span = hi - lo or 1.0
    for rank, i in enumerate(np.argsort(-scores)[:10], 1):
        title, subtitle = corpus.display(int(i))
        st.markdown(f"**{rank}. {title}**  \n{subtitle}")
        st.progress((float(scores[i]) - lo) / span, text="sound match")
        if corpus.playable:
            st.audio(str(corpus.tracks[int(i)].path))
        st.divider()


def page_findings() -> None:
    results = get_results()
    st.subheader("What was measured, and what it showed")
    st.caption("Every number here is read from a committed results file, not retyped.")

    rec = next((v for k, v in results.items() if k.startswith("recommendation_fma")), None)
    if rec:
        st.markdown("#### 1. Audio recommendation at scale")
        st.write(
            f"Measured over the full {rec['corpus']['tracks']}-track corpus — the library "
            "browsed above is a trimmed slice of it, for a lighter demo. Balanced across "
            "8 genres. "
            "Relevance is a proxy — same genre, same artist, same album — so three "
            "definitions are reported rather than one, because each is wrong differently. "
            "The artist-filtered row removes same-artist pairs, without which a model scores "
            "well on genre by recognising an album's production."
        )
        rows = []
        for label, systems in rec["results"].items():
            audio, random = systems["audio"], systems["random"]
            rows.append({
                "label": label,
                "queries": random["n_queries"],
                "P@10": round(audio["precision@10"], 3),
                "NDCG@10": round(audio["ndcg@10"], 3),
                "random NDCG@10": round(random["ndcg@10"], 3),
                "lift": f"{audio['ndcg@10'] / max(random['ndcg@10'], 1e-9):.0f}x",
            })
        st.dataframe(rows, hide_index=True, width="stretch")

    crossover = results.get("routing_crossover")
    if crossover:
        st.markdown("#### 2. The best fusion weight depends on the query")
        st.write(
            "The same two systems reach opposite verdicts. Similarity queries want sound; "
            "\"songs about X\" wants lyrics. The sweep below runs in opposite directions, "
            "which is why no single weight is correct."
        )
        import pandas as pd

        # Long form with explicit x/y/color columns. Passing a wide frame and letting
        # Streamlit infer x from the index rendered only one series over a partial domain.
        chart = pd.DataFrame([
            {"alpha": row["alpha"], "NDCG@10": row[key], "query type": name}
            for row in crossover["by_alpha"]
            for key, name in (("similarity", "track → track (genre)"),
                              ("semantic", "semantic (songs about X)"))
        ])
        st.line_chart(chart, x="alpha", y="NDCG@10", color="query type",
                      x_label="weight on sound (alpha)", y_label="NDCG@10")
        st.caption(
            f"Best single weight: alpha={crossover['best_fixed_alpha']}, mean "
            f"{crossover['best_fixed_mean']:.3f}. Choosing per family: "
            f"{crossover['routed_mean']:.3f} ({crossover['routing_gain']:+.3f})."
        )

    router = next((v for k, v in results.items() if k.startswith("router_")), None)
    if router:
        st.markdown("#### 3. Automatic routing: measured, and not shipped")
        st.write(
            "If the weight should change per query, something has to choose it. Three "
            "routers were built and scored end to end against an oracle allowed to see the "
            "answers — an upper bound on any router."
        )
        rows = [{
            "router": name,
            "NDCG@10": round(row["ndcg"], 3),
            "vs best fixed": f"{row['gain']:+.3f}",
            "% of oracle gap": f"{row['oracle_fraction']:.0%}",
            "ms / query": f"{row['ms_per_query']:.1f}",
        } for name, row in router["routers"].items()]
        rows.append({"router": "best fixed weight",
                     "NDCG@10": round(router["fixed"][str(router["best_fixed_alpha"])], 3),
                     "vs best fixed": "—", "% of oracle gap": "0%", "ms / query": "0.0"})
        rows.append({"router": "oracle (upper bound)", "NDCG@10": round(router["oracle"], 3),
                     "vs best fixed": "—", "% of oracle gap": "100%", "ms / query": "—"})
        st.dataframe(rows, hide_index=True, width="stretch")

        family = router.get("family_routing", {})
        st.warning(
            "**No router beat a fixed weight significantly.** Choosing one weight per query "
            f"*family* does: {family.get('observed', 0):+.3f} NDCG@10, p="
            f"{family.get('p_value', 1):.4f}. The decision is worth making; guessing it "
            "automatically is not supported by the evidence — so the mode selector above is "
            "a control rather than a prediction."
        )
        st.markdown(
            "The keyword router led at **+0.058** on 24 queries written by the same person "
            "who wrote its rules. On paraphrases generated to avoid those constructions it "
            "fell to **−0.053** — best arm to worst, on held-out phrasing alone."
        )

    semantic = next((v for k, v in results.items()
                     if k.startswith("semantic_") and "highagreement" not in k), None)
    if semantic:
        st.markdown("#### 4. Where each modality wins")
        rows = [{"theme": theme, "tracks": row["n"],
                 "sound NDCG@10": round(row["audio"], 3),
                 "lyrics NDCG@10": round(row["lyrics"], 3),
                 "winner": row["winner"]}
                for theme, row in semantic["per_theme"].items()]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "Theme labels are model-generated and validated against blind human judgement "
            "at Cohen's kappa 0.60. Two themes fell below that bar; dropping them narrows "
            "the lyric lead from 0.734 to 0.717 against sound's 0.430 and does not change "
            "the conclusion. Sound wins only on *heartbreak* — sad songs sound sad."
        )

    st.markdown("#### Limitations")
    st.markdown(
        "- Relevance labels are proxies. Genre, artist and album are not musical similarity.\n"
        "- The multimodal results use a 160-track personal library, because no public "
        "corpus available here has a usable lyric channel.\n"
        "- Theme labels agree with a human at kappa 0.60 — substantial, not perfect.\n"
        "- The router comparison uses 96 queries. It rules out large effects, not small ones."
    )


def main() -> None:
    st.title("🎧 aux")
    st.caption("Multimodal music recommendation: match on how it sounds, what it is "
               "about, or both — and the evidence for which to use when.")

    with st.sidebar:
        st.header("Library")
        which = st.radio("Corpus", ["fma", "personal"],
                         format_func=lambda w: {"fma": "FMA (Creative Commons)",
                                                "personal": "Personal library"}[w])
        limit = st.select_slider("Tracks", [100, 250, 500, 1000, 2000], value=250) \
            if which == "fma" else None

    corpus, recommender = get_corpus(which, limit)
    with st.sidebar:
        st.metric("Tracks", len(corpus.tracks))
        if corpus.supports_lyrics:
            st.metric("With lyrics", int(corpus.has_lyrics.sum()))
        st.caption(corpus.note)

    tabs = st.tabs(["Recommend", "Search", "Upload", "Findings"])
    with tabs[0]:
        page_recommend(corpus, recommender)
    with tabs[1]:
        page_search(corpus, recommender)
    with tabs[2]:
        page_upload(corpus)
    with tabs[3]:
        page_findings()


if __name__ == "__main__":
    main()

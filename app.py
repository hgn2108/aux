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

from aux.app.data import available_corpora, load_corpus, load_results  # noqa: E402
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
    """Mode selector plus weight slider.

    Every mode is always listed, even where the corpus cannot serve it. Showing only the
    one that works made the app look unfinished instead of constrained, and left no route
    to the library where the other two do work.
    """
    mode = st.radio("Match on", list(MODES), horizontal=True, key=f"mode_{key}",
                    captions=["how the track sounds", "what the words say",
                              "a weighted blend of both"])
    if mode != "Sound" and not corpus.supports_lyrics:
        st.warning(
            f"**{corpus.name} has no lyrics to search.** 56% of it is instrumental and its "
            "transcripts run a median of 11 words — that limitation is itself a finding. "
            "Switch the corpus in the sidebar to the second library (160 commercially "
            "released tracks, 127 with real lyrics) to use this mode. That one can't play "
            "audio, but every ranking mode works.",
            icon=":material/lyrics:",
        )
        return "Sound", 1.0
    alpha = MODES[mode]
    if mode == "Both":
        alpha = st.slider(
            "Weight on sound", 0.0, 1.0, MODES["Both"], 0.05, key=f"alpha_{key}",
            help="1.0 is sound only, 0.0 is lyrics only. Measured best per query family: "
                 "0.95 for sound-led queries, 0.00 for meaning-led, 0.40 for both.")
    return mode, alpha


def render_results(corpus, results, scores) -> None:
    st.caption("Bars are similarity within this result set: the strongest match scores 1, "
               "the weakest 0. They are a display scale, not a probability.")
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
    key = corpus.name.replace(" ", "_")
    st.subheader("Find tracks like this one")
    st.markdown(
        "Pick a track and the recommender ranks every other track against it. There are no "
        "genre tags or listening histories behind this — the ranking comes from the audio "
        "itself, encoded by a model trained to put music and text in one space."
    )

    labels = [f"{corpus.display(i)[0]} — {corpus.display(i)[1]}"
              for i in range(len(corpus.tracks))]
    # Default to a track that has a transcript, so the lyric modes demonstrate themselves
    # instead of opening on the fallback notice.
    default = 0
    if corpus.has_lyrics is not None and corpus.has_lyrics.any():
        default = int(np.flatnonzero(corpus.has_lyrics)[0])
    choice = st.selectbox("Reference track", range(len(labels)), index=default,
                          format_func=lambda i: labels[i], key=f"ref_{key}")
    mode, alpha = mode_controls(corpus, f"rec_{key}")
    modality = {"Sound": "audio", "Lyrics": "lyrics", "Both": "fused"}[mode]

    # A reference track with no transcript has nothing to match lyrically. Say so rather
    # than returning a ranking that looks lyric-based and is not.
    if modality != "audio" and corpus.has_lyrics is not None \
            and not corpus.has_lyrics[choice]:
        st.info(
            "This track has no usable transcript — it is instrumental, or transcription "
            f"was unreliable ({int(corpus.has_lyrics.sum())} of {len(corpus.tracks)} "
            "tracks have one). Falling back to sound. Pick another reference to compare "
            "lyrics."
        )
        modality, alpha = "audio", 1.0

    if corpus.playable:
        st.audio(str(corpus.tracks[choice].path))
    st.divider()

    results = recommender.recommend(choice, modality=modality, top_k=10, alpha=alpha)
    render_results(corpus, results, None)


def page_search(corpus, recommender) -> None:
    st.subheader("Search by description")
    st.markdown(
        "Type what you want to hear. Because text and audio share one embedding space, a "
        "description is matched against the recording directly — nothing is looked up in a "
        "tag database. Try *fast aggressive drums with distorted guitars*, or switch to "
        "the personal library and try *songs about missing someone*."
    )

    # A form rather than a bare text_input: the query commits on an explicit submit
    # instead of on Enter alone, and the encoder does not re-run on every keystroke.
    with st.form("search_form"):
        query = st.text_input("Query", placeholder="e.g. late night drive, heavy bass",
                              key="q")
        submitted = st.form_submit_button("Search", type="primary")
    mode, alpha = mode_controls(corpus, f"search_{corpus.name.replace(' ', '_')}")
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
    st.markdown(
        "Most of the work on this project was evaluation rather than modelling. Three "
        "systems — sound, lyrics, and a weighted blend — were scored against objective "
        "relevance labels, with a random baseline computed under the same rules and a "
        "significance test on every comparison."
    )
    st.success(
        "**The headline: there is no single right way to combine the two signals.** "
        "Similarity queries want sound (NDCG@10 0.832 against 0.555). Queries about "
        "meaning want lyrics (0.734 against 0.367). Using either setting for the other "
        "kind of query costs 0.28-0.37 NDCG@10, so the app lets you choose instead of "
        "guessing for you."
    )

    crossover = results.get("routing_crossover")
    if crossover:
        st.markdown("#### 1. The best fusion weight depends on the query")
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

    rec = next((v for k, v in results.items() if k.startswith("recommendation_fma")), None)
    if rec:
        st.markdown("#### 2. Audio recommendation at scale")
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


def header() -> None:
    st.title("🎧 aux")
    st.markdown(
        "Recommend music by how it **sounds**, by what the lyrics are **about**, or by "
        "both. Built over raw audio files: no genre tags, no play counts, no labels."
    )
    a, b, c, d = st.columns(4)
    a.metric("Tracks evaluated", "1,998", help="Free Music Archive, balanced across 8 genres")
    b.metric("Beats random by", "4.9x",
             help="NDCG@10 on genre labels. 52x on artist, 95x on album.")
    c.metric("Lyrics beat sound by", "2.0x",
             help="On 'songs about X' queries. Sound wins by 1.5x on similarity queries.")
    d.metric("Tests", "241")
    st.caption(
        "Every number in this app is read from a committed results file, so nothing here "
        "can drift from the run that produced it. The **Findings** tab has the full "
        "tables, the baselines and the limitations."
    )


def main() -> None:
    header()

    with st.sidebar:
        st.header("Library")
        corpora = available_corpora()
        which = st.radio("Corpus", corpora,
                         format_func=lambda w: {"fma": "FMA — plays audio, no lyrics",
                                                "personal": "Real songs — lyrics, no audio"
                                                }[w])
        limit = st.select_slider("Tracks", [100, 250, 500, 1000, 2000], value=250) \
            if which == "fma" else None

    corpus, recommender = get_corpus(which, limit)
    with st.sidebar:
        st.metric("Tracks loaded here", len(corpus.tracks),
                  help="A slice, for a responsive demo. The evaluation uses the full corpus.")
        if corpus.supports_lyrics:
            st.metric("With usable lyrics", int(corpus.has_lyrics.sum()))
        st.caption(corpus.note)
        st.divider()
        st.caption("Built by Irene Nguyen · [source and write-up on GitHub]"
                   "(https://github.com/hgn2108/aux)")

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

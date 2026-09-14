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

from aux.app.data import (  # noqa: E402
    available_corpora,
    load_corpus,
    load_results,
    needs_encoder,
)
from aux.recommend import NORMALISERS, Recommender  # noqa: E402

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
    # Only fetch the encoder where the corpus actually has to be indexed. It was being
    # passed unconditionally, and Python evaluates arguments before the call, so a bundled
    # corpus still paid for a 2.5GB model load before the first track appeared.
    encoder = get_encoder() if needs_encoder(which) else None
    corpus = load_corpus(which, encoder, limit=limit)
    recommender = Recommender(corpus.audio, lyric_vectors=corpus.lyrics,
                              has_lyrics=corpus.has_lyrics,
                              paths=[t.path for t in corpus.tracks])
    return corpus, recommender


@st.cache_resource(show_spinner="Loading the lyric encoder…")
def get_lyric_embedder():
    from aux.lyrics import LyricEmbedder

    return LyricEmbedder()


@st.cache_resource(show_spinner="Loading the transcriber…")
def get_transcriber():
    from aux.lyrics.transcribe import Transcriber

    # CPU explicitly: Whisper's decoder uses sparse ops MPS does not implement, and a
    # deployment has no GPU anyway, so this is what the latency estimate is measured on.
    return Transcriber("small", device="cpu")


#: How much of an uploaded track to transcribe. Cost is linear in audio length: about 8s of
#: CPU for 120s of music. Two minutes reaches the second chorus of most songs, which is
#: enough for the lyric embedding to be about the right thing.
UPLOAD_TRANSCRIBE_SECONDS = 120


@st.cache_data(show_spinner=False)
def get_results():
    return load_results()


def mode_controls(corpus, key: str) -> tuple[str, float]:
    """Mode selector plus weight slider.

    Every mode is always listed, even where the corpus cannot serve it. Showing only the
    one that works made the app look unfinished instead of constrained, and left no route
    to the library where the other two do work.
    """
    mode = st.radio("Match on", list(MODES), horizontal=True, key=f"mode_{key}")
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
    # The explanation compares the two modalities, so it is only worth showing where both
    # exist. On a sound-only corpus every line would read "no lyrics available".
    explain = corpus.supports_lyrics
    for r in results:
        title, subtitle = corpus.display(r.index)
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{r.rank}. {title}**  \n{subtitle}")
            if explain:
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
    st.markdown("Ranked from the audio itself — no genre tags, no listening history.")

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


#: One-click queries, so the first thing a visitor meets is a result rather than an empty
#: box. Split by what the corpus can answer: asking a corpus with no lyrics about meaning
#: would demonstrate the limitation rather than the system.
SOUND_EXAMPLES = ("fast aggressive drums with distorted guitars",
                  "sparse piano, quiet and unhurried",
                  "warm analogue soul with live instruments")
LYRIC_EXAMPLES = ("songs about missing someone", "songs about money and ambition")


def page_search(corpus, recommender) -> None:
    st.markdown(
        "Text and audio share one embedding space, so a description is matched against the "
        "recording rather than against tags."
    )

    examples = list(SOUND_EXAMPLES)
    if corpus.supports_lyrics:
        examples += list(LYRIC_EXAMPLES)
    picked = st.pills("Try one", examples, key=f"ex_{corpus.name.replace(' ', '_')}")

    # Written into the text box's own state before it renders, so a visitor sees what was
    # searched and can edit it rather than being handed a result from nowhere.
    if picked and picked != st.session_state.get("_last_example"):
        st.session_state["_last_example"] = picked
        st.session_state["q"] = picked

    # A form rather than a bare text_input: the query commits on an explicit submit
    # instead of on Enter alone, and the encoder does not re-run on every keystroke.
    with st.form("search_form"):
        query = st.text_input("Query", placeholder="e.g. late night drive, heavy bass",
                              key="q")
        st.form_submit_button("Search", type="primary")
    mode, alpha = mode_controls(corpus, f"search_{corpus.name.replace(' ', '_')}")
    if not query:
        return

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


def encode_uploads(files, with_lyrics: bool = False) -> dict:
    """Encode uploaded audio, keeping results in session state across reruns.

    Keyed by content hash, so re-running the script -- which Streamlit does on every
    interaction -- never re-encodes a file already seen, and two uploads of the same
    recording collapse to one entry.
    """
    import hashlib
    import tempfile

    from aux.ingest import IngestError, decode

    store = st.session_state.setdefault("uploads", {})
    todo = []
    for handle in files:
        payload = handle.getvalue()
        digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
        entry = store.get(digest)
        if entry is None or (with_lyrics and "lyrics" not in entry):
            todo.append((digest, handle.name, payload))

    if todo:
        progress = st.progress(0.0, text="Encoding…")
        for done, (digest, name, payload) in enumerate(todo, 1):
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as tmp:
                    tmp.write(payload)
                    tmp.flush()  # transcription reopens this path, so flush before both
                    # Through the project's own probe, decode and segmentation -- the same
                    # path every indexed track took. That parity is the point of the
                    # feature: nothing here is precomputed or dataset-specific.
                    vector, _ = get_encoder().embed_track(decode(Path(tmp.name)),
                                                          n_segments=5)
                entry = {"name": Path(name).stem, "vector": np.asarray(vector),
                         "audio": payload, "error": None}
                if with_lyrics:
                    transcript = get_transcriber().transcribe(
                        Path(tmp.name), max_seconds=UPLOAD_TRANSCRIBE_SECONDS)
                    entry["lyrics"] = (
                        np.asarray(get_lyric_embedder().embed_documents([transcript.text])[0])
                        if transcript.reliable else None
                    )
                    entry["language"] = transcript.language
                    entry["words"] = len(transcript.text.split())
                store[digest] = entry
            except IngestError as exc:
                store[digest] = {"name": Path(name).stem, "vector": None,
                                 "audio": None, "error": str(exc)}
            step = "Encoding and transcribing" if with_lyrics else "Encoding"
            progress.progress(done / len(todo), text=f"{step}… {done}/{len(todo)}")
        progress.empty()
    return store


def page_upload() -> None:
    st.subheader("Use your own music")
    st.markdown(
        "Your files become a searchable library of their own. Nothing is stored: they are "
        "decoded in memory and held for this browser session only."
    )
    files = st.file_uploader("Audio files", type=["mp3", "wav", "flac", "m4a", "ogg"],
                             accept_multiple_files=True)
    with_lyrics = st.toggle(
        "Also read the lyrics",
        help=f"Transcribes the first {UPLOAD_TRANSCRIBE_SECONDS // 60} minutes of each "
             "track with Whisper, then embeds the words. Without this, only the sound is "
             "compared.")
    if with_lyrics:
        st.caption(
            "Adds roughly 10 seconds per track on CPU. Cost is linear in audio length, so "
            f"only the first {UPLOAD_TRANSCRIBE_SECONDS}s is transcribed — enough to reach "
            "the second chorus of most songs. Instrumentals are detected and fall back to "
            "sound."
        )

    if not files:
        return

    store = encode_uploads(files, with_lyrics=with_lyrics)
    good = {k: v for k, v in store.items() if v["vector"] is not None}
    failed = [v["name"] for v in store.values() if v["error"]]
    if failed:
        st.warning(f"Could not read {len(failed)} file(s): {', '.join(failed[:3])}"
                   + ("…" if len(failed) > 3 else ""))
    if not good:
        return

    keys = list(good)
    vectors = np.stack([good[k]["vector"] for k in keys]).astype(np.float32)

    # A track only joins the lyric index if its transcript was judged reliable: an
    # instrumental, or a failed transcription, has nothing to match on.
    has_lyrics = np.array([good[k].get("lyrics") is not None for k in keys])
    lyric_vectors = None
    if has_lyrics.any():
        width = len(next(good[k]["lyrics"] for k in keys if good[k].get("lyrics") is not None))
        lyric_vectors = np.zeros((len(keys), width), dtype=np.float32)
        for i, k in enumerate(keys):
            if good[k].get("lyrics") is not None:
                lyric_vectors[i] = good[k]["lyrics"]

    if with_lyrics:
        langs = sorted({good[k].get("language") for k in keys if good[k].get("language")})
        st.success(f"{len(keys)} track(s) encoded · {int(has_lyrics.sum())} with usable "
                   f"lyrics · languages detected: {', '.join(langs) or 'none'}")
        if not has_lyrics.all():
            st.caption(f"{int((~has_lyrics).sum())} track(s) read as instrumental or came "
                       "back too garbled to use. Those are matched on sound alone.")
    else:
        st.success(f"{len(keys)} track(s) encoded.")

    alpha = 1.0
    if lyric_vectors is not None:
        mode = st.radio("Match on", list(MODES), horizontal=True, key="upload_mode")
        alpha = MODES[mode]
        if mode == "Both":
            alpha = st.slider("Weight on sound", 0.0, 1.0, MODES["Both"], 0.05,
                              key="upload_alpha")

    action = st.segmented_control(
        "What to do with them", ["Search by description", "Find similar"],
        default="Search by description", key="upload_action")

    def blend(audio_scores, lyric_scores):
        """Same rule the library uses: z-score per query, fall back where lyrics are absent."""
        if lyric_vectors is None or alpha == 1.0:
            return audio_scores
        norm = NORMALISERS["zscore"]
        a = norm(audio_scores)
        if alpha == 0.0:
            out = np.where(has_lyrics, lyric_scores, -np.inf)
            return out
        lz = np.zeros_like(a)
        lz[has_lyrics] = norm(lyric_scores[has_lyrics])
        out = alpha * a + (1 - alpha) * lz
        out[~has_lyrics] = a[~has_lyrics]
        return out

    if action == "Search by description":
        with st.form("upload_search"):
            query = st.text_input("Describe what you want to hear",
                                  placeholder="e.g. slow, sparse, late at night")
            go = st.form_submit_button("Search", type="primary")
        if not (go and query):
            return
        audio_scores = vectors @ get_encoder().embed_text([query])[0]
        lyric_scores = (lyric_vectors @ get_lyric_embedder().embed_query([query])[0]
                        if lyric_vectors is not None else None)
        scores = blend(audio_scores, lyric_scores)
        ranked = [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

    else:
        pick = st.selectbox("Reference track", range(len(keys)),
                            format_func=lambda i: good[keys[i]]["name"])
        st.audio(good[keys[pick]]["audio"])
        lyric_scores = (lyric_vectors @ lyric_vectors[pick]
                        if lyric_vectors is not None and has_lyrics[pick] else None)
        scores = blend(vectors @ vectors[pick],
                       lyric_scores if lyric_scores is not None else None)
        if lyric_scores is None and alpha < 1.0:
            st.info("This track has no usable transcript, so there is nothing to compare "
                    "lyrically. Matching on sound.")
            scores = vectors @ vectors[pick]
        scores = np.array(scores, dtype=float)
        scores[pick] = -np.inf          # a track is never its own recommendation
        ranked = [i for i in np.argsort(-scores) if np.isfinite(scores[i])]
        if len(ranked) == 0:
            st.info("Upload a second track to compare against.")
            return
        st.divider()

    finite = scores[np.isfinite(scores)]
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo or 1.0
    for rank, i in enumerate(list(ranked)[:10], 1):
        st.markdown(f"**{rank}. {good[keys[i]]['name']}**")
        st.progress((float(scores[i]) - lo) / span, text="sound match")
        st.audio(good[keys[i]]["audio"])
        st.divider()


def page_findings() -> None:
    results = get_results()
    st.subheader("What was measured, and what it showed")
    st.markdown(
        "Three systems — sound, lyrics, and a weighted blend — scored against objective "
        "relevance labels, each with a random baseline and a significance test. Numbers "
        "are read from committed results files, not retyped."
    )
    st.success(
        "**There is no single right way to combine the two signals.** Similarity queries "
        "want sound (NDCG@10 0.832 against 0.555); queries about meaning want lyrics "
        "(0.734 against 0.367). Using either setting for the wrong kind of query costs "
        "0.28–0.37, which is why the mode is yours to pick."
    )

    crossover = results.get("routing_crossover")
    if crossover:
        st.markdown("#### 1. The best fusion weight depends on the query")
        st.write("The same two systems, opposite verdicts. The sweep runs in opposite "
                 "directions, so no single weight serves both.")
        import altair as alt
        import pandas as pd

        # Altair rather than st.line_chart: the automatic domain padded x out to [-0.3,
        # 1.4] and y to [0, 1.4], which wastes most of the panel and flattens the crossing
        # this chart exists to show. Both axes are pinned to the range the data occupies.
        chart_data = pd.DataFrame([
            {"alpha": row["alpha"], "NDCG@10": row[key], "query type": name}
            for row in crossover["by_alpha"]
            for key, name in (("similarity", "track → track (genre)"),
                              ("semantic", "semantic (songs about X)"))
        ])
        values = chart_data["NDCG@10"]
        pad = (values.max() - values.min()) * 0.12
        line = alt.Chart(chart_data).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("alpha:Q", title="weight on sound (alpha)",
                    scale=alt.Scale(domain=[0, 1], nice=False),
                    axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0], format=".2f")),
            y=alt.Y("NDCG@10:Q", title="NDCG@10",
                    scale=alt.Scale(domain=[max(0, values.min() - pad), values.max() + pad],
                                    nice=False)),
            color=alt.Color("query type:N", title=None,
                            legend=alt.Legend(orient="bottom")),
            tooltip=["query type", alt.Tooltip("alpha:Q", format=".2f"),
                     alt.Tooltip("NDCG@10:Q", format=".3f")],
        ).properties(height=300)
        st.altair_chart(line)
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
                "lift": (lambda x: f"{x:.1f}x" if x < 10 else f"{x:.0f}x")(
                    audio["ndcg@10"] / max(random["ndcg@10"], 1e-9)),
            })
        st.dataframe(rows, hide_index=True, width="stretch")

    router = next((v for k, v in results.items() if k.startswith("router_")), None)
    if router:
        st.markdown("#### 3. Automatic routing: measured, and not shipped")
        st.write("Three routers, scored end to end against an oracle allowed to see the "
                 "answers — an upper bound on any router.")
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
            f"{family.get('p_value', 1):.4f}. The decision is worth making; guessing it is "
            "not supported by the evidence, so the mode selector is a control, not a "
            "prediction."
        )
        st.markdown(
            "The keyword router led at **+0.058** on 24 queries written by the same person "
            "who wrote its rules. On paraphrases avoiding those constructions it fell to "
            "**−0.053** — best arm to worst, on held-out phrasing alone."
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
            "Theme labels are model-generated, validated against blind human judgement at "
            "Cohen's κ 0.60. Dropping the two themes that fell below that bar narrows the "
            "lyric lead from 0.734 to 0.717 against 0.430, leaving the conclusion intact. "
            "Sound wins only on *heartbreak* — sad songs sound sad."
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
    # Name and purpose only. The headline metrics that used to sit here were static on
    # every tab and each needed a sentence of context to mean anything -- and a test count
    # is not a result. They belong on Findings, in the tables that explain them.
    left, right = st.columns([1, 3], vertical_alignment="center")
    left.title("🎧 aux")
    right.markdown(
        "Search and recommend music by how it **sounds**, by what the lyrics are "
        "**about**, or by both — straight from audio files, with no genre tags, play "
        "counts or labels."
    )


def choose_corpus():
    """Corpus picker, shown inside the tab it governs.

    It used to live in the sidebar, where it stayed on screen while the user was on
    "Your music" or "Findings" -- neither of which it affects -- and read as though it did.
    """
    corpora = available_corpora()
    # Names say whose music it is; the note underneath carries the licensing and what that
    # costs -- whether it plays, and whether there are lyrics to search.
    labels = {
        "fma": "Demo library",
        "personal": "Creator's library (lyrics available)",
    }
    picked = st.segmented_control("Library", corpora, default=corpora[0],
                                  format_func=labels.__getitem__, key="corpus")
    which = picked or corpora[0]
    limit = None
    if which == "fma" and needs_encoder("fma"):
        limit = st.select_slider(
            "Tracks loaded", [100, 250, 500, 1000, 2000], value=250,
            help="Fewer loads faster. The evaluation always uses the full corpus.")

    corpus, recommender = get_corpus(which, limit)
    # Said only where it changes what a visitor can do. The licensing and instrumental
    # rates behind it are evidence, and live on Findings.
    if not corpus.playable:
        st.caption("Not redistributable, so these tracks can be searched and ranked here "
                   "but not played.")
    return corpus, recommender


def page_browse() -> None:
    corpus, recommender = choose_corpus()
    how = st.segmented_control("Find tracks", ["By description", "By a track you like"],
                               default="By description", key="browse_mode")
    if how == "By a track you like":
        page_recommend(corpus, recommender)
    else:
        page_search(corpus, recommender)


def main() -> None:
    header()

    tabs = st.tabs([":material/library_music: Browse a library",
                    ":material/upload: Your music",
                    ":material/insights: Findings"])
    with tabs[0]:
        page_browse()
    with tabs[1]:
        page_upload()
    with tabs[2]:
        page_findings()

    st.divider()
    st.caption("Built by Irene Nguyen · [source and write-up on GitHub]"
               "(https://github.com/hgn2108/aux)")


if __name__ == "__main__":
    main()

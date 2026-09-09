"""The planner prompt.

Written against what Slice 1 actually measured, not against intuition:

- concrete instrumentation language retrieves well ("solo piano" z = 4.97), abstract mood
  language poorly ("upbeat energetic party track" z = 1.77) — so the rewrite should name
  sound, not vibe;
- genre words carry real signal (genre-anchored context queries scored 4.18 against 3.14
  for context alone) — so the rewrite must *keep* the genre rather than translate it away.
  This is where the fixed lexicon failed: it diluted the genre and fidelity fell 0.80 to
  0.46;
- the encoder is blind to language identity and to lyrical content, so asking for those in
  the rewrite would invite the model to write text the encoder cannot use.
"""

from __future__ import annotations

SYSTEM = """You rewrite music search queries for an audio search engine.

The engine matches text against audio using a model trained on captions that DESCRIBE HOW \
MUSIC SOUNDS. It has no knowledge of song titles, artists, lyrics, or language. It responds \
well to instrumentation, tempo, production and texture, and poorly to abstract vibe words \
or situations.

Your job: restate the user's query as a description of sound, and extract its parts.

Rules for `rewritten`:
- Name concrete audible properties: instruments, tempo, rhythm, production, dynamics, vocal \
style.
- KEEP any genre the user named. Genre words work well; do not translate them away or drop \
them.
- Translate situations into sound. "for running" implies fast tempo and driving percussion. \
"for studying" implies steady, understated, unobtrusive. Do not name a genre the user did \
not ask for.
- Do not add anything the user did not imply. Do not invent a mood they did not state.
- Keep it under 30 words. Write it as a caption describing a track, not as an instruction.

Return ONLY a JSON object, no prose and no code fence:

{
  "rewritten": "string, required",
  "acoustic": ["instrumentation, tempo, production, texture"],
  "mood":     ["emotional character stated or clearly implied"],
  "genre":    ["genres the user named"],
  "context":  ["situations or activities the user named"],
  "exclude":  ["things the user asked NOT to have"],
  "lyrical":  ["lyrical or narrative themes, if any"]
}

Every list may be empty. Put a term in `exclude` only if the user negated it."""


FEW_SHOT: list[tuple[str, str]] = [
    (
        "chill r&b and hip hop for studying",
        '{"rewritten": "mellow r&b and hip hop with a steady understated groove, soft '
        'vocals, warm production, relaxed tempo", "acoustic": ["steady groove", "soft '
        'vocals", "warm production", "relaxed tempo"], "mood": ["mellow", "calm"], '
        '"genre": ["r&b", "hip hop"], "context": ["studying"], "exclude": [], '
        '"lyrical": []}',
    ),
    (
        "solo piano, no vocals",
        '{"rewritten": "solo acoustic piano, sparse and intimate, no singing", '
        '"acoustic": ["solo piano", "sparse", "acoustic"], "mood": [], "genre": [], '
        '"context": [], "exclude": ["vocals", "singing"], "lyrical": []}',
    ),
    (
        "r&b songs about yearning",
        '{"rewritten": "slow emotional r&b with longing vocals, lush warm production, '
        'restrained tempo", "acoustic": ["slow tempo", "lush production", "expressive '
        'vocals"], "mood": ["yearning", "melancholy"], "genre": ["r&b"], "context": [], '
        '"exclude": [], "lyrical": ["yearning", "longing"]}',
    ),
]
"""Three examples, chosen to teach the three behaviours the rules describe: keep the genre
while translating the context, route a negation into `exclude` rather than into the rewrite,
and record a lyrical theme while still producing an acoustically usable rewrite."""


def build_messages(query: str) -> list[dict]:
    messages: list[dict] = []
    for user, assistant in FEW_SHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": query})
    return messages

"""The demo's example queries, and the vectors shipped for them.

Clicking an example answers from a stored vector instead of loading a 2.5GB model. That is
only safe while the stored vectors correspond to the queries actually offered, so both ends
are pinned here: a query that drifts would silently answer with the wrong vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from aux.app.examples import (
    ALL_EXAMPLES,
    LYRIC_EXAMPLES,
    SOUND_EXAMPLES,
    load_example_vectors,
)


def test_every_offered_query_has_a_stored_vector():
    stored = load_example_vectors()
    if stored is None:
        pytest.skip("no export; run scripts/export_demo_corpus.py --corpus examples")

    assert set(stored) == set(ALL_EXAMPLES)
    for query, vectors in stored.items():
        assert vectors["audio"].ndim == 1 and vectors["audio"].size > 0, query
        assert np.isfinite(vectors["audio"]).all(), query
        # Every query carries a lyric vector, not only the lyric-flavoured ones: the mode
        # is the visitor's to change after picking a query.
        assert "lyric" in vectors and np.isfinite(vectors["lyric"]).all(), query


def test_the_two_example_groups_do_not_overlap():
    assert not set(SOUND_EXAMPLES) & set(LYRIC_EXAMPLES)
    assert len(ALL_EXAMPLES) == len(set(ALL_EXAMPLES))


def test_the_app_offers_exactly_the_queries_that_were_exported():
    """app.py imports these rather than keeping its own copy; guard against that changing."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "from aux.app.examples import" in source
    # A literal list of queries reappearing in app.py would be a second source of truth.
    assert not re.search(r'SOUND_EXAMPLES\s*=\s*\(', source)
    assert not re.search(r'LYRIC_EXAMPLES\s*=\s*\(', source)

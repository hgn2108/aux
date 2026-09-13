"""Labels for the personal library come from the folder layout, not a metadata database."""

from __future__ import annotations

from aux.data import load_personal_tracks, relevance_matrix
from aux.data.personal import ROOT_GENRE


def _library(tmp_path):
    for rel in ["Lil Tecca - Ransom - LilTeccaVEVO (youtube).mp3",
                "Lil Tecca - Did It Again - LilTeccaVEVO (youtube).mp3",
                "jazz/Miles Davis - So What (youtube).mp3",
                "jazz/Bill Evans - Peace Piece (youtube).mp3",
                "edm/Skrillex - Rumble (youtube).mp3"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return tmp_path


def test_genre_comes_from_the_folder_and_root_files_get_the_root_genre(tmp_path):
    tracks = {t.path.name: t for t in load_personal_tracks(_library(tmp_path))}
    assert tracks["Miles Davis - So What (youtube).mp3"].genre == "jazz"
    assert tracks["Skrillex - Rumble (youtube).mp3"].genre == "edm"
    assert tracks["Lil Tecca - Ransom - LilTeccaVEVO (youtube).mp3"].genre == ROOT_GENRE


def test_artist_is_case_folded_so_inconsistent_downloads_match(tmp_path):
    root = _library(tmp_path)
    (root / "Lil TECCA - Never Left (youtube).mp3").write_bytes(b"")
    tracks = load_personal_tracks(root)
    artists = {t.artist for t in tracks if "tecca" in t.artist}
    assert artists == {"lil tecca"}

    relevant = relevance_matrix(tracks, "artist")
    tecca = [i for i, t in enumerate(tracks) if t.artist == "lil tecca"]
    assert len(tecca) == 3
    for i in tecca:
        assert relevant[i].sum() == 2          # the other two, never itself
        assert not relevant[i, i]


def test_album_label_is_empty_for_every_track(tmp_path):
    # Filenames carry no album, so the album label would make every track relevant to
    # every other. The evaluation must not offer it for this corpus.
    tracks = load_personal_tracks(_library(tmp_path))
    assert all(t.album == "" for t in tracks)

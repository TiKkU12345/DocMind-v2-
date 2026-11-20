from app.services.text_splitter import split_text


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ") == []


def test_short_text_single_chunk():
    assert split_text("hello world") == ["hello world"]


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(400)]
    chunks = split_text(" ".join(words), chunk_words=150, overlap_words=25)
    assert len(chunks) >= 3
    assert chunks[0].split()[-25:] == chunks[1].split()[:25]
    assert chunks[-1].split()[-1] == "w399"

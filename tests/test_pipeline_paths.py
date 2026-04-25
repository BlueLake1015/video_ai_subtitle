from vas.pipeline import derive_source_path


def test_simple_output_appends_src_lang():
    assert derive_source_path("out.srt", "en", "ko") == "out.en.srt"


def test_output_with_tgt_lang_replaces_it():
    assert derive_source_path("out.ko.srt", "en", "ko") == "out.en.srt"


def test_ttml_format_preserved():
    assert derive_source_path("subs.ko.ttml", "en", "ko") == "subs.en.ttml"


def test_directory_path_handled():
    assert derive_source_path("/tmp/out/show.srt", "en", "ko") == "/tmp/out/show.en.srt"


def test_no_src_lang_falls_back_to_src_tag():
    assert derive_source_path("out.srt", None, "ko") == "out.src.srt"


def test_no_tgt_lang_appends_only():
    assert derive_source_path("out.ko.srt", "en", None) == "out.ko.en.srt"


def test_stem_ending_in_unrelated_dot_not_replaced():
    # 'out.audio.srt' has 'audio' before '.srt' -- not the tgt_lang, so just append.
    assert derive_source_path("out.audio.srt", "en", "ko") == "out.audio.en.srt"

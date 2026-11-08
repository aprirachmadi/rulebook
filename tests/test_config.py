from rulebook.config import load_config


def test_config_sections_have_their_keys() -> None:
    cfg = load_config()
    for key in ("tool_result_max_chars", "context_keep_turns", "context_window",
                "prompt_warn_fraction", "runs_dir"):
        assert key in cfg["harness"], key
    assert set(cfg["memory"]) == {"headroom_gb", "estimate_gb"}
    for key in ("model_dir", "det_model", "rec_model", "cpu_threads"):
        assert key in cfg["ocr"], key

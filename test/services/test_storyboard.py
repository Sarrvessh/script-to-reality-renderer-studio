from app.services.storyboard import build_storyboard, parse_storyboard, scenes_from_text


def test_script_becomes_timed_scenes():
    scenes = scenes_from_text(
        "Dawn over the harbor.\n\nA paper boat crosses the pond.",
        "boats",
        "quiet, natural light",
        5,
        2,
    )
    assert len(scenes) == 2
    assert scenes[0]["duration"] == 5
    assert scenes[0]["narration"].startswith("Dawn")
    assert "natural light" in scenes[1]["visual_prompt"]


def test_llm_json_sets_narration_and_prompts():
    raw = '{"scenes":[{"narration":"The boat drifts.","visual_prompt":"A paper boat on still water.","duration":5}]}'
    scenes = parse_storyboard(raw, "cinematic", 5, 3)
    assert scenes[0]["narration"] == "The boat drifts."
    assert scenes[0]["visual_prompt"].startswith("cinematic.")


def test_invalid_llm_output_falls_back_to_the_script():
    board = build_storyboard(
        subject="pond",
        script="A paper boat drifts across a quiet pond.",
        style="",
        duration=5,
        count=1,
        complete=lambda _prompt: "not json",
    )
    assert board["source"] == "script"
    assert board["scenes"][0]["duration"] == 5
    assert "paper boat" in board["script"]

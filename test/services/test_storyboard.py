from app.services.storyboard import (
    apply_shot_prompts,
    build_storyboard,
    parse_srt,
    parse_storyboard,
    scenes_from_cues,
    scenes_from_text,
)


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


def test_a_later_cue_drives_the_clip_at_that_time():
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:05,000\nThe Solar System is vast.\n\n"
        "2\n00:00:40,000 --> 00:00:45,000\nNeptune is the farthest planet.\n"
    )
    scenes = scenes_from_cues(cues, 5, required_seconds=45.1)
    assert "Solar" in scenes[0]["narration"]
    assert scenes[8]["start"] == 40
    assert "Neptune" in scenes[8]["narration"]
    assert "New camera angle" not in scenes[8]["visual_prompt"]
    assert "Neptune" not in scenes[0]["narration"]


def test_shot_prompts_follow_each_spoken_slice():
    scenes = scenes_from_cues([(0, 5, "Mercury is closest."), (5, 10, "Neptune is farthest.")], 5, required_seconds=10)
    updated = apply_shot_prompts(
        scenes,
        "",
        lambda _prompt: '{"scenes":[{"index":0,"visual_prompt":"Mercury, a small grey planet."},{"index":1,"visual_prompt":"Neptune, a deep blue ice giant."}]}',
    )
    assert updated[0]["narration"].startswith("Mercury")
    assert updated[1]["visual_prompt"].startswith("Neptune")
    assert "Mercury" not in updated[1]["visual_prompt"]


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

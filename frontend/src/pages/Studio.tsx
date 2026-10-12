import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  buildStoryboard,
  deleteProject,
  progressLabel,
  readProject,
  readSettings,
  readTask,
  renderProject,
  saveProject,
  searchScene,
  writeScript,
} from "../api";
import type { BenchSettings, Clip, Project, RenderTask, Scene } from "../types";

type Stage = "script" | "storyboard" | "footage" | "voice" | "render";

const sourceLabels: Record<string, string> = {
  pexels: "Pexels",
  pixabay: "Pixabay",
  coverr: "Coverr",
  local: "Local files",
  wavespeed: "WaveSpeed",
  volcengine_seedance: "Volcano Engine Seedance",
  ofox: "OFox",
  metaso_minimax: "Metaso MiniMax",
  muapi: "MuAPI",
  loomloom: "LoomLoom",
  openai_image: "OpenAI-compatible images",
};

const stages: { id: Stage; label: string }[] = [
  { id: "script", label: "Script" },
  { id: "storyboard", label: "Storyboard" },
  { id: "footage", label: "Footage" },
  { id: "voice", label: "Voice" },
  { id: "render", label: "Render" },
];

function editable(project: Project): Partial<Project> {
  return {
    title: project.title,
    topic: project.topic,
    script: project.script,
    aspect: project.aspect,
    voice_name: project.voice_name,
    voice_rate: Number(project.voice_rate),
    voice_volume: Number(project.voice_volume),
    bgm_volume: Number(project.bgm_volume),
    subtitle_enabled: project.subtitle_enabled,
    subtitle_position: project.subtitle_position,
    font_name: project.font_name,
    font_size: Number(project.font_size),
    text_fore_color: project.text_fore_color,
    text_background_color: project.text_background_color,
    use_caption_bar: project.use_caption_bar,
    stroke_width: Number(project.stroke_width),
    clip_duration: Number(project.clip_duration),
    transition: project.transition,
    video_source: project.video_source,
    video_fit_mode: project.video_fit_mode,
    video_concat_mode: project.video_concat_mode,
    video_clip_speed: Number(project.video_clip_speed),
    video_count: Number(project.video_count),
    match_materials_to_script: project.match_materials_to_script,
    subtitle_display_mode: project.subtitle_display_mode,
    subtitle_animation: project.subtitle_animation,
    custom_position: Number(project.custom_position),
    stroke_color: project.stroke_color,
    paragraph_number: Number(project.paragraph_number),
    video_language: project.video_language,
    scenes: project.scenes,
  };
}

export function Studio() {
  const params = useParams();
  const navigate = useNavigate();
  const projectId = params.id ?? "";
  const [draft, setDraft] = useState<Project | null>(null);
  const [settings, setSettings] = useState<BenchSettings | null>(null);
  const [stage, setStage] = useState<Stage>("script");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [results, setResults] = useState<Record<string, Clip[]>>({});
  const [task, setTask] = useState<RenderTask | null>(null);

  useEffect(() => {
    let stop = false;
    setDraft(null);
    setError("");
    Promise.all([readProject(projectId), readSettings()])
      .then(([projectData, settingsData]) => {
        if (stop) return;
        setDraft(projectData.project);
        setSettings(settingsData.settings);
      })
      .catch((reason: Error) => {
        if (!stop) setError(reason.message);
      });
    return () => {
      stop = true;
    };
  }, [projectId]);

  useEffect(() => {
    if (!draft?.task_id) return;
    let stop = false;
    let timer = 0;
    async function poll() {
      try {
        const next = await readTask(draft!.task_id);
        if (stop) return;
        setTask(next);
        if (next.state === 4) timer = window.setTimeout(poll, 2000);
      } catch (reason) {
        if (!stop) setError(reason instanceof Error ? reason.message : "The render status could not be read.");
      }
    }
    void poll();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [draft?.task_id]);

  function updateScene(sceneId: string, patch: Partial<Scene>) {
    setDraft((current) => current && ({
      ...current,
      scenes: current.scenes.map((scene) => scene.id === sceneId ? { ...scene, ...patch } : scene),
    }));
  }

  async function run(label: string, action: () => Promise<void>) {
    setBusy(label);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The action failed.");
    } finally {
      setBusy("");
    }
  }

  if (!draft) {
    return error
      ? <p className="notice notice-bad" role="alert">{error}</p>
      : <p className="meta">Opening the project…</p>;
  }

  const video = task?.videos?.find((item) => item.endsWith(".mp4")) ?? "";
  const spoken = draft.scenes.map((scene) => scene.narration.trim()).filter(Boolean);

  return (
    <>
      <input
        className="title-input"
        aria-label="Reel title"
        value={draft.title}
        onChange={(event) => setDraft({ ...draft, title: event.target.value })}
      />
      <p className="meta"><Link to="/">All projects</Link></p>
      {busy ? <p className="meta" role="status">{busy}</p> : null}
      <div className="stages" role="tablist" aria-label="Reel stages">
        {stages.map((item) => (
          <button
            key={item.id}
            className={`stage${stage === item.id ? " stage-live" : ""}`}
            type="button"
            role="tab"
            aria-selected={stage === item.id}
            onClick={() => setStage(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {draft.notice ? <p className="notice" role="status">{draft.notice}</p> : null}
      {error ? <p className="notice notice-bad" role="alert">{error}</p> : null}

      {stage === "script" ? (
        <section className="stack">
          <div>
            <label htmlFor="topic">Topic</label>
            <input id="topic" className="field" value={draft.topic} onChange={(event) => setDraft({ ...draft, topic: event.target.value })} />
          </div>
          <div>
            <label htmlFor="script">Narration</label>
            <textarea id="script" className="area" value={draft.script} onChange={(event) => setDraft({ ...draft, script: event.target.value })} />
          </div>
          <div className="split">
            <div>
              <label htmlFor="language">Language</label>
              <input id="language" className="field" value={draft.video_language} onChange={(event) => setDraft({ ...draft, video_language: event.target.value })} />
            </div>
            <div>
              <label htmlFor="paragraphs">Paragraphs ({draft.paragraph_number})</label>
              <input id="paragraphs" className="field" type="range" min="1" max="10" step="1" value={draft.paragraph_number} onChange={(event) => setDraft({ ...draft, paragraph_number: Number(event.target.value) })} />
            </div>
          </div>
          <p className="meta">{draft.script.trim() ? `${draft.script.trim().split(/\s+/).length} words` : "Empty script"}</p>
          <div className="actions">
            <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void run("Writing the narration…", async () => {
              const saved = await saveProject(draft.id, editable(draft));
              const written = await writeScript(saved.project.id);
              setDraft({ ...written.project, title: draft.title });
            })}>
              Write narration
            </button>
            <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void run("Saving the script…", async () => {
              const saved = await saveProject(draft.id, editable(draft));
              setDraft(saved.project);
            })}>
              Save script
            </button>
          </div>
        </section>
      ) : null}

      {stage === "storyboard" ? (
        <section className="stack">
          <p className="lede">Each block is spoken in order. The search term under it is what stock footage follows.</p>
          <div className="actions">
            <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void run("Building the storyboard…", async () => {
              const saved = await saveProject(draft.id, editable(draft));
              const built = await buildStoryboard(saved.project.id);
              setDraft(built.project);
            })}>
              Match footage terms
            </button>
            <button className="btn-quiet" type="button" disabled={Boolean(busy) || draft.scenes.length >= 8} onClick={() => setDraft({
              ...draft,
              scenes: [...draft.scenes, { id: crypto.randomUUID(), narration: "", term: "", clip: null }],
            })}>
              Add scene
            </button>
          </div>
          <div className="film">
            {draft.scenes.map((scene, index) => (
              <article className="scene" key={scene.id}>
                <div className="scene-head">
                  <strong>Scene {index + 1}</strong>
                  <span className="actions">
                    <button className="btn-quiet" type="button" disabled={index === 0} onClick={() => {
                      const scenes = [...draft.scenes];
                      [scenes[index - 1], scenes[index]] = [scenes[index], scenes[index - 1]];
                      setDraft({ ...draft, scenes });
                    }}>Up</button>
                    <button className="btn-quiet" type="button" disabled={index === draft.scenes.length - 1} onClick={() => {
                      const scenes = [...draft.scenes];
                      [scenes[index + 1], scenes[index]] = [scenes[index], scenes[index + 1]];
                      setDraft({ ...draft, scenes });
                    }}>Down</button>
                    <button className="btn-danger" type="button" onClick={() => setDraft({
                      ...draft,
                      scenes: draft.scenes.filter((item) => item.id !== scene.id),
                    })}>Remove</button>
                  </span>
                </div>
                <label htmlFor={`narration-${scene.id}`}>Narration</label>
                <textarea id={`narration-${scene.id}`} className="area" style={{ minHeight: "5.5rem" }} value={scene.narration} onChange={(event) => updateScene(scene.id, { narration: event.target.value })} />
                <label htmlFor={`term-${scene.id}`}>Search term</label>
                <input id={`term-${scene.id}`} className="field" value={scene.term} onChange={(event) => updateScene(scene.id, { term: event.target.value })} />
              </article>
            ))}
          </div>
          {draft.scenes.length === 0 ? <p className="meta">No scenes yet. Write a script, then match footage terms.</p> : null}
          <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void run("Saving the storyboard…", async () => {
            const saved = await saveProject(draft.id, editable(draft));
            setDraft(saved.project);
          })}>Save storyboard</button>
        </section>
      ) : null}

      {stage === "footage" ? (
        <section className="stack">
          <p className="lede">Search each scene and choose a clip. Scenes you leave empty are filled from their search term during render.</p>
          {draft.scenes.map((scene, index) => (
            <article className="plate stack" key={scene.id}>
              <div className="scene-head">
                <strong>Scene {index + 1}</strong>
                <span className="meta">{scene.clip ? "Clip selected" : "Using search"}</span>
              </div>
              <p>{scene.narration || "This scene has no narration yet."}</p>
              <label htmlFor={`search-${scene.id}`}>Search term</label>
              <input id={`search-${scene.id}`} className="field" value={scene.term} onChange={(event) => updateScene(scene.id, { term: event.target.value })} />
              <div className="actions">
                <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void run("Searching footage…", async () => {
                  const saved = await saveProject(draft.id, editable({ ...draft, scenes: draft.scenes }));
                  setDraft(saved.project);
                  const found = await searchScene(draft.id, scene.id, scene.term);
                  setResults((current) => ({ ...current, [scene.id]: found.clips }));
                  if (found.clips.length === 0) setError(`No clip matched “${scene.term || "that term"}”. Try a more visual phrase.`);
                })}>
                  Search
                </button>
                {scene.clip ? (
                  <button className="btn-quiet" type="button" onClick={() => updateScene(scene.id, { clip: null })}>Clear selection</button>
                ) : null}
              </div>
              <div className="clips">
                {(results[scene.id] ?? (scene.clip ? [scene.clip] : [])).map((clip) => (
                  <button
                    className={`clip${scene.clip?.url === clip.url ? " clip-live" : ""}`}
                    type="button"
                    key={clip.url}
                    onClick={() => updateScene(scene.id, { clip })}
                  >
                    {clip.thumbnail ? <img src={clip.thumbnail} alt="" /> : <span className="still" />}
                    <p>{clip.creator || "Pexels"} · {clip.duration}s</p>
                  </button>
                ))}
              </div>
            </article>
          ))}
          {draft.scenes.length === 0 ? <p className="meta">Build a storyboard before choosing footage.</p> : null}
          <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void run("Saving footage…", async () => {
            const saved = await saveProject(draft.id, editable(draft));
            setDraft(saved.project);
          })}>Save footage choices</button>
        </section>
      ) : null}

      {stage === "voice" ? (
        <section className="plate stack">
          <div className="split">
            <div>
              <label htmlFor="voice">Voice</label>
              <select id="voice" className="select" value={draft.voice_name} onChange={(event) => setDraft({ ...draft, voice_name: event.target.value })}>
                {(settings?.voices ?? []).map((voice) => (
                  <option key={voice.id} value={voice.id}>{voice.label} — {voice.detail}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="frame">Frame</label>
              <select id="frame" className="select" value={draft.aspect} onChange={(event) => setDraft({ ...draft, aspect: event.target.value as Project["aspect"] })}>
                <option value="9:16">Portrait 9:16</option>
                <option value="16:9">Landscape 16:9</option>
                <option value="1:1">Square 1:1</option>
              </select>
            </div>
            <div>
              <label htmlFor="source">Footage source</label>
              <select id="source" className="select" value={draft.video_source} onChange={(event) => setDraft({ ...draft, video_source: event.target.value })}>
                {(settings?.sources ?? ["pexels", "pixabay", "coverr", "local"]).map((source) => (
                  <option key={source} value={source}>{sourceLabels[source] || source}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="fit">Fit</label>
              <select id="fit" className="select" value={draft.video_fit_mode} onChange={(event) => setDraft({ ...draft, video_fit_mode: event.target.value })}>
                <option value="cover">Fill and crop</option>
                <option value="contain">Fit with black bars</option>
              </select>
            </div>
            <div>
              <label htmlFor="transition">Transition</label>
              <select id="transition" className="select" value={draft.transition} onChange={(event) => setDraft({ ...draft, transition: event.target.value })}>
                <option value="none">None</option>
                <option value="FadeIn">Fade in</option>
                <option value="FadeOut">Fade out</option>
                <option value="SlideIn">Slide in</option>
                <option value="SlideOut">Slide out</option>
                <option value="ZoomIn">Zoom in</option>
                <option value="ZoomOut">Zoom out</option>
                <option value="Shuffle">Shuffle</option>
              </select>
            </div>
          </div>
          <div className="split">
            <div>
              <label htmlFor="rate">Voice rate ({draft.voice_rate})</label>
              <input id="rate" className="field" type="range" min="0.8" max="1.25" step="0.05" value={draft.voice_rate} onChange={(event) => setDraft({ ...draft, voice_rate: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="volume">Voice volume ({draft.voice_volume})</label>
              <input id="volume" className="field" type="range" min="0.5" max="1.5" step="0.05" value={draft.voice_volume} onChange={(event) => setDraft({ ...draft, voice_volume: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="music">Music under the voice ({draft.bgm_volume})</label>
              <input id="music" className="field" type="range" min="0" max="0.4" step="0.02" value={draft.bgm_volume} onChange={(event) => setDraft({ ...draft, bgm_volume: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="clip">Seconds per clip ({draft.clip_duration})</label>
              <input id="clip" className="field" type="range" min="2" max="15" step="1" value={draft.clip_duration} onChange={(event) => setDraft({ ...draft, clip_duration: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="speed">Clip speed ({draft.video_clip_speed}x)</label>
              <input id="speed" className="field" type="range" min="0.5" max="2" step="0.05" value={draft.video_clip_speed} onChange={(event) => setDraft({ ...draft, video_clip_speed: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="count">Videos per render</label>
              <select id="count" className="select" value={draft.video_count} onChange={(event) => setDraft({ ...draft, video_count: Number(event.target.value) })}>
                {[1, 2, 3, 4, 5].map((count) => <option key={count} value={count}>{count}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="concat">Assembly</label>
              <select id="concat" className="select" value={draft.video_concat_mode} disabled={draft.match_materials_to_script} onChange={(event) => setDraft({ ...draft, video_concat_mode: event.target.value })}>
                <option value="sequential">In scene order</option>
                <option value="random">Shuffled</option>
              </select>
            </div>
          </div>
          <label><input type="checkbox" checked={draft.match_materials_to_script} onChange={(event) => setDraft({ ...draft, match_materials_to_script: event.target.checked })} /> Match footage to the script order</label>
          <label><input type="checkbox" checked={draft.subtitle_enabled} onChange={(event) => setDraft({ ...draft, subtitle_enabled: event.target.checked })} /> Burn captions into the video</label>
          <div className="split">
            <div>
              <label htmlFor="position">Caption position</label>
              <select id="position" className="select" value={draft.subtitle_position} onChange={(event) => setDraft({ ...draft, subtitle_position: event.target.value as Project["subtitle_position"] })}>
                <option value="bottom">Bottom</option>
                <option value="center">Center</option>
                <option value="top">Top</option>
                <option value="two_thirds_bottom">Lower third</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div>
              <label htmlFor="display">Caption display</label>
              <select id="display" className="select" value={draft.subtitle_display_mode} onChange={(event) => setDraft({ ...draft, subtitle_display_mode: event.target.value })}>
                <option value="sentence">Sentence by sentence</option>
                <option value="word_by_word">Word by word</option>
              </select>
            </div>
            <div>
              <label htmlFor="animation">Caption animation</label>
              <select id="animation" className="select" value={draft.subtitle_animation} onChange={(event) => setDraft({ ...draft, subtitle_animation: event.target.value })}>
                <option value="none">None</option>
                <option value="pop_spring">Pop</option>
              </select>
            </div>
            <div>
              <label htmlFor="caption-font">Caption font</label>
              <select id="caption-font" className="select" value={draft.font_name} onChange={(event) => setDraft({ ...draft, font_name: event.target.value })}>
                {(settings?.fonts ?? []).map((font) => <option key={font.id} value={font.id}>{font.label}</option>)}
              </select>
            </div>
            {draft.subtitle_position === "custom" ? (
              <div>
                <label htmlFor="custom">Custom position ({draft.custom_position}% from top)</label>
                <input id="custom" className="field" type="range" min="0" max="100" step="1" value={draft.custom_position} onChange={(event) => setDraft({ ...draft, custom_position: Number(event.target.value) })} />
              </div>
            ) : null}
            <div>
              <label htmlFor="size">Caption size ({draft.font_size})</label>
              <input id="size" className="field" type="range" min="30" max="100" step="1" value={draft.font_size} onChange={(event) => setDraft({ ...draft, font_size: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="ink">Caption color</label>
              <input id="ink" className="field" type="color" value={draft.text_fore_color} onChange={(event) => setDraft({ ...draft, text_fore_color: event.target.value.toUpperCase() })} />
            </div>
            <div>
              <label htmlFor="stroke">Stroke color</label>
              <input id="stroke" className="field" type="color" value={draft.stroke_color} onChange={(event) => setDraft({ ...draft, stroke_color: event.target.value.toUpperCase() })} />
            </div>
            <div>
              <label htmlFor="stroke-width">Stroke width ({draft.stroke_width})</label>
              <input id="stroke-width" className="field" type="range" min="0" max="10" step="0.1" value={draft.stroke_width} onChange={(event) => setDraft({ ...draft, stroke_width: Number(event.target.value) })} />
            </div>
            <div>
              <label htmlFor="bar">Caption bar</label>
              <input id="bar" className="field" type="color" value={draft.text_background_color} disabled={!draft.use_caption_bar} onChange={(event) => setDraft({ ...draft, text_background_color: event.target.value.toUpperCase() })} />
              <label><input type="checkbox" checked={draft.use_caption_bar} onChange={(event) => setDraft({ ...draft, use_caption_bar: event.target.checked })} /> Dark bar behind each line</label>
            </div>
          </div>
          <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void run("Saving voice and captions…", async () => {
            const saved = await saveProject(draft.id, editable(draft));
            setDraft(saved.project);
          })}>Save voice and captions</button>
        </section>
      ) : null}

      {stage === "render" ? (
        <section className="stack">
          <p className="lede">
            {spoken.length
              ? "The storyboard is what will be spoken, in this order."
              : "Add narration in the script or storyboard before rendering."}
          </p>
          {spoken.map((line, index) => <p key={`${index}-${line.slice(0, 12)}`} className="plate">{index + 1}. {line}</p>)}
          <p className="meta">
            {draft.scenes.some((scene) => scene.clip)
              ? "Selected clips are downloaded before rendering starts."
              : "Footage is searched from each scene term while the render progress moves."}
          </p>
          <div className="actions">
            <button className="btn" type="button" disabled={Boolean(busy) || task?.state === 4} onClick={() => void run(
              draft.scenes.some((scene) => scene.clip) ? "Preparing selected clips…" : "Queuing the render…",
              async () => {
                const queued = await renderProject(draft.id, editable(draft));
                setDraft(queued.project);
                setTask(null);
                setStage("render");
              },
            )}>
              {task?.state === 4 ? "Rendering…" : "Render video"}
            </button>
          </div>
          {task ? (
            <div className="stack">
              <progress className="meter" max={100} value={task.state === -1 ? task.progress : task.progress} />
              <p role="status">
                {task.state === -1
                  ? `Render failed${task.failed_stage ? ` during ${task.failed_stage}` : ""}. ${task.error || "See the API log for the stage that stopped."}`
                  : `${progressLabel(task.progress)} · ${task.progress}%`}
              </p>
              {video ? (
                <>
                  <video className={`player${draft.aspect === "16:9" ? " wide" : ""}`} controls src={video} />
                  <a className="btn" href={video} download>Download MP4</a>
                </>
              ) : null}
            </div>
          ) : null}
          <div className="actions">
            {confirmDelete ? (
              <>
                <button className="btn-danger" type="button" onClick={() => void run("Deleting the reel…", async () => {
                  await deleteProject(draft.id);
                  navigate("/");
                })}>Delete this reel</button>
                <button className="btn-quiet" type="button" onClick={() => setConfirmDelete(false)}>Keep it</button>
              </>
            ) : (
              <button className="btn-danger" type="button" onClick={() => setConfirmDelete(true)}>Delete reel</button>
            )}
          </div>
        </section>
      ) : null}
    </>
  );
}

import { useEffect, useState } from "react";
import {
  checkSubtitles, openTaskFolder, previewPrompt, previewVoice, progressLabel, quoteOpenRouterVideo, quoteScripts, quoteVideo, readCatalog,
  readOpenRouterVideoModels,
  readProviderVoices, readScriptRun, readSettings, readTask, readTaskLogs, readVideoOptions,
  runScripts, saveSettings, startAudio, startSubtitle, startVideo, uploadMaterial, uploadMusic,
  uploadVoiceover, writeNarration, writeSocial, writeTerms,
} from "../api";
import { useT } from "../i18n";
import type { BenchSettings, RenderTask } from "../types";

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
  openrouter: "OpenRouter video",
};

const sourceGroups: { label: string; ids: string[] }[] = [
  { label: "Stock video", ids: ["pexels", "pixabay", "coverr"] },
  { label: "AI video", ids: ["openrouter", "wavespeed", "volcengine_seedance", "ofox", "metaso_minimax", "muapi", "loomloom"] },
  { label: "AI image", ids: ["openai_image"] },
  { label: "Local material", ids: ["local"] },
];

const languageNames: Record<string, string> = {
  "": "Auto detect",
  English: "English",
  "ca-ES": "Catalan",
  "zh-CN": "Chinese (Simplified)",
  "zh-HK": "Chinese (Hong Kong)",
  "zh-TW": "Chinese (Traditional)",
  "de-DE": "German",
  "en-US": "English (US)",
  "es-ES": "Spanish",
  "fr-FR": "French",
  "it-IT": "Italian",
  "ru-RU": "Russian",
  "vi-VN": "Vietnamese",
  "th-TH": "Thai",
  "tr-TR": "Turkish",
  "ja-JP": "Japanese",
  "ko-KR": "Korean",
  "pt-BR": "Portuguese (Brazil)",
  "hi-IN": "Hindi",
  "ar-SA": "Arabic",
};

const fallbackLanguages = Object.keys(languageNames);

type VoiceMode = "tts" | "none" | "upload";
type Catalog = {
  languages: string[];
  tts_servers: { id: string; label: string }[];
  bgm_types: { id: string; label: string }[];
  songs: string[];
  music_files: string[];
  default_system_prompt: string;
  script_backends: { id: string; label: string }[];
};
type Draft = {
  video_subject: string;
  video_script: string;
  video_terms: string;
  video_language: string;
  paragraph_number: number;
  video_script_prompt: string;
  custom_system_prompt: string;
  video_source: string;
  video_concat_mode: string;
  match_materials_to_script: boolean;
  transition: string;
  aspect: string;
  video_fit_mode: string;
  clip_duration: number;
  video_clip_speed: number;
  video_count: number;
  video_codec: string;
  tts_server: string;
  voice_mode: VoiceMode;
  voice_name: string;
  voice_rate: number;
  voice_volume: number;
  bgm_type: string;
  bgm_file: string;
  video_music_prompt: string;
  bgm_volume: number;
  n_threads: number;
  candidate_count: number;
  duration_seconds: number;
  voxcpm_prompt_text: string;
  subtitle_enabled: boolean;
  font_name: string;
  subtitle_position: string;
  custom_position: number;
  subtitle_display_mode: string;
  subtitle_animation: string;
  font_size: number;
  text_fore_color: string;
  stroke_color: string;
  stroke_width: number;
  use_caption_bar: boolean;
  text_background_color: string;
  rounded_subtitle_background: boolean;
  subtitle_provider: string;
};

function draftFrom(settings: BenchSettings): Draft {
  return {
    video_subject: "",
    video_script: "",
    video_terms: "",
    video_language: settings.video_language === "English" ? "English" : settings.video_language || "",
    paragraph_number: settings.paragraph_number || 1,
    video_script_prompt: "",
    custom_system_prompt: "",
    video_source: settings.video_source || "pexels",
    video_concat_mode: settings.video_concat_mode || "random",
    match_materials_to_script: settings.match_materials_to_script,
    transition: settings.transition || "none",
    aspect: settings.default_aspect || "9:16",
    video_fit_mode: settings.video_fit_mode || "cover",
    clip_duration: settings.clip_duration || 3,
    video_clip_speed: settings.video_clip_speed || 1,
    video_count: settings.video_count || 1,
    video_codec: settings.video_codec || "__default__",
    tts_server: settings.tts_server || "azure-tts-v1",
    voice_mode: "tts",
    voice_name: settings.default_voice,
    voice_rate: settings.voice_rate || 1,
    voice_volume: settings.voice_volume || 1,
    bgm_type: settings.bgm_type ?? ((settings.bgm_volume || 0) > 0 ? "random" : ""),
    bgm_file: settings.bgm_file || "",
    video_music_prompt: "",
    bgm_volume: settings.bgm_volume || 0.2,
    n_threads: 2,
    candidate_count: 3,
    duration_seconds: 60,
    voxcpm_prompt_text: "",
    subtitle_enabled: settings.subtitle_enabled,
    font_name: settings.font_name,
    subtitle_position: settings.subtitle_position || "bottom",
    custom_position: settings.custom_position || 70,
    subtitle_display_mode: settings.subtitle_display_mode || "sentence",
    subtitle_animation: settings.subtitle_animation || "none",
    font_size: settings.font_size || 60,
    text_fore_color: settings.text_fore_color || "#FFFFFF",
    stroke_color: settings.stroke_color || "#000000",
    stroke_width: settings.stroke_width || 1.5,
    use_caption_bar: settings.use_caption_bar,
    text_background_color: settings.text_background_color || "#000000",
    rounded_subtitle_background: false,
    subtitle_provider: settings.subtitle_provider || "edge",
  };
}

function Toggle({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="row">
      <span id={`${id}-label`}>{label}</span>
      <label className="switch">
        <input id={id} aria-labelledby={`${id}-label`} type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
        <span />
      </label>
    </div>
  );
}

const sourceNeeds: Record<string, keyof BenchSettings> = {
  pexels: "pexels_configured",
  pixabay: "pixabay_configured",
  coverr: "coverr_configured",
  wavespeed: "wavespeed_configured",
  volcengine_seedance: "seedance_configured",
  ofox: "ofox_configured",
  metaso_minimax: "metaso_configured",
  muapi: "muapi_configured",
  openai_image: "openai_image_configured",
  loomloom: "loomloom_configured",
  openrouter: "openrouter_configured",
};

export function Dashboard() {
  const t = useT();
  const [settings, setSettings] = useState<BenchSettings | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [voices, setVoices] = useState<{ id: string; label: string }[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [voiceoverToken, setVoiceoverToken] = useState("");
  const [voiceoverName, setVoiceoverName] = useState("");
  const [referenceToken, setReferenceToken] = useState("");
  const [promptToken, setPromptToken] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [captionWarning, setCaptionWarning] = useState("");
  const [busy, setBusy] = useState("");
  const [task, setTask] = useState<RenderTask | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [promptText, setPromptText] = useState("");
  const [social, setSocial] = useState("");
  const [scriptQuote, setScriptQuote] = useState<{ token: string; amount: string; currency: string; task_count: number } | null>(null);
  const [scriptConfirm, setScriptConfirm] = useState(false);
  const [candidates, setCandidates] = useState<{ script: string; terms: string[] }[]>([]);
  const [videoModels, setVideoModels] = useState<{ id: string; label: string }[]>([]);
  const [videoModel, setVideoModel] = useState("");
  const [videoQuote, setVideoQuote] = useState<{ token: string; amount: string; currency: string; task_count: number } | null>(null);
  const [videoConfirm, setVideoConfirm] = useState(false);
  const [orModels, setOrModels] = useState<{ id: string; name: string; durations: number[]; resolutions: string[]; aspects: string[] }[]>([]);
  const [orModel, setOrModel] = useState("heygen/heygen-video-1");
  const [orResolution, setOrResolution] = useState("480p");
  const [orDuration, setOrDuration] = useState(5);
  const [orQuote, setOrQuote] = useState<{
    model: string;
    name: string;
    duration: number;
    resolution: string;
    aspect_ratio: string;
    rate_usd: string;
    rate_unit: string;
    estimated_usd: string;
    budget_usd: string;
    within_budget: boolean;
    submitted: boolean;
    max_scenes?: number;
  } | null>(null);
  const [projectBudget, setProjectBudget] = useState("1.00");
  const [visualStyle, setVisualStyle] = useState("");
  const [sceneRetries, setSceneRetries] = useState(1);
  const [orNotice, setOrNotice] = useState("");
  const [opened, setOpened] = useState("");

  useEffect(() => {
    let stop = false;
    readSettings()
      .then(async (data) => {
        if (stop) return;
        const next = draftFrom(data.settings);
        const stored = sessionStorage.getItem("frameforge-draft");
        sessionStorage.removeItem("frameforge-draft");
        if (stored) {
          try {
            Object.assign(next, JSON.parse(stored));
          } catch {
            /* keep the saved defaults */
          }
        }
        setSettings(data.settings);
        setDraft(next);
        setOrModel(data.settings.openrouter_video_model || "heygen/heygen-video-1");
        setOrResolution(data.settings.openrouter_video_resolution || "480p");
        const listed = await readProviderVoices(next.tts_server).catch(() => ({ voices: data.settings.voices }));
        if (!stop) setVoices(listed.voices?.length ? listed.voices : data.settings.voices);
      })
      .catch((reason: Error) => {
        if (!stop) setError(reason.message);
      });
    readCatalog()
      .then((data) => {
        if (!stop) setCatalog(data);
      })
      .catch(() => undefined);
    return () => {
      stop = true;
    };
  }, []);

  useEffect(() => {
    if (!task || task.state !== 4) return;
    let stop = false;
    const timer = window.setTimeout(() => {
      readTask(task.task_id)
        .then((next) => {
          if (!stop) setTask(next);
        })
        .catch((reason: Error) => {
          if (!stop) setError(reason.message);
        });
    }, 2000);
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [task]);

  useEffect(() => {
    if (!draft?.subtitle_enabled) {
      setCaptionWarning("");
      return;
    }
    const timer = window.setTimeout(() => {
      checkSubtitles({
        video_subject: draft.video_subject || "preview",
        video_script: draft.video_script,
        subtitle_enabled: true,
        font_name: draft.font_name,
        text_fore_color: draft.text_fore_color,
        stroke_color: draft.stroke_color,
        text_background_color: draft.use_caption_bar ? draft.text_background_color : false,
      })
        .then((result) => {
          const notes = [];
          if (result.colors) notes.push("Text and stroke are hard to tell apart.");
          if (result.font) notes.push("This font does not cover some characters in the script.");
          setCaptionWarning(notes.join(" "));
        })
        .catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [draft?.subtitle_enabled, draft?.font_name, draft?.text_fore_color, draft?.stroke_color, draft?.text_background_color, draft?.use_caption_bar, draft?.video_script, draft?.video_subject]);

  useEffect(() => {
    if (!task || settings?.hide_log) return;
    let stop = false;
    const timer = window.setTimeout(() => {
      readTaskLogs(task.task_id)
        .then((data) => {
          if (!stop) setLogs(data.lines || []);
        })
        .catch(() => undefined);
    }, 1500);
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [task, logs, settings?.hide_log]);

  useEffect(() => {
    if (!task || task.state !== 1 || !settings?.open_task_folder_on_completion || opened === task.task_id) return;
    setOpened(task.task_id);
    openTaskFolder(task.task_id).catch(() => undefined);
  }, [task, settings?.open_task_folder_on_completion, opened]);

  useEffect(() => {
    if (draft?.video_source !== "openrouter") return;
    let stop = false;
    readOpenRouterVideoModels()
      .then((data) => {
        if (!stop) setOrModels(data.models || []);
      })
      .catch(() => undefined);
    return () => {
      stop = true;
    };
  }, [draft?.video_source]);

  useEffect(() => {
    if (draft?.video_source !== "openrouter") return;
    setOrQuote(null);
    const timer = window.setTimeout(() => {
      quoteOpenRouterVideo({
        model: orModel,
        resolution: orResolution,
        duration: orDuration,
        aspect_ratio: draft.aspect,
        budget_usd: projectBudget || "0",
        project_budget: true,
      })
        .then((quote) => {
          setOrQuote(quote);
          setOrNotice("");
        })
        .catch((reason: Error) => {
          setOrQuote(null);
          setOrNotice(reason.message || "The OpenRouter estimate could not be loaded.");
        });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [draft?.video_source, draft?.aspect, orModel, orResolution, orDuration, projectBudget]);

  function patch(next: Partial<Draft>) {
    setDraft((current) => current && ({ ...current, ...next }));
  }

  async function generateScript() {
    if (!draft?.video_subject.trim()) {
      setError("Enter a video subject before generating a script.");
      return;
    }
    setBusy("Writing the script…");
    setError("");
    try {
      const result = await writeNarration({
        video_subject: draft.video_subject.trim(),
        video_language: draft.video_language,
        paragraph_number: draft.paragraph_number,
        video_script_prompt: draft.video_script_prompt,
        custom_system_prompt: draft.custom_system_prompt,
      });
      patch({ video_script: result.video_script });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The script could not be written.");
    } finally {
      setBusy("");
    }
  }

  async function generateTerms() {
    if (!draft?.video_script.trim()) {
      setError("Write a script before generating keywords.");
      return;
    }
    setBusy("Finding keywords…");
    setError("");
    try {
      const result = await writeTerms({
        video_subject: draft.video_subject.trim(),
        video_script: draft.video_script,
        amount: draft.match_materials_to_script ? 8 : 5,
        match_materials_to_script: draft.match_materials_to_script,
      });
      const terms = Array.isArray(result.video_terms) ? result.video_terms.join(", ") : String(result.video_terms || "");
      patch({ video_terms: terms });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Keywords could not be generated.");
    } finally {
      setBusy("");
    }
  }

  async function changeServer(server: string) {
    patch({ tts_server: server });
    setBusy("Loading voices…");
    try {
      const listed = await readProviderVoices(server);
      setVoices(listed.voices);
      if (listed.voices[0]) patch({ voice_name: listed.voices[0].id });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Voices could not be loaded.");
    } finally {
      setBusy("");
    }
  }

  async function showPrompt() {
    if (!draft) return;
    setBusy("Building the prompt…");
    setError("");
    try {
      const result = await previewPrompt(draft);
      setPromptText(result.prompt);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The prompt could not be built.");
    } finally {
      setBusy("");
    }
  }

  async function hearVoice() {
    if (!draft) return;
    setBusy("Speaking a sample…");
    setError("");
    try {
      const result = await previewVoice({
        voice_name: draft.voice_name,
        voice_rate: draft.voice_rate,
        voice_volume: draft.voice_volume,
      });
      const player = new Audio(`/api/v1/frameforge/previews/${result.file}`);
      await player.play();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The voice sample could not be played.");
    } finally {
      setBusy("");
    }
  }

  async function quoteScriptBatch() {
    if (!draft?.video_subject.trim()) {
      setError("Enter a video subject before requesting a quote.");
      return;
    }
    setBusy("Requesting a script quote…");
    setError("");
    setScriptConfirm(false);
    try {
      setScriptQuote(await quoteScripts({
        video_subject: draft.video_subject.trim(),
        video_language: draft.video_language,
        video_script_prompt: draft.video_script_prompt,
        candidate_count: draft.candidate_count,
        duration_seconds: draft.duration_seconds,
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The quote could not be created.");
    } finally {
      setBusy("");
    }
  }

  async function startScriptBatch() {
    if (!scriptQuote || !scriptConfirm) return;
    setBusy("Starting the paid script batch…");
    setError("");
    try {
      const started = await runScripts(scriptQuote.token);
      setScriptQuote(null);
      let runId = started.run_id;
      for (let attempt = 0; attempt < 40; attempt += 1) {
        const status = await readScriptRun(runId);
        if (status.status === "completed") {
          setCandidates(status.candidates);
          if (status.candidates[0]) {
            patch({
              video_script: status.candidates[0].script,
              video_terms: status.candidates[0].terms.join(", "),
            });
          }
          if (status.errors.length) setNotice(status.errors.join(" "));
          return;
        }
        if (status.status && status.status !== "running" && status.status !== "queued" && status.status !== "pending") {
          throw new Error(status.error || "The script batch did not finish.");
        }
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
      }
      setNotice("The batch is still running. Open Library and check again in a moment.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The script batch could not be started.");
    } finally {
      setBusy("");
    }
  }

  async function loadVideoModels() {
    setBusy("Loading video models…");
    setError("");
    try {
      const options = await readVideoOptions();
      setVideoModels(options.models);
      setVideoModel(options.default_model);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Video models could not be loaded.");
    } finally {
      setBusy("");
    }
  }

  async function quotePaidVideo() {
    if (!draft) return;
    const scenes = draft.video_terms.split(",").map((term) => term.trim()).filter(Boolean);
    if (!draft.video_subject.trim() || scenes.length === 0 || !videoModel) {
      setError("Enter a subject, keywords, and a model before requesting a video quote.");
      return;
    }
    setBusy("Requesting a video quote…");
    setError("");
    setVideoConfirm(false);
    try {
      setVideoQuote(await quoteVideo({
        video_subject: draft.video_subject.trim(),
        scenes,
        model_id: videoModel,
        aspect: draft.aspect === "1:1" ? "9:16" : draft.aspect,
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The video quote could not be created.");
    } finally {
      setBusy("");
    }
  }

  function requestBody(materials: { provider: string; url: string; duration: number }[]) {
    if (!draft) return {};
    const terms = draft.video_terms.split(",").map((term) => term.trim()).filter(Boolean);
    return {
      video_subject: draft.video_subject.trim() || draft.video_script.trim().slice(0, 80),
      video_script: draft.video_script.trim(),
      video_terms: terms.length ? terms : draft.video_terms.trim(),
      video_language: draft.video_language,
      paragraph_number: draft.paragraph_number,
      video_script_prompt: draft.video_script_prompt,
      custom_system_prompt: draft.custom_system_prompt === catalog?.default_system_prompt ? "" : draft.custom_system_prompt,
      video_source: draft.video_source,
      video_materials: draft.video_source === "local" ? materials : undefined,
      video_concat_mode: draft.match_materials_to_script ? "sequential" : draft.video_concat_mode,
      match_materials_to_script: draft.match_materials_to_script,
      video_transition_mode: draft.transition === "none" ? null : draft.transition,
      video_aspect: draft.aspect,
      video_fit_mode: draft.video_fit_mode,
      video_clip_duration: draft.video_source === "openrouter" ? Math.max(draft.clip_duration, orDuration) : draft.clip_duration,
      openrouter_video_model: draft.video_source === "openrouter" ? orModel : "",
      openrouter_video_resolution: draft.video_source === "openrouter" ? orResolution : "480p",
      openrouter_video_duration: draft.video_source === "openrouter" ? orDuration : 5,
      openrouter_video_approved: false,
      openrouter_video_budget_usd: Number(projectBudget) || 0,
      project_budget_usd: Number(projectBudget) || 0,
      visual_style: visualStyle,
      max_scene_retries: sceneRetries,
      video_clip_speed: draft.video_clip_speed,
      video_count: draft.video_count,
      voice_name: draft.voice_mode === "none" || draft.voice_mode === "upload" ? "no-voice" : draft.voice_name,
      voice_rate: draft.voice_rate,
      voice_volume: draft.voice_volume,
      bgm_type: draft.bgm_type,
      bgm_file: draft.bgm_type === "preset" || draft.bgm_type === "custom" ? draft.bgm_file : "",
      bgm_volume: draft.bgm_type ? draft.bgm_volume : 0,
      video_music_prompt: draft.video_music_prompt,
      subtitle_enabled: draft.subtitle_enabled,
      font_name: draft.font_name,
      subtitle_position: draft.subtitle_position,
      custom_position: draft.custom_position,
      subtitle_display_mode: draft.subtitle_display_mode,
      subtitle_animation: draft.subtitle_animation,
      font_size: draft.font_size,
      text_fore_color: draft.text_fore_color,
      stroke_color: draft.stroke_color,
      stroke_width: draft.stroke_width,
      text_background_color: draft.use_caption_bar ? draft.text_background_color : false,
      rounded_subtitle_background: draft.use_caption_bar && draft.rounded_subtitle_background,
      n_threads: draft.n_threads,
      voiceover_token: draft.voice_mode === "upload" ? voiceoverToken : "",
      loomloom_token: draft.video_source === "loomloom" && videoConfirm ? videoQuote?.token || "" : "",
      voxcpm_reference_token: referenceToken,
      voxcpm_prompt_token: promptToken,
      voxcpm_prompt_text: draft.voxcpm_prompt_text,
    };
  }

  async function persistDraft() {
    if (!draft || !settings) return;
    await saveSettings({
      ...settings,
      openrouter_model_name: settings.model_override || settings.model_name,
      video_source: draft.video_source,
      video_concat_mode: draft.match_materials_to_script ? "sequential" : draft.video_concat_mode,
      match_materials_to_script: draft.match_materials_to_script,
      transition: draft.transition,
      default_aspect: draft.aspect,
      video_fit_mode: draft.video_fit_mode,
      clip_duration: draft.clip_duration,
      video_clip_speed: draft.video_clip_speed,
      video_count: draft.video_count,
      video_codec: draft.video_codec,
      default_voice: draft.voice_mode === "tts" ? draft.voice_name : settings.default_voice,
      voice_rate: draft.voice_rate,
      voice_volume: draft.voice_volume,
      bgm_type: draft.bgm_type,
      bgm_file: draft.bgm_file,
      bgm_volume: draft.bgm_type ? draft.bgm_volume : 0,
      tts_server: draft.tts_server,
      subtitle_enabled: draft.subtitle_enabled,
      font_name: draft.font_name,
      subtitle_position: draft.subtitle_position,
      custom_position: draft.custom_position,
      subtitle_display_mode: draft.subtitle_display_mode,
      subtitle_animation: draft.subtitle_animation,
      font_size: draft.font_size,
      text_fore_color: draft.text_fore_color,
      stroke_color: draft.stroke_color,
      stroke_width: draft.stroke_width,
      use_caption_bar: draft.use_caption_bar,
      text_background_color: draft.text_background_color,
      video_language: draft.video_language || "English",
      paragraph_number: draft.paragraph_number,
      subtitle_provider: draft.subtitle_provider,
      openrouter_video_model: draft.video_source === "openrouter" ? orModel : settings.openrouter_video_model,
      openrouter_video_resolution: draft.video_source === "openrouter" ? orResolution : settings.openrouter_video_resolution,
      script_generation_backend: settings.script_generation_backend,
    });
  }

  async function generate() {
    if (!draft || !settings) return;
    if (!draft.video_subject.trim() && !draft.video_script.trim()) {
      setError("Enter a subject or a script before generating.");
      return;
    }
    const needed = sourceNeeds[draft.video_source];
    if (needed && !settings[needed]) {
      setError("Add this source’s key in Settings before generating.");
      return;
    }
    if (draft.video_source === "local" && files.length === 0) {
      setError("Choose local files before generating.");
      return;
    }
    if (draft.voice_mode === "upload" && !voiceoverToken) {
      setError("Upload a voiceover before generating.");
      return;
    }
    if (draft.video_source === "loomloom" && (!videoQuote || !videoConfirm)) {
      setError("Get a LoomLoom quote and confirm the charge before generating.");
      return;
    }
    const openRouterReady = Boolean(
      orQuote
      && orQuote.submitted === false
      && orQuote.within_budget
      && (orQuote.max_scenes ?? 0) >= 1
      && orQuote.model === orModel
      && orQuote.resolution === orResolution
      && orQuote.duration === orDuration
      && orQuote.aspect_ratio === draft.aspect
      && Math.abs(Number(orQuote.budget_usd) - Number(projectBudget)) < 0.001
    );
    if (draft.video_source === "openrouter" && !openRouterReady) {
      setError(orQuote && !orQuote.within_budget
        ? `One scene is estimated at $${orQuote.estimated_usd}, above the $${orQuote.budget_usd} project budget. Nothing was submitted.`
        : "Wait for the project estimate before generating.");
      return;
    }
    setBusy("Starting the video…");
    setError("");
    setTask(null);
    setLogs([]);
    try {
      const materials = [];
      for (const file of files) {
        const stored = await uploadMaterial(file);
        materials.push({ provider: "local", url: stored, duration: 0 });
      }
      await persistDraft();
      const queued = await startVideo(requestBody(materials));
      setTask({ task_id: queued.task_id, state: 4, progress: 0 });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The video could not be started.");
    } finally {
      setBusy("");
    }
  }

  async function generatePart(kind: "audio" | "subtitle") {
    if (!draft?.video_script.trim()) {
      setError("Write a script before generating audio or subtitles.");
      return;
    }
    setBusy(kind === "audio" ? "Starting the audio…" : "Starting the subtitles…");
    setError("");
    setTask(null);
    try {
      await persistDraft();
      const body = requestBody([]);
      const queued = kind === "audio" ? await startAudio(body) : await startSubtitle(body);
      setTask({ task_id: queued.task_id, state: 4, progress: 0 });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "That job could not be started.");
    } finally {
      setBusy("");
    }
  }

  async function socialCopy() {
    if (!draft?.video_subject.trim() && !draft?.video_script.trim()) {
      setError("Enter a subject or script before writing a post.");
      return;
    }
    setBusy("Writing the post…");
    setError("");
    try {
      const result = await writeSocial({
        video_subject: draft.video_subject.trim(),
        video_script: draft.video_script,
        language: draft.video_language || "auto",
        platform: settings?.upload_post_platforms?.[0] || "tiktok",
      });
      const tags = Array.isArray(result.hashtags) ? result.hashtags.join(" ") : String(result.hashtags || "");
      setSocial([result.title, result.caption, tags].filter(Boolean).join("\n\n"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The post could not be written.");
    } finally {
      setBusy("");
    }
  }

  function exportPreset() {
    if (!draft) return;
    const blob = new Blob([JSON.stringify({ schema: "frameforge.settings-preset", params: draft }, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "frameforge-settings.json";
    link.click();
  }

  async function importPreset(file: File) {
    try {
      const payload = JSON.parse(await file.text()) as { params?: Partial<Draft> };
      const params = payload.params || (payload as Partial<Draft>);
      patch(params);
      setNotice("Settings preset loaded into this form.");
    } catch {
      setError("That settings file could not be read.");
    }
  }

  if (!draft || !settings) {
    return error ? <p className="notice notice-bad" role="alert">{error}</p> : <p className="meta">Loading…</p>;
  }

  const video = task?.videos?.find((item) => item.endsWith(".mp4")) ?? "";
  const clipChoices = draft.video_source === "metaso_minimax" || draft.video_source === "muapi"
    ? [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    : [2, 3, 4, 5, 6, 7, 8, 9, 10];

  return (
    <>
      <h1 className="page-title">Generate</h1>
      {error ? <p className="notice notice-bad" role="alert">{error}</p> : null}
      {busy ? <p className="meta" role="status">{busy}</p> : null}

      <section className="section">
        <h2>Video Script</h2>
        <div className="group">
          <div className="block">
            <label htmlFor="subject">Video subject</label>
            <textarea id="subject" className="area" value={draft.video_subject} placeholder="Three fascinating facts about artificial intelligence" onChange={(event) => patch({ video_subject: event.target.value })} />
          </div>
          <div className="row">
            <label htmlFor="language">Script language</label>
            <select id="language" className="select" value={draft.video_language} onChange={(event) => patch({ video_language: event.target.value })}>
              {(catalog?.languages || fallbackLanguages).map((value) => <option key={value || "auto"} value={value}>{languageNames[value] || value || "Auto detect"}</option>)}
            </select>
          </div>
          <details className="block">
            <summary>{t("Advanced Script Settings", "Advanced script settings")}</summary>
            <div className="stack">
              <div className="row">
                <label htmlFor="backend">{t("Script Generation Method", "Script method")}</label>
                <select id="backend" className="select" value={settings.script_generation_backend || "local"} onChange={(event) => setSettings({ ...settings, script_generation_backend: event.target.value })}>
                  {(catalog?.script_backends || [{ id: "local", label: "This computer" }, { id: "loomloom", label: "Shengsuan Cloud batch" }]).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                </select>
              </div>
              <div>
                <label htmlFor="paragraphs">{t("Script Paragraph Number", "Paragraphs")} ({draft.paragraph_number})</label>
                <input id="paragraphs" className="field" type="range" min="1" max="10" step="1" value={draft.paragraph_number} onChange={(event) => patch({ paragraph_number: Number(event.target.value) })} />
              </div>
              <div>
                <label htmlFor="requirements">{t("Custom Script Requirements", "Custom script requirements")}</label>
                <textarea id="requirements" className="area" value={draft.video_script_prompt} onChange={(event) => patch({ video_script_prompt: event.target.value })} />
              </div>
              <div>
                <label htmlFor="system">{t("Custom System Prompt", "Custom system prompt")}</label>
                <textarea id="system" className="area" value={draft.custom_system_prompt} onChange={(event) => patch({ custom_system_prompt: event.target.value })} />
              </div>
              <div className="actions">
                <button className="btn-quiet" type="button" onClick={() => patch({ custom_system_prompt: catalog?.default_system_prompt || "" })}>{t("Restore Default System Prompt", "Restore default prompt")}</button>
                <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void showPrompt()}>{t("Preview Final Prompt", "Preview prompt")}</button>
              </div>
              {promptText ? <pre className="area">{promptText}</pre> : null}
            </div>
          </details>
          {settings.script_generation_backend === "loomloom" ? (
            <div className="block stack">
              <div className="row">
                <label htmlFor="candidates">Script candidates</label>
                <input id="candidates" className="field" type="number" min="1" max="10" value={draft.candidate_count} onChange={(event) => patch({ candidate_count: Number(event.target.value) })} />
              </div>
              <div className="row">
                <label htmlFor="duration">Target duration (seconds)</label>
                <input id="duration" className="field" type="number" min="10" max="600" step="10" value={draft.duration_seconds} onChange={(event) => patch({ duration_seconds: Number(event.target.value) })} />
              </div>
              <div className="actions">
                <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void quoteScriptBatch()}>Get quote</button>
              </div>
              {scriptQuote ? (
                <>
                  <p className="meta">{scriptQuote.task_count} scripts · {scriptQuote.amount} {scriptQuote.currency}</p>
                  <Toggle id="script-charge" label="Confirm this charge" checked={scriptConfirm} onChange={setScriptConfirm} />
                  <button className="btn" type="button" disabled={!scriptConfirm || Boolean(busy)} onClick={() => void startScriptBatch()}>Run paid batch</button>
                </>
              ) : null}
              {candidates.length > 1 ? (
                <div className="stack">
                  {candidates.map((item, index) => (
                    <button className="btn-quiet" key={item.script.slice(0, 24)} type="button" onClick={() => patch({ video_script: item.script, video_terms: item.terms.join(", ") })}>
                      Use candidate {index + 1}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="block">
              <div className="actions">
                <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void generateScript()}>{t("Generate Video Script", "Generate script")}</button>
              </div>
            </div>
          )}
          <div className="block">
            <label htmlFor="script">Video script</label>
            <textarea id="script" className="area" style={{ minHeight: "160px" }} value={draft.video_script} onChange={(event) => patch({ video_script: event.target.value })} />
          </div>
          <div className="block">
            <div className="actions">
              <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void generateTerms()}>Generate video keywords</button>
            </div>
          </div>
          <div className="block">
            <label htmlFor="terms">Video keywords</label>
            <textarea id="terms" className="area" value={draft.video_terms} onChange={(event) => patch({ video_terms: event.target.value })} />
          </div>
        </div>
        <p className="section-foot">Keywords are separated by commas. Leave them empty and the render will create them from the script.</p>
      </section>

      <section className="section">
        <h2>Video</h2>
        <div className="group">
          <div className="row">
            <label htmlFor="source">Video source</label>
            <select id="source" className="select" value={draft.video_source} onChange={(event) => {
              const video_source = event.target.value;
              patch(video_source === "openrouter"
                ? { video_source, aspect: "9:16", clip_duration: 5 }
                : { video_source });
            }}>
              {sourceGroups.map((group) => (
                <optgroup key={group.label} label={group.label}>
                  {group.ids.map((id) => <option key={id} value={id}>{sourceLabels[id]}</option>)}
                </optgroup>
              ))}
            </select>
          </div>
          {draft.video_source === "local" ? (
            <div className="block">
              <label htmlFor="files">{t("Upload Local Files", "Upload local files")}</label>
              <input id="files" className="field" type="file" multiple accept="video/*,image/*" onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
              {files.length ? <p className="meta">{files.map((file) => file.name).join(", ")}</p> : null}
            </div>
          ) : null}
          {draft.video_source === "loomloom" ? (
            <div className="block stack">
              <p className="meta">Shengsuan Cloud charges for each scene. Keywords become the scenes.</p>
              <div className="actions">
                <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void loadVideoModels()}>Load models</button>
              </div>
              {videoModels.length ? (
                <div className="row">
                  <label htmlFor="loom-model">Model</label>
                  <select id="loom-model" className="select" value={videoModel} onChange={(event) => setVideoModel(event.target.value)}>
                    {videoModels.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                  </select>
                </div>
              ) : null}
              <button className="btn" type="button" disabled={Boolean(busy)} onClick={() => void quotePaidVideo()}>Get video quote</button>
              {videoQuote ? (
                <>
                  <p className="meta">{videoQuote.task_count} clips · {videoQuote.amount} {videoQuote.currency}</p>
                  <Toggle id="video-charge" label="Confirm this charge" checked={videoConfirm} onChange={setVideoConfirm} />
                </>
              ) : null}
            </div>
          ) : null}
          {draft.video_source === "openrouter" ? (
            <div className="block stack">
              <p className="meta">One paid clip. Narration, subtitles, and export stay on the existing pipeline. Stock footage is not used. A failed request is not sent again.</p>
              <div className="row">
                <label htmlFor="or-model">Model</label>
                <select id="or-model" className="select" value={orModel} onChange={(event) => {
                  const id = event.target.value;
                  const model = orModels.find((item) => item.id === id);
                  const resolution = model?.resolutions?.includes(orResolution)
                    ? orResolution
                    : (model?.resolutions?.includes("480p") ? "480p" : model?.resolutions?.[0] || orResolution);
                  const duration = model?.durations?.map(Number).includes(orDuration)
                    ? orDuration
                    : (model?.durations?.map(Number).includes(5) ? 5 : Number(model?.durations?.[0] || orDuration));
                  setOrModel(id);
                  setOrResolution(resolution);
                  setOrDuration(duration);
                }}>
                  {(orModels.length ? orModels : [{ id: "heygen/heygen-video-1", name: "HeyGen: HeyGen Video", durations: [5], resolutions: ["480p"], aspects: ["9:16"] }]).map((item) => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
              </div>
              <div className="row">
                <label htmlFor="or-resolution">Resolution</label>
                <select id="or-resolution" className="select" value={orResolution} onChange={(event) => setOrResolution(event.target.value)}>
                  {(orModels.find((item) => item.id === orModel)?.resolutions || ["480p", "768p", "2K"]).map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </div>
              <div className="row">
                <label htmlFor="or-duration">Duration</label>
                <select id="or-duration" className="select" value={orDuration} onChange={(event) => setOrDuration(Number(event.target.value))}>
                  {(orModels.find((item) => item.id === orModel)?.durations || [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).map((value) => (
                    <option key={value} value={value}>{value} seconds</option>
                  ))}
                </select>
              </div>
              {orQuote ? (
                <p className="meta">
                  {orQuote.name} · {orQuote.duration}s · {orQuote.resolution} · {orQuote.aspect_ratio}
                  {" · "}estimated ${orQuote.estimated_usd}
                  {orQuote.rate_unit === "second" ? ` ($${orQuote.rate_usd} per second)` : ""}
                  {" · "}budget ${orQuote.budget_usd}
                  {orQuote.within_budget ? " · within budget" : " · over budget"}
                  {typeof orQuote.max_scenes === "number" ? ` · up to ${orQuote.max_scenes} scenes` : ""}
                  {orQuote.submitted ? "" : " · not submitted"}
                </p>
              ) : <p className="meta">{orNotice || "Checking the OpenRouter price…"}</p>}
              <div className="row">
                <label htmlFor="project-budget">Project budget (USD)</label>
                <input id="project-budget" className="field" inputMode="decimal" value={projectBudget} onChange={(event) => setProjectBudget(event.target.value)} />
              </div>
              <div className="row">
                <label htmlFor="visual-style">Visual style</label>
                <input id="visual-style" className="field" value={visualStyle} placeholder="cinematic, natural light" onChange={(event) => setVisualStyle(event.target.value)} />
              </div>
              <div className="row">
                <label htmlFor="scene-retries">Retries per scene</label>
                <select id="scene-retries" className="select" value={sceneRetries} onChange={(event) => setSceneRetries(Number(event.target.value))}>
                  <option value={0}>0</option>
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                </select>
              </div>
            </div>
          ) : null}
          {draft.video_source !== "local" && draft.video_source !== "loomloom" && draft.video_source !== "openrouter" ? (
            <p className="section-foot">Keys, models, and endpoints for {sourceLabels[draft.video_source] || draft.video_source} are in Settings.</p>
          ) : null}
          <div className="row">
            <label htmlFor="concat">Concat mode</label>
            <select id="concat" className="select" value={draft.match_materials_to_script ? "sequential" : draft.video_concat_mode} disabled={draft.match_materials_to_script} onChange={(event) => patch({ video_concat_mode: event.target.value })}>
              <option value="sequential">Sequential</option>
              <option value="random">Random</option>
            </select>
          </div>
          <Toggle id="match" label="Match materials to script order" checked={draft.match_materials_to_script} onChange={(value) => patch({ match_materials_to_script: value })} />
          <div className="row">
            <label htmlFor="transition">Transition</label>
            <select id="transition" className="select" value={draft.transition} onChange={(event) => patch({ transition: event.target.value })}>
              <option value="none">None</option>
              <option value="Shuffle">Shuffle</option>
              <option value="FadeIn">Fade in</option>
              <option value="FadeOut">Fade out</option>
              <option value="SlideIn">Slide in</option>
              <option value="SlideOut">Slide out</option>
              <option value="ZoomIn">Zoom in</option>
              <option value="ZoomOut">Zoom out</option>
            </select>
          </div>
          <div className="row">
            <span id="aspect-label">Video ratio</span>
            <div className="segments" role="group" aria-labelledby="aspect-label">
              {[["9:16", "Portrait"], ["16:9", "Landscape"], ["1:1", "Square"]].map(([value, label]) => (
                <button key={value} type="button" aria-pressed={draft.aspect === value} onClick={() => patch({ aspect: value })}>{label}</button>
              ))}
            </div>
          </div>
          <div className="row">
            <span id="fit-label">Fit</span>
            <div className="segments" role="group" aria-labelledby="fit-label">
              <button type="button" aria-pressed={draft.video_fit_mode === "cover"} onClick={() => patch({ video_fit_mode: "cover" })}>Fill and crop</button>
              <button type="button" aria-pressed={draft.video_fit_mode === "contain"} onClick={() => patch({ video_fit_mode: "contain" })}>Fit with bars</button>
            </div>
          </div>
          <div className="row">
            <label htmlFor="clip">Clip duration</label>
            <select id="clip" className="select" value={draft.clip_duration} onChange={(event) => patch({ clip_duration: Number(event.target.value) })}>
              {clipChoices.map((seconds) => <option key={seconds} value={seconds}>{seconds} seconds</option>)}
            </select>
          </div>
          <div className="row">
            <label htmlFor="speed">Clip speed ({draft.video_clip_speed.toFixed(2)}x)</label>
            <input id="speed" className="field" type="range" min="0.5" max="2" step="0.05" value={draft.video_clip_speed} onChange={(event) => patch({ video_clip_speed: Number(event.target.value) })} />
          </div>
          <div className="row">
            <label htmlFor="count">Videos at once</label>
            <select id="count" className="select" value={draft.video_count} onChange={(event) => patch({ video_count: Number(event.target.value) })}>
              {[1, 2, 3, 4, 5].map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
          </div>
          <div className="row">
            <label htmlFor="codec">Encoder</label>
            <select id="codec" className="select" value={draft.video_codec} onChange={(event) => patch({ video_codec: event.target.value })}>
              <option value="__default__">Default</option>
              <option value="libx264">libx264 (CPU)</option>
              <option value="h264_nvenc">NVIDIA NVENC</option>
              <option value="h264_amf">AMD AMF</option>
              <option value="h264_qsv">Intel Quick Sync</option>
              <option value="h264_mf">Windows Media Foundation</option>
              <option value="h264_videotoolbox">macOS VideoToolbox</option>
            </select>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>Audio</h2>
        <div className="group">
          <div className="row">
            <span id="voice-mode-label">Voiceover</span>
            <div className="segments" role="group" aria-labelledby="voice-mode-label">
              <button type="button" aria-pressed={draft.voice_mode === "tts"} onClick={() => patch({ voice_mode: "tts" })}>{t("Automatic", "Automatic")}</button>
              <button type="button" aria-pressed={draft.voice_mode === "upload"} onClick={() => patch({ voice_mode: "upload" })}>{t("Upload Voiceover", "Upload")}</button>
              <button type="button" aria-pressed={draft.voice_mode === "none"} onClick={() => patch({ voice_mode: "none" })}>{t("None", "None")}</button>
            </div>
          </div>
          {draft.voice_mode === "tts" ? (
            <>
              <div className="row">
                <label htmlFor="tts">{t("TTS Server", "Narration service")}</label>
                <select id="tts" className="select" value={draft.tts_server} onChange={(event) => void changeServer(event.target.value)}>
                  {(catalog?.tts_servers || [{ id: "azure-tts-v1", label: "Edge TTS" }]).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                </select>
              </div>
              <div className="row">
                <label htmlFor="voice">{t("Voice", "Voice")}</label>
                <select id="voice" className="select" value={draft.voice_name} onChange={(event) => patch({ voice_name: event.target.value })}>
                  {voices.map((voice) => <option key={voice.id} value={voice.id}>{voice.label}</option>)}
                </select>
              </div>
              <div className="block">
                <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void hearVoice()}>{t("Preview Voice", "Preview voice")}</button>
              </div>
              <div className="row">
                <label htmlFor="rate">{t("Voice Rate", "Voice rate")} ({draft.voice_rate.toFixed(2)})</label>
                <input id="rate" className="field" type="range" min="0.8" max="1.25" step="0.05" value={draft.voice_rate} onChange={(event) => patch({ voice_rate: Number(event.target.value) })} />
              </div>
              <div className="row">
                <label htmlFor="volume">{t("Voice Volume", "Voice volume")} ({draft.voice_volume.toFixed(2)})</label>
                <input id="volume" className="field" type="range" min="0.5" max="1.5" step="0.05" value={draft.voice_volume} onChange={(event) => patch({ voice_volume: Number(event.target.value) })} />
              </div>
              {draft.tts_server === "voxcpm" ? (
                <div className="block stack">
                  <label htmlFor="reference">Reference audio</label>
                  <input id="reference" className="field" type="file" accept="audio/*" onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (!file) return;
                    void uploadVoiceover(file).then(setReferenceToken).catch((reason: Error) => setError(reason.message));
                  }} />
                  <label htmlFor="prompt-audio">Prompt audio</label>
                  <input id="prompt-audio" className="field" type="file" accept="audio/*" onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (!file) return;
                    void uploadVoiceover(file).then(setPromptToken).catch((reason: Error) => setError(reason.message));
                  }} />
                  <label htmlFor="prompt-text">Prompt text</label>
                  <textarea id="prompt-text" className="area" value={draft.voxcpm_prompt_text} onChange={(event) => patch({ voxcpm_prompt_text: event.target.value })} />
                </div>
              ) : null}
            </>
          ) : null}
          {draft.voice_mode === "upload" ? (
            <div className="block">
              <label htmlFor="voiceover">{t("Upload Voiceover", "Voiceover file")}</label>
              <input id="voiceover" className="field" type="file" accept="audio/*" onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                setBusy("Saving the voiceover…");
                void uploadVoiceover(file)
                  .then((token) => {
                    setVoiceoverToken(token);
                    setVoiceoverName(file.name);
                  })
                  .catch((reason: Error) => setError(reason.message))
                  .finally(() => setBusy(""));
              }} />
              {voiceoverName ? <p className="meta">{voiceoverName}</p> : null}
            </div>
          ) : null}
          <div className="row">
            <label htmlFor="bgm">{t("Background Music", "Background music")}</label>
            <select id="bgm" className="select" value={draft.bgm_type} onChange={(event) => patch({ bgm_type: event.target.value })}>
              {(catalog?.bgm_types || [{ id: "", label: "No background music" }, { id: "random", label: "Random" }]).map((item) => <option key={item.id || "none"} value={item.id}>{item.label}</option>)}
            </select>
          </div>
          {draft.bgm_type === "preset" ? (
            <div className="row">
              <label htmlFor="song">Preset song</label>
              <select id="song" className="select" value={draft.bgm_file} onChange={(event) => patch({ bgm_file: event.target.value })}>
                <option value="">Choose a song</option>
                {(catalog?.songs || []).map((song) => <option key={song} value={song}>{song}</option>)}
              </select>
            </div>
          ) : null}
          {draft.bgm_type === "custom" ? (
            <div className="block">
              <label htmlFor="music-file">Music file</label>
              <input id="music-file" className="field" type="file" accept="audio/*" onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                setBusy("Saving the music…");
                void uploadMusic(file)
                  .then((name) => patch({ bgm_file: name }))
                  .catch((reason: Error) => setError(reason.message))
                  .finally(() => setBusy(""));
              }} />
              {draft.bgm_file ? <p className="meta">{draft.bgm_file}</p> : null}
            </div>
          ) : null}
          {draft.bgm_type === "sonilo" || draft.bgm_type === "elevenlabs" ? (
            <div className="block">
              <label htmlFor="music-prompt">Music prompt</label>
              <textarea id="music-prompt" className="area" value={draft.video_music_prompt} onChange={(event) => patch({ video_music_prompt: event.target.value })} />
            </div>
          ) : null}
          {draft.bgm_type ? (
            <div className="row">
              <label htmlFor="music">{t("Music Volume", "Music level")} ({draft.bgm_volume.toFixed(2)})</label>
              <input id="music" className="field" type="range" min="0" max="0.4" step="0.02" value={draft.bgm_volume} onChange={(event) => patch({ bgm_volume: Number(event.target.value) })} />
            </div>
          ) : null}
        </div>
      </section>

      <section className="section">
        <h2>Subtitles</h2>
        <div className="group">
          <Toggle id="subs" label="Enable subtitles" checked={draft.subtitle_enabled} onChange={(value) => patch({ subtitle_enabled: value })} />
          <div className="row">
            <label htmlFor="font">Font</label>
            <select id="font" className="select" value={draft.font_name} disabled={!draft.subtitle_enabled} onChange={(event) => patch({ font_name: event.target.value })}>
              {settings.fonts.map((font) => <option key={font.id} value={font.id}>{font.label}</option>)}
            </select>
          </div>
          <div className="row">
            <label htmlFor="position">Position</label>
            <select id="position" className="select" value={draft.subtitle_position} disabled={!draft.subtitle_enabled} onChange={(event) => patch({ subtitle_position: event.target.value })}>
              <option value="top">Top</option>
              <option value="center">Center</option>
              <option value="bottom">Bottom</option>
              <option value="two_thirds_bottom">2/3 from bottom</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          {draft.subtitle_position === "custom" ? (
            <div className="row">
              <label htmlFor="custom">Custom position ({draft.custom_position}%)</label>
              <input id="custom" className="field" type="range" min="0" max="100" step="1" disabled={!draft.subtitle_enabled} value={draft.custom_position} onChange={(event) => patch({ custom_position: Number(event.target.value) })} />
            </div>
          ) : null}
          <div className="row">
            <label htmlFor="display">Display mode</label>
            <select id="display" className="select" value={draft.subtitle_display_mode} disabled={!draft.subtitle_enabled} onChange={(event) => patch({ subtitle_display_mode: event.target.value })}>
              <option value="sentence">Sentence by sentence</option>
              <option value="word_by_word">Single word</option>
            </select>
          </div>
          <div className="row">
            <label htmlFor="animation">Animation</label>
            <select id="animation" className="select" value={draft.subtitle_animation} disabled={!draft.subtitle_enabled} onChange={(event) => patch({ subtitle_animation: event.target.value })}>
              <option value="none">None</option>
              <option value="pop_spring">Pop</option>
            </select>
          </div>
          <div className="row">
            <label htmlFor="size">Font size ({draft.font_size})</label>
            <input id="size" className="field" type="range" min="30" max="100" step="1" disabled={!draft.subtitle_enabled} value={draft.font_size} onChange={(event) => patch({ font_size: Number(event.target.value) })} />
          </div>
          <div className="row">
            <label htmlFor="ink">Text color</label>
            <input id="ink" className="field" type="color" disabled={!draft.subtitle_enabled} value={draft.text_fore_color} onChange={(event) => patch({ text_fore_color: event.target.value.toUpperCase() })} />
          </div>
          <div className="row">
            <label htmlFor="stroke">Stroke color</label>
            <input id="stroke" className="field" type="color" disabled={!draft.subtitle_enabled} value={draft.stroke_color} onChange={(event) => patch({ stroke_color: event.target.value.toUpperCase() })} />
          </div>
          <div className="row">
            <label htmlFor="stroke-width">Stroke width ({draft.stroke_width.toFixed(1)})</label>
            <input id="stroke-width" className="field" type="range" min="0" max="10" step="0.1" disabled={!draft.subtitle_enabled} value={draft.stroke_width} onChange={(event) => patch({ stroke_width: Number(event.target.value) })} />
          </div>
          <Toggle id="bar" label="Subtitle background" checked={draft.use_caption_bar} onChange={(value) => patch({ use_caption_bar: value })} />
          <div className="row">
            <label htmlFor="bar-color">Background color</label>
            <input id="bar-color" className="field" type="color" disabled={!draft.subtitle_enabled || !draft.use_caption_bar} value={draft.text_background_color} onChange={(event) => patch({ text_background_color: event.target.value.toUpperCase() })} />
          </div>
          <Toggle id="rounded" label="Rounded subtitle background" checked={draft.rounded_subtitle_background} onChange={(value) => patch({ rounded_subtitle_background: value })} />
          <div className="block">
            <button className="btn-quiet" type="button" onClick={() => patch({
              subtitle_enabled: true,
              subtitle_position: "bottom",
              custom_position: 70,
              subtitle_display_mode: "sentence",
              subtitle_animation: "none",
              font_size: 60,
              text_fore_color: "#FFFFFF",
              stroke_color: "#000000",
              stroke_width: 1.5,
              use_caption_bar: false,
              rounded_subtitle_background: false,
              subtitle_provider: "edge",
            })}>{t("Restore Default Subtitles", "Restore default subtitles")}</button>
          </div>
          {captionWarning ? <p className="notice">{captionWarning}</p> : null}
          <div className="row">
            <label htmlFor="timing">{t("Subtitle Provider", "Timing")}</label>
            <select id="timing" className="select" value={draft.subtitle_provider} disabled={!draft.subtitle_enabled} onChange={(event) => patch({ subtitle_provider: event.target.value })}>
              <option value="edge">Edge timestamps</option>
              <option value="whisper">Local Whisper</option>
            </select>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>Generate</h2>
        <div className="group">
          <div className="block stack">
            <button className="btn btn-block" type="button" disabled={Boolean(busy) || task?.state === 4} onClick={() => void generate()}>
              {task?.state === 4 ? "Generating…" : t("Generate Video", "Generate video")}
            </button>
            <div className="actions">
              <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void generatePart("audio")}>{t("Generate Audio", "Audio only")}</button>
              <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void generatePart("subtitle")}>{t("Generate Subtitle", "Subtitles only")}</button>
              <button className="btn-quiet" type="button" disabled={Boolean(busy)} onClick={() => void socialCopy()}>{t("Generate Social Metadata", "Write a post")}</button>
            </div>
            <div className="row">
              <label htmlFor="threads">{t("Threads", "Encoder threads")}</label>
              <input id="threads" className="field" type="number" min="1" max="16" value={draft.n_threads} onChange={(event) => patch({ n_threads: Number(event.target.value) })} />
            </div>
            <div className="actions">
              <button className="btn-quiet" type="button" onClick={exportPreset}>{t("Export Settings", "Export settings")}</button>
              <label className="btn-quiet" htmlFor="preset">
                {t("Import Settings", "Import settings")}
                <input id="preset" hidden type="file" accept="application/json" onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void importPreset(file);
                }} />
              </label>
            </div>
          </div>
          {notice ? <p className="notice" role="status">{notice}</p> : null}
          {social ? <pre className="area">{social}</pre> : null}
          {task ? (
            <div className="block stack">
              <progress className="meter" max={100} value={task.progress} />
              <p role="status">
                {task.state === -1
                  ? `Generation failed${task.failed_stage ? ` during ${task.failed_stage}` : ""}. ${task.error || "Check the API log for the stage that stopped."}`
                  : `${progressLabel(task.progress)} · ${task.progress}%`}
              </p>
              {task.autonomous_spent_usd ? <p className="meta">Spent ${task.autonomous_spent_usd}{task.cv_report?.disclaimer ? `. ${task.cv_report.disclaimer}` : ""}</p> : null}
              {task.scene_status?.length ? (
                <ul className="meta">
                  {task.scene_status.map((scene) => (
                    <li key={`${scene.index}-${scene.status}`}>
                      Scene {Number(scene.index) + 1}: {scene.status}
                      {typeof scene.score === "number" ? ` · score ${scene.score.toFixed(2)}` : ""}
                      {scene.warnings?.length ? ` · ${scene.warnings.join("; ")}` : ""}
                    </li>
                  ))}
                </ul>
              ) : null}
              {video ? (
                <>
                  <video className={`player${draft.aspect === "16:9" ? " wide" : ""}`} controls src={video} />
                  <a className="btn" href={video} download>Download MP4</a>
                </>
              ) : null}
              {task.audio_file ? <a className="btn-quiet" href={task.audio_file} download>Download audio</a> : null}
              {task.subtitle_path ? <a className="btn-quiet" href={task.subtitle_path} download>Download subtitles</a> : null}
              {!settings.hide_log && logs.length ? <pre className="area" style={{ maxHeight: "220px", overflow: "auto" }}>{logs.join("\n")}</pre> : null}
            </div>
          ) : null}
        </div>
      </section>
    </>
  );
}

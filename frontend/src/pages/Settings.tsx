import { useEffect, useRef, useState, type FormEvent } from "react";
import { cleanCache, exportKeys, importKeys, readCache, readCatalog, readGroqModels, readSettings, saveSettings, testLanguageModel } from "../api";
import { useT } from "../i18n";
import type { BenchSettings } from "../types";

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

type SecretField = { id: string; label: string; flag: keyof BenchSettings };
type TextField = { id: keyof BenchSettings; label: string };
type SourceBundle = { id: string; title: string; secrets: SecretField[]; texts: TextField[] };

const stockSecrets: SecretField[] = [
  { id: "pexels_secret", label: "Pexels key", flag: "pexels_configured" },
  { id: "pixabay_secret", label: "Pixabay key", flag: "pixabay_configured" },
  { id: "coverr_secret", label: "Coverr key", flag: "coverr_configured" },
  { id: "twelvelabs_secret", label: "Twelve Labs key", flag: "twelvelabs_configured" },
];

const videoSources: SourceBundle[] = [
  { id: "wavespeed", title: "WaveSpeed", secrets: [{ id: "wavespeed_secret", label: "Key", flag: "wavespeed_configured" }], texts: [{ id: "wavespeed_model", label: "Model" }] },
  { id: "volcengine_seedance", title: "Volcano Engine Seedance", secrets: [{ id: "seedance_secret", label: "Key", flag: "seedance_configured" }], texts: [{ id: "seedance_base_url", label: "Base URL" }, { id: "seedance_model", label: "Model" }, { id: "seedance_resolution", label: "Resolution" }] },
  { id: "ofox", title: "OFox", secrets: [{ id: "ofox_secret", label: "Key", flag: "ofox_configured" }], texts: [{ id: "ofox_base_url", label: "Base URL" }, { id: "ofox_model", label: "Model" }, { id: "ofox_resolution", label: "Resolution" }, { id: "ofox_provider", label: "Vendor" }] },
  { id: "metaso_minimax", title: "Metaso MiniMax", secrets: [{ id: "metaso_secret", label: "Key", flag: "metaso_configured" }], texts: [{ id: "metaso_base_url", label: "Base URL" }, { id: "metaso_resolution", label: "Resolution" }] },
  { id: "muapi", title: "MuAPI", secrets: [{ id: "muapi_secret", label: "Key", flag: "muapi_configured" }], texts: [{ id: "muapi_base_url", label: "Base URL" }, { id: "muapi_endpoint", label: "Endpoint" }, { id: "muapi_resolution", label: "Resolution" }] },
  { id: "loomloom", title: "Shengsuan Cloud", secrets: [{ id: "loomloom_secret", label: "Key", flag: "loomloom_configured" }], texts: [{ id: "loomloom_base_url", label: "Base URL" }] },
  { id: "openrouter", title: "OpenRouter video", secrets: [], texts: [{ id: "openrouter_video_model", label: "Model" }, { id: "openrouter_video_resolution", label: "Resolution" }] },
];

const imageSource: SourceBundle = {
  id: "openai_image",
  title: "OpenAI-compatible images",
  secrets: [{ id: "openai_image_secret", label: "Key", flag: "openai_image_configured" }],
  texts: [
    { id: "openai_image_base_url", label: "Base URL" },
    { id: "openai_image_model", label: "Model" },
    { id: "openai_image_size", label: "Size" },
    { id: "openai_image_prompt_template", label: "Prompt template" },
  ],
};

const voiceSources: SourceBundle[] = [
  { id: "azure", title: "Azure Speech", secrets: [{ id: "azure_speech_secret", label: "Key", flag: "azure_speech_configured" }], texts: [{ id: "azure_speech_region", label: "Region" }] },
  { id: "siliconflow", title: "SiliconFlow", secrets: [{ id: "siliconflow_secret", label: "Key", flag: "siliconflow_configured" }], texts: [{ id: "siliconflow_base_url", label: "Base URL" }] },
  { id: "gemini", title: "Gemini", secrets: [{ id: "gemini_secret", label: "Key", flag: "gemini_configured" }], texts: [] },
  { id: "mimo", title: "MiMo", secrets: [{ id: "mimo_secret", label: "Key", flag: "mimo_configured" }], texts: [] },
  { id: "minimax", title: "MiniMax", secrets: [{ id: "minimax_tts_secret", label: "Key", flag: "minimax_tts_configured" }], texts: [{ id: "minimax_tts_base_url", label: "Base URL" }, { id: "minimax_tts_model", label: "Model" }] },
  { id: "elevenlabs", title: "ElevenLabs", secrets: [{ id: "elevenlabs_secret", label: "Key", flag: "elevenlabs_configured" }], texts: [{ id: "elevenlabs_model", label: "Voice model" }] },
  { id: "chatterbox", title: "Chatterbox", secrets: [{ id: "chatterbox_secret", label: "Key", flag: "chatterbox_configured" }], texts: [{ id: "chatterbox_base_url", label: "Base URL" }] },
  { id: "kokoro", title: "Kokoro", secrets: [{ id: "kokoro_secret", label: "Key", flag: "kokoro_configured" }], texts: [{ id: "kokoro_base_url", label: "Base URL" }] },
  { id: "fish", title: "Fish Audio", secrets: [{ id: "fish_secret", label: "Key", flag: "fish_configured" }], texts: [{ id: "fish_model", label: "Model" }] },
  { id: "voxcpm", title: "VoxCPM", secrets: [{ id: "voxcpm_secret", label: "Key", flag: "voxcpm_configured" }], texts: [{ id: "voxcpm_base_url", label: "Base URL" }, { id: "voxcpm_model", label: "Model" }, { id: "voxcpm_voice", label: "Voice" }] },
];

const musicFields: SourceBundle = {
  id: "music",
  title: "Music",
  secrets: [{ id: "sonilo_secret", label: "Sonilo key", flag: "sonilo_configured" }],
  texts: [{ id: "sonilo_base_url", label: "Sonilo base URL" }, { id: "elevenlabs_music_model", label: "ElevenLabs music model" }],
};

const localeNames: Record<string, string> = {
  en: "English", zh: "中文", de: "Deutsch", es: "Español", fr: "Français", pt: "Português",
  ru: "Русский", ko: "한국어", vi: "Tiếng Việt", tr: "Türkçe", it: "Italiano", id: "Bahasa Indonesia",
  ca: "Català", az: "Azərbaycan",
};

function bundleReady(bundle: SourceBundle, form: BenchSettings) {
  if (bundle.secrets.length === 0) return Boolean(form.openrouter_configured);
  return bundle.secrets.some((field) => Boolean(form[field.flag]));
}

function SecretRow({ field, form, value, onChange }: { field: SecretField; form: BenchSettings; value: string; onChange: (value: string) => void }) {
  return (
    <div className="row">
      <label htmlFor={field.id}>{field.label}</label>
      <input id={field.id} className="field" type="password" autoComplete="off" value={value} placeholder={form[field.flag] ? "Saved" : "Not set"} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function TextRow({ field, value, onChange }: { field: TextField; value: string; onChange: (value: string) => void }) {
  return (
    <div className="row">
      <label htmlFor={String(field.id)}>{field.label}</label>
      <input id={String(field.id)} className="field" value={value} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function SourceDisclosure({
  bundle,
  form,
  secrets,
  setSecret,
  setText,
}: {
  bundle: SourceBundle;
  form: BenchSettings;
  secrets: Record<string, string>;
  setSecret: (id: string, value: string) => void;
  setText: (id: keyof BenchSettings, value: string) => void;
}) {
  const panel = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (panel.current && form.video_source === bundle.id) panel.current.open = true;
  }, [form.video_source, bundle.id]);
  return (
    <details className="source" ref={panel}>
      <summary>
        <span>{bundle.title}</span>
        <span className="meta">{bundle.secrets.length === 0
          ? (bundleReady(bundle, form) ? "Uses the saved OpenRouter key" : "OpenRouter key required")
          : (bundleReady(bundle, form) ? "Key saved" : "Not set")}</span>
      </summary>
      <div>
        {bundle.id === "openrouter" ? <p className="meta">Video uses the OpenRouter key already saved for the language model. Leave the model blank to use HeyGen Video.</p> : null}
        {bundle.secrets.map((field) => (
          <SecretRow key={field.id} field={field} form={form} value={secrets[field.id] || ""} onChange={(value) => setSecret(field.id, value)} />
        ))}
        {bundle.texts.map((field) => (
          <TextRow key={String(field.id)} field={field} value={String(form[field.id] || "")} onChange={(value) => setText(field.id, value)} />
        ))}
      </div>
    </details>
  );
}

export function Settings() {
  const t = useT();
  const [form, setForm] = useState<BenchSettings | null>(null);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [locales, setLocales] = useState<string[]>(Object.keys(localeNames));
  const [groqModels, setGroqModels] = useState<string[]>([]);
  const [cache, setCache] = useState("");
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    readSettings()
      .then((data) => setForm(data.settings))
      .catch((reason: Error) => setError(reason.message));
    readCatalog()
      .then((data) => setLocales(data.ui_locales?.length ? data.ui_locales : Object.keys(localeNames)))
      .catch(() => undefined);
    readCache()
      .then((data) => setCache(`${data.file_count} files · ${Math.round(data.total_size / 1024 / 1024)} MB`))
      .catch(() => setCache("Unavailable"));
  }, []);

  function set<K extends keyof BenchSettings>(key: K, value: BenchSettings[K]) {
    setForm((current) => current && ({ ...current, [key]: value }));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!form) return;
    setBusy(true);
    setError("");
    setSaved("");
    try {
      const result = await saveSettings({
        ...form,
        openrouter_model_name: form.model_override || form.model_name,
        ...secrets,
      });
      setForm(result.settings);
      setSecrets({});
      window.dispatchEvent(new CustomEvent("frameforge-language", { detail: result.settings.ui_language || "en" }));
      setSaved("Saved. A blank key field keeps the key already stored on this computer.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Settings could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function checkModel() {
    setBusy(true);
    setError("");
    setSaved("");
    try {
      const result = await testLanguageModel();
      setSaved(result.ok ? `Language model responded in ${result.elapsed}s.` : result.error);
      if (!result.ok) setError(result.error);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The language model could not be reached.");
    } finally {
      setBusy(false);
    }
  }

  async function clearCache(days: number | null) {
    setBusy(true);
    setError("");
    try {
      const result = await cleanCache(days);
      setSaved(`Removed ${result.deleted_count} cached files.`);
      const next = await readCache();
      setCache(`${next.file_count} files · ${Math.round(next.total_size / 1024 / 1024)} MB`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The cache could not be cleared.");
    } finally {
      setBusy(false);
    }
  }

  async function downloadKeys() {
    setBusy(true);
    setError("");
    try {
      const payload = await exportKeys();
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "frameforge-keys.json";
      link.click();
      setSaved("Key file downloaded. Keep it private.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Keys could not be exported.");
    } finally {
      setBusy(false);
    }
  }

  async function restoreKeys(file: File) {
    setBusy(true);
    setError("");
    try {
      const payload = JSON.parse(await file.text()) as unknown;
      await importKeys(payload);
      const data = await readSettings();
      setForm(data.settings);
      setSaved("Keys restored on this computer.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "That key file could not be restored.");
    } finally {
      setBusy(false);
    }
  }

  if (!form) {
    return error ? <p className="notice notice-bad" role="alert">{error}</p> : <p className="meta">Loading settings…</p>;
  }

  const provider = form.providers.find((item) => item.id === form.llm_provider);
  const platforms = form.upload_post_platforms || [];

  return (
    <form onSubmit={(event) => void save(event)}>
      <h1 className="page-title">{t("Settings", "Settings")}</h1>
      <p className="lede">Keys stay on this computer. A blank key field keeps the key already stored.</p>

      <section className="section">
        <h2>{t("LLM Settings Tab", "Language model")}</h2>
        <div className="group">
          <div className="row">
            <label htmlFor="provider">{t("LLM Provider", "Provider")}</label>
            <select id="provider" className="select" value={form.llm_provider} onChange={(event) => set("llm_provider", event.target.value)}>
              {form.providers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
          </div>
          <div className="row">
            <label htmlFor="model">{t("LLM Model", "Model")}</label>
            {form.llm_provider === "groq" && groqModels.length ? (
              <select id="model" className="select" value={form.model_override || form.model_name} onChange={(event) => set("model_override", event.target.value)}>
                {groqModels.map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            ) : (
              <input id="model" className="field" value={form.model_override || form.model_name} onChange={(event) => set("model_override", event.target.value)} />
            )}
          </div>
          {form.llm_provider === "groq" ? (
            <div className="block">
              <button className="btn-quiet" type="button" disabled={busy} onClick={() => {
                setBusy(true);
                readGroqModels().then((data) => setGroqModels(data.models)).catch((reason: Error) => setError(reason.message)).finally(() => setBusy(false));
              }}>Load Groq models</button>
            </div>
          ) : null}
          {provider?.show_base_url !== false ? (
            <div className="row">
              <label htmlFor="base">{t("Base URL", "Base URL")}</label>
              <input id="base" className="field" value={form.llm_base_url || ""} placeholder="Leave blank for the provider default" onChange={(event) => set("llm_base_url", event.target.value)} />
            </div>
          ) : null}
          {provider?.show_secret !== false ? (
            <div className="row">
              <label htmlFor="llm-key">{t("API Key", "API key")}</label>
              <input id="llm-key" className="field" type="password" autoComplete="off" value={secrets.llm_secret || ""} placeholder={form.llm_secret_set || form.openrouter_configured ? "Saved" : "Not set"} onChange={(event) => setSecrets({ ...secrets, llm_secret: event.target.value })} />
            </div>
          ) : null}
          <div className="block">
            <button className="btn-quiet" type="button" disabled={busy} onClick={() => void checkModel()}>{t("Test LLM Connection", "Test connection")}</button>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>{t("Material API Tab", "Material sources")}</h2>
        <div className="group">
          <div className="row">
            <label htmlFor="source">{t("Video Source", "Video source")}</label>
            <select id="source" className="select" value={form.video_source} onChange={(event) => set("video_source", event.target.value)}>
              {form.sources.map((source) => <option key={source} value={source}>{sourceLabels[source] || source}</option>)}
            </select>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>{t("Stock Video", "Stock video")}</h2>
        <div className="group">
          {stockSecrets.map((field) => (
            <SecretRow key={field.id} field={field} form={form} value={secrets[field.id] || ""} onChange={(value) => setSecrets({ ...secrets, [field.id]: value })} />
          ))}
        </div>
      </section>

      <section className="section">
        <h2>{t("AI Video", "AI video")}</h2>
        <div className="group">
          {videoSources.map((bundle) => (
            <SourceDisclosure key={`${bundle.id}-${form.video_source}`} bundle={bundle} form={form} secrets={secrets} setSecret={(id, value) => setSecrets({ ...secrets, [id]: value })} setText={(id, value) => set(id, value)} />
          ))}
        </div>
        <p className="section-foot">Open a source to set its key, address, and model. The source selected above starts open.</p>
      </section>

      <section className="section">
        <h2>{t("AI Image", "AI image")}</h2>
        <div className="group">
          <SourceDisclosure key={`image-${form.video_source}`} bundle={imageSource} form={form} secrets={secrets} setSecret={(id, value) => setSecrets({ ...secrets, [id]: value })} setText={(id, value) => set(id, value)} />
        </div>
      </section>

      <section className="section">
        <h2>{t("Narration", "Narration")}</h2>
        <div className="group">
          {voiceSources.map((bundle) => (
            <SourceDisclosure key={bundle.id} bundle={bundle} form={form} secrets={secrets} setSecret={(id, value) => setSecrets({ ...secrets, [id]: value })} setText={(id, value) => set(id, value)} />
          ))}
        </div>
      </section>

      <section className="section">
        <h2>{t("Background Music", "Music")}</h2>
        <div className="group">
          {musicFields.secrets.map((field) => (
            <SecretRow key={field.id} field={field} form={form} value={secrets[field.id] || ""} onChange={(value) => setSecrets({ ...secrets, [field.id]: value })} />
          ))}
          {musicFields.texts.map((field) => (
            <TextRow key={String(field.id)} field={field} value={String(form[field.id] || "")} onChange={(value) => set(field.id, value)} />
          ))}
        </div>
      </section>

      <section className="section">
        <h2>{t("Auto-Publish Settings", "Auto-publish")}</h2>
        <div className="group">
          <div className="row">
            <span id="publish-label">{t("Enable Upload-Post", "Enable Upload-Post")}</span>
            <label className="switch">
              <input aria-labelledby="publish-label" type="checkbox" checked={Boolean(form.upload_post_enabled)} onChange={(event) => set("upload_post_enabled", event.target.checked)} />
              <span />
            </label>
          </div>
          <div className="row">
            <span id="auto-label">{t("Auto Upload", "Upload when a video finishes")}</span>
            <label className="switch">
              <input aria-labelledby="auto-label" type="checkbox" checked={Boolean(form.upload_post_auto_upload)} onChange={(event) => set("upload_post_auto_upload", event.target.checked)} />
              <span />
            </label>
          </div>
          <div className="row">
            <label htmlFor="upload-post-secret">{t("Upload-Post API Key", "Upload-Post key")}</label>
            <input id="upload-post-secret" className="field" type="password" autoComplete="off" value={secrets.upload_post_secret || ""} placeholder={form.upload_post_configured ? "Saved" : "Not set"} onChange={(event) => setSecrets({ ...secrets, upload_post_secret: event.target.value })} />
          </div>
          <div className="row">
            <label htmlFor="profile">{t("Upload-Post Profile Username", "Profile username")}</label>
            <input id="profile" className="field" value={form.upload_post_username || ""} onChange={(event) => set("upload_post_username", event.target.value)} />
          </div>
          <div className="block">
            <span id="platforms-label">{t("Platforms", "Platforms")}</span>
            <div className="segments" role="group" aria-labelledby="platforms-label">
              {["tiktok", "instagram", "youtube"].map((platform) => (
                <button key={platform} type="button" aria-pressed={platforms.includes(platform)} onClick={() => {
                  const next = platforms.includes(platform) ? platforms.filter((item) => item !== platform) : [...platforms, platform];
                  set("upload_post_platforms", next);
                }}>{platform}</button>
              ))}
            </div>
          </div>
          {platforms.includes("youtube") ? (
            <>
              <div className="row">
                <label htmlFor="privacy">{t("YouTube Privacy Status", "YouTube privacy")}</label>
                <select id="privacy" className="select" value={form.upload_post_youtube_privacy || "public"} onChange={(event) => set("upload_post_youtube_privacy", event.target.value)}>
                  <option value="public">Public</option>
                  <option value="unlisted">Unlisted</option>
                  <option value="private">Private</option>
                </select>
              </div>
              <div className="row">
                <span id="kids-label">{t("YouTube Audience", "Made for kids")}</span>
                <label className="switch">
                  <input aria-labelledby="kids-label" type="checkbox" checked={Boolean(form.upload_post_youtube_made_for_kids)} onChange={(event) => set("upload_post_youtube_made_for_kids", event.target.checked)} />
                  <span />
                </label>
              </div>
            </>
          ) : null}
        </div>
      </section>

      <section className="section">
        <h2>{t("Interface Settings Tab", "Interface")}</h2>
        <div className="group">
          <div className="row">
            <label htmlFor="ui-language">{t("Language", "Language")}</label>
            <select id="ui-language" className="select" value={form.ui_language || "en"} onChange={(event) => set("ui_language", event.target.value)}>
              {locales.map((code) => <option key={code} value={code}>{localeNames[code] || code}</option>)}
            </select>
          </div>
          <div className="row">
            <span id="hide-log-label">{t("Hide Log", "Hide generation log")}</span>
            <label className="switch">
              <input aria-labelledby="hide-log-label" type="checkbox" checked={Boolean(form.hide_log)} onChange={(event) => set("hide_log", event.target.checked)} />
              <span />
            </label>
          </div>
          <div className="row">
            <span id="open-folder-label">{t("Open Task Folder", "Open the folder when a video finishes")}</span>
            <label className="switch">
              <input aria-labelledby="open-folder-label" type="checkbox" checked={form.open_task_folder_on_completion !== false} onChange={(event) => set("open_task_folder_on_completion", event.target.checked)} />
              <span />
            </label>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>{t("Cache Management Tab", "Cache")}</h2>
        <div className="group">
          <div className="row"><span>{t("Video Cache", "Downloaded clips")}</span><span className="meta">{cache || "Checking…"}</span></div>
          <div className="block">
            <div className="actions">
              <button className="btn-quiet" type="button" disabled={busy} onClick={() => void clearCache(7)}>{t("Clean Old Cache", "Delete clips older than 7 days")}</button>
              <button className="btn-danger" type="button" disabled={busy} onClick={() => void clearCache(null)}>{t("Clean All Cache", "Delete all cached clips")}</button>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>{t("Key Backup Tab", "Key backup")}</h2>
        <div className="group">
          <div className="block stack">
            <p className="meta">The download contains the keys stored on this computer. Do not share that file.</p>
            <div className="actions">
              <button className="btn-quiet" type="button" disabled={busy} onClick={() => void downloadKeys()}>{t("Export Keys", "Download keys")}</button>
              <label className="btn-quiet" htmlFor="key-file">
                {t("Import Keys", "Restore keys")}
                <input id="key-file" hidden type="file" accept="application/json" onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void restoreKeys(file);
                }} />
              </label>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>{t("This machine", "This machine")}</h2>
        <div className="group">
          <div className="row"><span>FFmpeg</span><span className="meta">{form.ffmpeg_ready ? "Ready" : "Not found"}</span></div>
          <div className="row"><span>Active model</span><span className="meta">{form.model_name}</span></div>
        </div>
      </section>

      {error ? <p className="notice notice-bad" role="alert">{error}</p> : null}
      {saved && !error ? <p className="notice" role="status">{saved}</p> : null}
      <div className="actions">
        <button className="btn" type="submit" disabled={busy}>{busy ? "Saving…" : t("Save", "Save")}</button>
      </div>
    </form>
  );
}

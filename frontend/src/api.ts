import type { BenchSettings, Clip, LibraryItem, Project, ProjectSummary, RenderTask } from "./types";

export function completeProject(project: Partial<Project>): Project {
  const filled = {
    video_source: "pexels",
    video_fit_mode: "cover",
    video_concat_mode: "sequential",
    video_clip_speed: 1,
    video_count: 1,
    match_materials_to_script: true,
    subtitle_display_mode: "sentence",
    subtitle_animation: "none",
    custom_position: 70,
    stroke_color: "#141210",
    paragraph_number: 1,
    video_language: "English",
    ...project,
  };
  return filled as Project;
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError("The Script-to-Reality Renderer Studio API is not reachable on this machine.", 0);
  }
  let body: { status?: number; message?: string; data?: T } = {};
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The server returned an unreadable response.", response.status);
  }
  if (!response.ok || (typeof body.status === "number" && body.status >= 400)) {
    throw new ApiError(body.message || "The request failed.", response.status || body.status || 500);
  }
  return body.data as T;
}

export function stamp(iso: string): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function progressLabel(progress: number): string {
  if (progress < 15) return "Preparing the narration";
  if (progress < 30) return "Recording the voice";
  if (progress < 45) return "Timing the captions";
  if (progress < 70) return "Cutting the footage";
  if (progress < 100) return "Rendering the master";
  return "Render complete";
}

export async function createProject(topic: string) {
  const result = await api<{ project: Project }>("/api/v1/frameforge/projects", {
    method: "POST",
    body: JSON.stringify({ topic }),
  });
  return { project: completeProject(result.project) };
}

export function listProjects() {
  return api<{ projects: ProjectSummary[] }>("/api/v1/frameforge/projects");
}

export async function readProject(id: string) {
  const result = await api<{ project: Project }>(`/api/v1/frameforge/projects/${id}`);
  return { project: completeProject(result.project) };
}

export async function saveProject(id: string, patch: Partial<Project>) {
  const result = await api<{ project: Project }>(`/api/v1/frameforge/projects/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  return { project: completeProject(result.project) };
}

export function deleteProject(id: string) {
  return api<{ deleted: string }>(`/api/v1/frameforge/projects/${id}`, { method: "DELETE" });
}

export async function writeScript(id: string) {
  const result = await api<{ project: Project }>(`/api/v1/frameforge/projects/${id}/script`, { method: "POST" });
  return { project: completeProject(result.project) };
}

export async function buildStoryboard(id: string) {
  const result = await api<{ project: Project }>(`/api/v1/frameforge/projects/${id}/storyboard`, { method: "POST" });
  return { project: completeProject(result.project) };
}

export function searchScene(id: string, sceneId: string, term: string) {
  return api<{ clips: Clip[] }>(`/api/v1/frameforge/projects/${id}/scenes/${sceneId}/search`, {
    method: "POST",
    body: JSON.stringify({ term }),
  });
}

export async function renderProject(id: string, patch: Partial<Project>) {
  const result = await api<{ project: Project; task_id: string }>(`/api/v1/frameforge/projects/${id}/render`, {
    method: "POST",
    body: JSON.stringify(patch),
  });
  return { project: completeProject(result.project), task_id: result.task_id };
}

export function readTask(taskId: string) {
  return api<RenderTask>(`/api/v1/tasks/${taskId}`);
}

export function readLibrary() {
  return api<{ renders: LibraryItem[] }>("/api/v1/frameforge/library");
}

export function readSettings() {
  return api<{ settings: BenchSettings }>("/api/v1/frameforge/settings");
}

export function saveSettings(patch: Partial<BenchSettings> & Record<string, unknown>) {
  return api<{ settings: BenchSettings }>("/api/v1/frameforge/settings", {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function writeNarration(body: {
  video_subject: string;
  video_language: string;
  paragraph_number: number;
  video_script_prompt: string;
  custom_system_prompt: string;
}) {
  return api<{ video_script: string }>("/api/v1/scripts", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function writeTerms(body: {
  video_subject: string;
  video_script: string;
  amount: number;
  match_materials_to_script: boolean;
}) {
  return api<{ video_terms: string[] | string }>("/api/v1/terms", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function startVideo(body: Record<string, unknown>) {
  return api<{ task_id: string }>("/api/v1/frameforge/generate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function startAudio(body: Record<string, unknown>) {
  return api<{ task_id: string }>("/api/v1/audio", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function startSubtitle(body: Record<string, unknown>) {
  return api<{ task_id: string }>("/api/v1/subtitle", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function readCatalog() {
  return api<{
    languages: string[];
    tts_servers: { id: string; label: string }[];
    bgm_types: { id: string; label: string }[];
    songs: string[];
    music_files: string[];
    ui_locales: string[];
    default_system_prompt: string;
    script_backends: { id: string; label: string }[];
  }>("/api/v1/frameforge/catalog");
}

export function readProviderVoices(server: string) {
  return api<{ voices: { id: string; label: string }[] }>(`/api/v1/frameforge/provider-voices?server=${encodeURIComponent(server)}`);
}

export function previewPrompt(body: Record<string, unknown>) {
  return api<{ prompt: string }>("/api/v1/frameforge/prompt-preview", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function readTasks() {
  return api<{ tasks: RenderTask[]; total: number }>("/api/v1/tasks?page=1&page_size=50");
}

export function deleteTask(taskId: string) {
  return api<unknown>(`/api/v1/tasks/${taskId}`, { method: "DELETE" });
}

export function readTaskLogs(taskId: string) {
  return api<{ lines: string[] }>(`/api/v1/frameforge/tasks/${taskId}/logs`);
}

export function openTaskFolder(taskId: string) {
  return api<{ opened: string }>(`/api/v1/frameforge/tasks/${taskId}/open`, { method: "POST" });
}

export function readCache() {
  return api<{ file_count: number; total_size: number }>("/api/v1/frameforge/cache");
}

export function cleanCache(maxAgeDays: number | null) {
  return api<{ deleted_count: number; deleted_size: number; failed_count: number }>("/api/v1/frameforge/cache/clean", {
    method: "POST",
    body: JSON.stringify({ max_age_days: maxAgeDays }),
  });
}

export function exportKeys() {
  return api<{ app: Record<string, unknown>; sections: Record<string, Record<string, unknown>> }>("/api/v1/frameforge/key-backup");
}

export function importKeys(payload: unknown) {
  return api<{ restored: boolean }>("/api/v1/frameforge/key-backup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function readLocale(code: string) {
  return api<{ code: string; strings: Record<string, string> }>(`/api/v1/frameforge/locale?code=${encodeURIComponent(code || "en")}`);
}

export function readGroqModels() {
  return api<{ models: string[] }>("/api/v1/frameforge/groq-models");
}

export function checkSubtitles(body: Record<string, unknown>) {
  return api<{ colors: boolean; font: boolean }>("/api/v1/frameforge/subtitle-check", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function testLanguageModel() {
  return api<{ ok: boolean; error: string; elapsed: number }>("/api/v1/frameforge/llm-test", { method: "POST" });
}

export function previewVoice(body: Record<string, unknown>) {
  return api<{ file: string }>("/api/v1/frameforge/voice-preview", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function quoteScripts(body: Record<string, unknown>) {
  return api<{ token: string; task_count: number; currency: string; amount: string }>("/api/v1/frameforge/loomloom/scripts/quote", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function runScripts(token: string) {
  return api<{ run_id: string }>("/api/v1/frameforge/loomloom/scripts/run", {
    method: "POST",
    body: JSON.stringify({ token, confirm: true }),
  });
}

export function readScriptRun(runId: string) {
  return api<{ status: string; total: number; completed: number; failed: number; error: string; candidates: { script: string; terms: string[] }[]; errors: string[] }>(`/api/v1/frameforge/loomloom/scripts/${runId}`);
}

export function readVideoOptions() {
  return api<{ models: { id: string; label: string }[]; default_model: string; aspects: string[] }>("/api/v1/frameforge/loomloom/video-options");
}

export function readOpenRouterVideoModels() {
  return api<{ models: { id: string; name: string; durations: number[]; resolutions: string[]; aspects: string[] }[] }>("/api/v1/frameforge/openrouter/video-models");
}

export function quoteOpenRouterVideo(body: Record<string, unknown>) {
  return api<{
    model: string;
    name: string;
    duration: number;
    resolution: string;
    aspect_ratio: string;
    clips: number;
    rate_usd: string;
    rate_unit: string;
    estimated_usd: string;
    budget_usd: string;
    within_budget: boolean;
    requires_approval: boolean;
    submitted: boolean;
  }>("/api/v1/frameforge/openrouter/video-quote", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function quoteVideo(body: Record<string, unknown>) {
  return api<{ token: string; task_count: number; currency: string; amount: string }>("/api/v1/frameforge/loomloom/video/quote", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function writeSocial(body: Record<string, unknown>) {
  return api<Record<string, string>>("/api/v1/social-metadata", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function uploadVoiceover(file: File) {
  const payload = new FormData();
  payload.append("file", file);
  let response: Response;
  try {
    response = await fetch("/api/v1/frameforge/voiceover", { method: "POST", body: payload });
  } catch {
    throw new ApiError("The Script-to-Reality Renderer Studio API is not reachable on this machine.", 0);
  }
  let body: { status?: number; message?: string; data?: { token?: string } } = {};
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The server returned an unreadable response.", response.status);
  }
  if (!response.ok || !body.data?.token) {
    throw new ApiError(body.message || "The voiceover could not be saved.", response.status || 500);
  }
  return body.data.token;
}

export async function uploadMusic(file: File) {
  const payload = new FormData();
  payload.append("file", file);
  let response: Response;
  try {
    response = await fetch("/api/v1/musics", { method: "POST", body: payload });
  } catch {
    throw new ApiError("The Script-to-Reality Renderer Studio API is not reachable on this machine.", 0);
  }
  let body: { status?: number; message?: string; data?: { file?: string } } = {};
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The server returned an unreadable response.", response.status);
  }
  if (!response.ok || !body.data?.file) {
    throw new ApiError(body.message || "The music file could not be saved.", response.status || 500);
  }
  return body.data.file;
}

export async function uploadMaterial(file: File) {
  const payload = new FormData();
  payload.append("file", file);
  let response: Response;
  try {
    response = await fetch("/api/v1/video_materials", { method: "POST", body: payload });
  } catch {
    throw new ApiError("The Script-to-Reality Renderer Studio API is not reachable on this machine.", 0);
  }
  let body: { status?: number; message?: string; data?: { file?: string } } = {};
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The server returned an unreadable response.", response.status);
  }
  if (!response.ok || (typeof body.status === "number" && body.status >= 400) || !body.data?.file) {
    throw new ApiError(body.message || "The file could not be saved.", response.status || body.status || 500);
  }
  return body.data.file;
}

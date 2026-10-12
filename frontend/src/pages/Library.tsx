import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteTask, openTaskFolder, readLibrary, readTaskLogs, readTasks, stamp } from "../api";
import { useT } from "../i18n";
import type { LibraryItem, RenderTask } from "../types";

const filters = [
  { id: "all", label: "All" },
  { id: "processing", label: "Processing", state: 4 },
  { id: "complete", label: "Complete", state: 1 },
  { id: "failed", label: "Failed", state: -1 },
];

function statusLabel(state: number) {
  if (state === 1) return "Complete";
  if (state === -1) return "Failed";
  if (state === 4) return "Processing";
  return "Waiting";
}

export function Library() {
  const t = useT();
  const navigate = useNavigate();
  const [renders, setRenders] = useState<LibraryItem[]>([]);
  const [tasks, setTasks] = useState<RenderTask[]>([]);
  const [filter, setFilter] = useState("all");
  const [openId, setOpenId] = useState("");
  const [logs, setLogs] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  function load() {
    setLoading(true);
    setError("");
    Promise.all([readLibrary(), readTasks()])
      .then(([library, listed]) => {
        setRenders(library.renders);
        setTasks(listed.tasks || []);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenId("");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const visible = tasks.filter((task) => {
    const match = filters.find((item) => item.id === filter);
    return !match?.state || task.state === match.state;
  });

  function toggle(id: string) {
    setLogs("");
    setOpenId((current) => (current === id ? "" : id));
  }

  function restore(task: RenderTask) {
    const params = task.params || {};
    sessionStorage.setItem("frameforge-draft", JSON.stringify({
      ...params,
      video_subject: params.video_subject || task.video_subject || "",
      aspect: params.video_aspect || "9:16",
      transition: params.video_transition_mode || "none",
      clip_duration: params.video_clip_duration,
    }));
    navigate("/");
  }

  return (
    <>
      <h1 className="page-title">{t("Task Management", "Library")}</h1>
      <p className="lede">Click a project to open it. Everything else stays in the list.</p>
      <div className="actions" style={{ marginBottom: "1rem" }}>
        <button className="btn-quiet" type="button" onClick={load} disabled={loading}>{loading ? "Checking…" : t("Refresh", "Refresh")}</button>
      </div>
      {error ? <p className="notice notice-bad" role="alert">{error}</p> : null}

      <section className="section">
        <h2>{t("Tasks", "Tasks")}</h2>
        <div className="group">
          <div className="block">
            <div className="segments" role="group" aria-label="Task filter">
              {filters.map((item) => (
                <button key={item.id} type="button" aria-pressed={filter === item.id} onClick={() => setFilter(item.id)}>{t(item.label, item.label)}</button>
              ))}
            </div>
          </div>
          {!loading && visible.length === 0 ? <p className="meta" style={{ padding: "12px 16px" }}>No tasks in this view.</p> : null}
          {visible.map((task) => {
            const id = `task-${task.task_id}`;
            const open = openId === id;
            const video = task.videos?.find((item) => item.endsWith(".mp4")) ?? "";
            return (
              <div key={task.task_id}>
                <button className="pick" type="button" aria-expanded={open} onClick={() => toggle(id)}>
                  <span>
                    <p className="can-title">{task.video_subject || task.task_id}</p>
                    <p className="meta">{statusLabel(task.state)} · {task.progress}%</p>
                  </span>
                  <span className="pick-arrow" aria-hidden="true" />
                </button>
                {open ? (
                  <div className="drop" role="region" aria-label={task.video_subject || "Task"}>
                    {task.error ? <p className="notice notice-bad">{task.error}</p> : null}
                    {video ? <video className="player" controls src={video} /> : null}
                    <div className="actions">
                      <button className="btn" type="button" onClick={() => restore(task)}>{t("Load Task", "Load into Generate")}</button>
                      <button className="btn-quiet" type="button" onClick={() => {
                        readTaskLogs(task.task_id).then((data) => setLogs((data.lines || []).join("\n") || "No log for this task.")).catch((reason: Error) => setError(reason.message));
                      }}>{t("View Log", "Log")}</button>
                      <button className="btn-quiet" type="button" onClick={() => {
                        openTaskFolder(task.task_id).catch((reason: Error) => setError(reason.message));
                      }}>{t("Open Folder", "Open folder")}</button>
                      <button className="btn-danger" type="button" onClick={() => {
                        deleteTask(task.task_id).then(() => {
                          setOpenId("");
                          load();
                        }).catch((reason: Error) => setError(reason.message));
                      }}>{t("Delete", "Delete")}</button>
                    </div>
                    {logs ? <pre className="area" style={{ maxHeight: "180px", overflow: "auto" }}>{logs}</pre> : null}
                    {video ? <a className="btn-quiet" href={video} download>Download MP4</a> : null}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>

      <section className="section">
        <h2>{t("Finished Videos", "Finished videos")}</h2>
        <div className="group">
          {!loading && renders.length === 0 ? <p className="meta" style={{ padding: "12px 16px" }}>No finished videos yet.</p> : null}
          {renders.map((item) => {
            const id = `video-${item.id}`;
            const open = openId === id;
            return (
              <div key={item.id}>
                <button className="pick" type="button" aria-expanded={open} onClick={() => toggle(id)}>
                  <span>
                    <p className="can-title">{item.title}</p>
                    <p className="meta">{stamp(item.updated_at)}</p>
                  </span>
                  <span className="pick-arrow" aria-hidden="true" />
                </button>
                {open ? (
                  <div className="drop" role="region" aria-label={item.title}>
                    {item.script ? <p>{item.script}</p> : null}
                    <video className="player" controls src={item.video} />
                    <a className="btn" href={item.video} download>Download MP4</a>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}

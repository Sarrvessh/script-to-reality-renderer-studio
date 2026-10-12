import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { readSettings } from "./api";
import { LocaleProvider, useT } from "./i18n";
import { Dashboard } from "./pages/Dashboard";
import { Library } from "./pages/Library";
import { Settings } from "./pages/Settings";

function Shell() {
  const t = useT();
  return (
    <div className="shell">
      <aside className="cart">
        <p className="brand">Script-to-Reality<br /><span>Renderer Studio</span></p>
        <nav aria-label="Main">
          <NavLink to="/" className={({ isActive }) => `slot${isActive ? " slot-live" : ""}`} end>{t("Generate Video", "Generate")}</NavLink>
          <NavLink to="/library" className={({ isActive }) => `slot${isActive ? " slot-live" : ""}`}>{t("Task Management", "Library")}</NavLink>
          <NavLink to="/settings" className={({ isActive }) => `slot${isActive ? " slot-live" : ""}`}>{t("Settings", "Settings")}</NavLink>
        </nav>
      </aside>
      <main className="bench">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/library" element={<Library />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </main>
    </div>
  );
}

export function App() {
  const [language, setLanguage] = useState("en");

  useEffect(() => {
    readSettings()
      .then((data) => setLanguage(data.settings.ui_language || "en"))
      .catch(() => setLanguage("en"));
    const onChange = (event: Event) => setLanguage(String((event as CustomEvent).detail || "en"));
    window.addEventListener("frameforge-language", onChange);
    return () => window.removeEventListener("frameforge-language", onChange);
  }, []);

  return (
    <LocaleProvider code={language}>
      <Shell />
    </LocaleProvider>
  );
}

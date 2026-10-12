import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { readLocale } from "./api";

const LocaleContext = createContext<(key: string, fallback?: string) => string>((_key, fallback) => fallback || _key);

function readable(value: string) {
  return value
    .replace(/:\w+\[([^\]]+)\]/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}

export function LocaleProvider({ code, children }: { code: string; children: ReactNode }) {
  const [strings, setStrings] = useState<Record<string, string>>({});

  useEffect(() => {
    let stop = false;
    readLocale(code || "en")
      .then((data) => {
        if (!stop) setStrings(data.strings || {});
      })
      .catch(() => {
        if (!stop) setStrings({});
      });
    return () => {
      stop = true;
    };
  }, [code]);

  return (
    <LocaleContext.Provider value={(key, fallback) => readable(strings[key] || fallback || key)}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useT() {
  return useContext(LocaleContext);
}

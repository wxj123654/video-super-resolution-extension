import React from "react";
import ReactDOM from "react-dom/client";

import "@src/ui/styles/globals.css";

function OptionsPlaceholder() {
  return (
    <main className="popup-shell">
      <section className="ui-card">
        <div className="ui-card__header">
          <h1 className="ui-card__title">Full settings</h1>
          <p className="popup-subtitle">Full settings coming in Task 4.</p>
        </div>
        <div className="ui-card__content">
          <p className="inline-note">
            当前入口已接通，后续会在 Task 4 补齐完整设置、诊断和高级选项。
          </p>
        </div>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <OptionsPlaceholder />
  </React.StrictMode>,
);

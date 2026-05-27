import React from "react";
import ReactDOM from "react-dom/client";

import "@src/ui/styles/globals.css";

import { PopupRoot } from "./app";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <PopupRoot />
  </React.StrictMode>,
);

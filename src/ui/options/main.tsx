import React from "react";
import ReactDOM from "react-dom/client";

import "@src/ui/styles/globals.css";

import { OptionsRoot } from "./app";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <OptionsRoot />
  </React.StrictMode>,
);

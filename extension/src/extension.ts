/**
 * extension.ts
 * ============
 * Entry point for the UnstuckAI VS Code extension.
 *
 * Phase 1 (current): Data collection only.
 *   - Captures keystroke, error, and cursor events
 *   - Writes to sessions.jsonl every 5 seconds
 *   - Registers Cmd+Shift+C confusion label shortcut
 *
 * Phase 2 (after real data collected): Add scorer.ts, webview.ts, claude.ts
 *   - Runs ONNX model every 5 seconds
 *   - Activates animated character when confusion score > 0.65
 *   - Calls Claude API for adaptive visual explanation
 *
 * To run in development:
 *   1. Open this folder in VS Code
 *   2. Press F5 — opens Extension Development Host window
 *   3. Code normally in the new window
 *   4. Check data/sessions-YYYY-MM-DD.jsonl for captured events
 */

import * as vscode from "vscode";
import { SignalCollector } from "./signals";
import { SessionLogger }   from "./logger";
import { LabelManager }    from "./labels";

// Keep references alive for the extension lifetime
let collector: SignalCollector | undefined;
let logger:    SessionLogger   | undefined;
let labeler:   LabelManager    | undefined;

// ─────────────────────────────────────────────
// Activate — called once when VS Code starts
// ─────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  console.log("[UnstuckAI] Extension activating...");

  // 1. Start signal collection
  collector = new SignalCollector(context);

  // 2. Start JSONL logger (writes every 5 seconds)
  logger = new SessionLogger(collector, context);

  // 3. Register confusion label shortcut
  labeler = new LabelManager(collector, logger, context);

  const labelCommand = vscode.commands.registerCommand(
    "unstuckai.label",
    () => labeler!.handleLabel()
  );

  // 4. Status bar item showing collection is active
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 100
  );
  statusBar.text     = "$(record) UnstuckAI";
  statusBar.tooltip  = "UnstuckAI is collecting data. Press Ctrl+Shift+C when confused.";
  statusBar.command  = "unstuckai.label";
  statusBar.show();

  // 5. Show welcome message on first install
  const isFirstRun = context.globalState.get("unstuckai.firstRun", true);
  if (isFirstRun) {
    vscode.window.showInformationMessage(
      "UnstuckAI is now collecting your coding behaviour. " +
      "Press Ctrl+Shift+C (or Cmd+Shift+C on Mac) whenever you feel confused. " +
      "The more you label, the better the model gets.",
      "Got it"
    );
    context.globalState.update("unstuckai.firstRun", false);
  }

  // 6. Register stats command (shows how much data collected)
  const statsCommand = vscode.commands.registerCommand(
    "unstuckai.stats",
    () => {
      const stats = logger!.getStats();
      vscode.window.showInformationMessage(
        `UnstuckAI Stats — ` +
        `Records written: ${stats.records} | ` +
        `Confusion labels: ${labeler!.getCount()} | ` +
        `File: ${stats.file}`
      );
    }
  );

  // Push all disposables so VS Code cleans them up on deactivate
  context.subscriptions.push(
    labelCommand,
    statsCommand,
    statusBar,
    { dispose: () => collector?.dispose() },
  );

  console.log("[UnstuckAI] Extension active. Collecting data...");
}

// ─────────────────────────────────────────────
// Deactivate — called when VS Code closes
// ─────────────────────────────────────────────

export function deactivate() {
  // Flush final records before shutdown
  logger?.stop();
  console.log("[UnstuckAI] Extension deactivated. Data saved.");
}
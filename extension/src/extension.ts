/**
 * extension.ts
 * ============
 * Entry point. Wires SignalCollector → SessionLogger → LabelManager.
 * Key fix: collector.setLogger(logger) connects terminal error flow.
 */

import * as vscode from "vscode";
import { SignalCollector } from "./signals";
import { SessionLogger }   from "./logger";
import { LabelManager }    from "./labels";

let collector: SignalCollector | undefined;
let logger:    SessionLogger   | undefined;
let labeler:   LabelManager    | undefined;

export function activate(context: vscode.ExtensionContext) {
  console.log("[UnstuckAI] Activating...");

  // 1. Create signal collector
  collector = new SignalCollector(context);

  // 2. Create logger
  logger = new SessionLogger(collector, context);

  // 3. CRITICAL: connect logger to collector so terminal errors flow through
  collector.setLogger(logger);

  // 4. Create label manager
  labeler = new LabelManager(collector, logger, context);

  // 5. Register Ctrl+Shift+C command — silent, no popup
  const labelCommand = vscode.commands.registerCommand(
    "unstuckai.label",
    () => labeler!.handleLabel()
  );

  // 6. Register stats command
  const statsCommand = vscode.commands.registerCommand(
    "unstuckai.stats",
    () => {
      const stats = logger!.getStats();
      vscode.window.showInformationMessage(
        `UnstuckAI — Records: ${stats.records} | ` +
        `Labels: ${labeler!.getCount()} | ` +
        `Terminal errors: ${(logger as any).terminalErrorCount ?? 0} | ` +
        `File: ${stats.file}`
      );
    }
  );

  // 7. Status bar indicator
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 100
  );
  statusBar.text    = "$(record) UnstuckAI";
  statusBar.tooltip = "UnstuckAI collecting data. Ctrl+Shift+C = confused";
  statusBar.command = "unstuckai.stats";
  statusBar.show();

  // 8. First run message
  const isFirstRun = context.globalState.get("unstuckai.firstRun", true);
  if (isFirstRun) {
    vscode.window.showInformationMessage(
      "UnstuckAI is running. Press Ctrl+Shift+C when you feel confused. " +
      "Confusion type is detected automatically from your errors."
    );
    context.globalState.update("unstuckai.firstRun", false);
  }

  context.subscriptions.push(
    labelCommand,
    statsCommand,
    statusBar,
    { dispose: () => collector?.dispose() },
  );

  console.log("[UnstuckAI] Active. Terminal errors will auto-detect confusion type.");
}

export function deactivate() {
  logger?.stop();
  console.log("[UnstuckAI] Deactivated.");
}
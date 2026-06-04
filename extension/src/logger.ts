/**
 * logger.ts — FINAL VERSION
 * =========================
 * All 6 fixes applied:
 *   1. terminal_error_type  — persists across windows correctly
 *   2. confusion_type       — auto-filled from terminal error every window
 *   3. same_error_count     — resets after 3 clean windows
 *   4. wpm                  — correct calculation
 *   5. delete_ratio         — correct calculation
 *   6. total_keystrokes     — accumulates properly
 */

import * as vscode from "vscode";
import * as fs     from "fs";
import * as path   from "path";
import * as os     from "os";
import { SignalCollector, RawEvent } from "./signals";

// ─────────────────────────────────────────────
// CHANGE THIS IF YOU MOVE THE PROJECT
// ─────────────────────────────────────────────
const PROJECT_DATA_PATH =
  "C:\\Users\\SHRUTI\\Downloads\\diploma projects\\Instuck AI\\data";

const WRITE_INTERVAL_MS  = 5000;
const IDLE_THRESHOLD_SEC = 3.0;

// ─────────────────────────────────────────────
// Maps terminal error → confusion type
// Auto-fills confusion_type on every window
// ─────────────────────────────────────────────
const ERROR_TO_CONFUSION: Record<string, string> = {
  "SyntaxError":         "syntax_error",
  "IndentationError":    "indentation",
  "NameError":           "name_error",
  "TypeError":           "type_error",
  "ValueError":          "type_error",
  "IndexError":          "index_error",
  "KeyError":            "key_error",
  "AttributeError":      "class_self",
  "ImportError":         "import_error",
  "ModuleNotFoundError": "import_error",
  "RecursionError":      "recursion",
  "ZeroDivisionError":   "type_error",
  "FileNotFoundError":   "procedure",
  "OSError":             "procedure",
  "RuntimeError":        "logic",
  "StopIteration":       "loop_boundary",
  "OverflowError":       "type_error",
  "MemoryError":         "memory_leak",
  "AssertionError":      "logic",
  "NotImplementedError": "wrong_return",
  "TimeoutError":        "async_await",
  "PermissionError":     "procedure",
  "Traceback":           "logic",
  "ReferenceError":      "name_error",
  "TSCannotFindName":    "name_error",
  "TSTypeAssignment":    "type_error",
  "TSPropertyMissing":   "class_self",
  "TSWrongArgCount":     "wrong_return",
  "JSNotDefined":        "name_error",
  "JSNotAFunction":      "type_error",
  "JSCannotReadProp":    "reference_copy",
  "JSUncaught":          "logic",
};

export class SessionLogger {
  private timer:            NodeJS.Timeout | null = null;
  private dataDir:          string;
  private currentFile:      string = "";
  private writeCount        = 0;
  private loggerErrorCount  = 0;

  // ── Error tracking (persists across windows) ──
  private lastErrorType:   string | null = null;
  private sameErrorCount   = 0;
  private attemptCount     = 0;
  private noErrorWindows   = 0;
  private readonly RESET_AFTER_CLEAN_WINDOWS = 3;

  // ── Per-window flags (reset each tick) ────────
  private runCount              = 0;
  private hintClickedFlag       = 0;
  private cartoonActivated      = 0;
  private understoodAfterHelp   = 0;
  private confusionType: string = "unknown";  // set by Ctrl+Shift+C

  // ── Terminal error state (public so signals.ts can write) ──
  public terminalErrorType:  string | null = null;
  public terminalErrorCount  = 0;
  public terminalErrorLine   = 0;
  public lastTerminalErrorTs = 0;
  private readonly TERMINAL_DECAY_MS = 5 * 60 * 1000;

  constructor(
    private collector: SignalCollector,
    private context:   vscode.ExtensionContext,
  ) {
    this.dataDir = this.resolveDataDir();
    fs.mkdirSync(this.dataDir, { recursive: true });
    this.rotateFile();
    this.start();
    console.log("[UnstuckAI] Logger writing to:", this.dataDir);
    vscode.window.showInformationMessage(
      `UnstuckAI: Saving data to ${this.dataDir}`
    );
  }

  // ── Public API ───────────────────────────────

  start(): void {
    if (this.timer) { return; }
    this.timer = setInterval(() => this.tick(), WRITE_INTERVAL_MS);
  }

  stop(): void {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    this.tick();
    console.log(`[UnstuckAI] Stopped. Wrote ${this.writeCount} records.`);
  }

  getStats() {
    return {
      records: this.writeCount,
      errors:  this.loggerErrorCount,
      file:    this.currentFile,
      dataDir: this.dataDir,
    };
  }

  // Called by labels.ts — overrides auto-detection for this window
  setConfusionType(type: string): void {
    this.confusionType = type;
    console.log("[UnstuckAI] Confusion type manually set:", type);
  }

  // Called by labels.ts to read current terminal error
  getLastTerminalErrorType(): string | null {
    return this.terminalErrorType;
  }

  // Called by signals.ts when terminal output contains an error
  updateTerminalError(type: string, line: number): void {
    this.terminalErrorType   = type;
    this.terminalErrorLine   = line;
    this.terminalErrorCount++;
    this.lastTerminalErrorTs = Date.now();
    console.log(`[UnstuckAI] Terminal error: ${type} line ${line}`);
  }

  markCartoonActivated():      void { this.cartoonActivated    = 1; }
  markUnderstoodAfterHelp():   void { this.understoodAfterHelp = 1; }
  incrementRunCount():         void { this.runCount++; }

  // ── Core write loop ──────────────────────────

  private tick(): void {
    this.rotateFile();
    const events = this.collector.drain(500);

    // Decay terminal error after 5 minutes of silence
    if (this.terminalErrorType &&
        Date.now() - this.lastTerminalErrorTs > this.TERMINAL_DECAY_MS) {
      this.terminalErrorType  = null;
      this.terminalErrorCount = 0;
      this.terminalErrorLine  = 0;
    }

    if (events.length === 0) { return; }

    const record = this.aggregate(events);
    if (record) { this.append(record); }

    // Reset per-window flags after writing
    this.hintClickedFlag     = 0;
    this.cartoonActivated    = 0;
    this.understoodAfterHelp = 0;
    this.confusionType       = "unknown";
    this.runCount            = 0;
  }

  private aggregate(events: RawEvent[]): object | null {
    if (events.length === 0) { return null; }

    const now        = Date.now() / 1000;
    const last       = events[events.length - 1];
    const session_id = last.session_id;
    const language   = last.language;
    const windowSec  = WRITE_INTERVAL_MS / 1000;
    const windowMin  = windowSec / 60;

    // ── Typing signals ────────────────────────
    const totalKeys   = events.length;
    const deletes     = events.filter(e => e.is_delete === 1).length;
    const inserts     = totalKeys - deletes;
    const wpm         = Math.min(
      Math.round((inserts / 5 / windowMin) * 10) / 10, 300
    );
    const deleteRatio = totalKeys > 0
      ? Math.round((deletes / totalKeys) * 1000) / 1000 : 0;

    // ── Idle pauses ───────────────────────────
    const idles    = events.map(e => e.idle_sec).filter(i => i > 0 && i < 120);
    const idleMean = idles.length > 0
      ? Math.round((idles.reduce((a, b) => a + b, 0) / idles.length) * 1000) / 1000 : 0;
    const idleMax  = idles.length > 0
      ? Math.round(Math.max(...idles) * 100) / 100 : 0;
    const idleCount = idles.filter(i => i > IDLE_THRESHOLD_SEC).length;

    // ── Problems panel errors ─────────────────
    const problemsErrorCount = Math.max(
      ...events.map(e => (e as any).problems_error_count ?? e.error_count ?? 0), 0
    );
    const problemsErrorTypes = events
      .map(e => (e as any).problems_error_type ?? "").filter(Boolean) as string[];
    const problemsErrorType  = problemsErrorTypes.length > 0
      ? this.mostCommon(problemsErrorTypes) : null;

    // ── Terminal errors (persistent state) ────
    const currentTerminalType  = this.terminalErrorType;
    const currentTerminalCount = this.terminalErrorCount;
    const currentTerminalLine  = this.terminalErrorLine;

    // ── Combined error signal ─────────────────
    const combinedErrorType = currentTerminalType ?? problemsErrorType;
    const totalErrorCount   = problemsErrorCount + (currentTerminalCount > 0 ? 1 : 0);

    // ── same_error_count with reset ───────────
    if (combinedErrorType) {
      this.noErrorWindows = 0;
      if (combinedErrorType === this.lastErrorType) {
        this.sameErrorCount++;
        this.attemptCount++;
      } else {
        this.sameErrorCount = 1;
        this.attemptCount++;
        this.lastErrorType  = combinedErrorType;
      }
    } else {
      this.noErrorWindows++;
      if (this.noErrorWindows >= this.RESET_AFTER_CLEAN_WINDOWS) {
        this.sameErrorCount = 0;
        this.attemptCount   = 0;
        this.lastErrorType  = null;
        this.noErrorWindows = 0;
      }
    }

    const errorRepeat = this.sameErrorCount >= 3 ? 1 : 0;

    // ── Cursor ────────────────────────────────
    const lines       = events.map(e => e.line_number);
    const lineChanges = lines.filter((l, i) => i > 0 && l !== lines[i - 1]).length;
    const lineFreq    = parseInt(this.mostCommon(lines.map(String)), 10);
    const timeOnLine  = lines.filter(l => l === lineFreq).length /
                        Math.max(totalKeys, 1) * windowSec;

    // ── Confusion label ───────────────────────
    const confused = events.some(e => e.confused === 1) ? 1 : 0;

    // ── AUTO confusion_type ───────────────────
    // Priority 1: manually set via Ctrl+Shift+C (setConfusionType)
    // Priority 2: auto-mapped from terminal error type
    // Priority 3: auto-mapped from problems panel error
    // Priority 4: "unknown"
    let resolvedConfusionType = "unknown";
    if (this.confusionType !== "unknown") {
      // User manually pressed Ctrl+Shift+C and type was detected
      resolvedConfusionType = this.confusionType;
    } else if (currentTerminalType) {
      // Auto-fill from terminal error every window
      resolvedConfusionType = ERROR_TO_CONFUSION[currentTerminalType] ?? "unknown";
    } else if (problemsErrorType) {
      // Auto-fill from problems panel
      resolvedConfusionType = ERROR_TO_CONFUSION[problemsErrorType] ?? "unknown";
    }

    return {
      // Metadata
      session_id,
      timestamp:            now,
      window_start:         now - windowSec,
      confused,
      language,

      // Typing signals
      wpm,
      delete_ratio:         deleteRatio,
      max_idle_sec:         idleMax,
      idle_mean:            idleMean,
      idle_count:           idleCount,
      total_keystrokes:     totalKeys,
      insert_count:         inserts,
      delete_count:         deletes,

      // Combined error
      error_count:          totalErrorCount,
      error_type:           combinedErrorType,
      error_repeat_count:   errorRepeat,

      // Problems panel
      problems_error_count: problemsErrorCount,
      problems_error_type:  problemsErrorType,

      // Terminal error
      terminal_error_count: currentTerminalCount,
      terminal_error_type:  currentTerminalType,
      terminal_error_line:  currentTerminalLine,

      // Cursor
      time_on_line_sec:     Math.round(timeOnLine * 10) / 10,
      cursor_moves:         lineChanges,

      // Raw fields for features.py
      is_delete:            deletes > inserts ? 1 : 0,
      idle_sec:             idleMean,
      line_number:          lines[lines.length - 1] ?? 0,

      // Error tracking
      attempt_count:        this.attemptCount,
      same_error_count:     this.sameErrorCount,

      // Context
      run_count:            this.runCount,
      hint_clicked:         this.hintClickedFlag,
      question_asked:       0,
      cartoon_activated:    this.cartoonActivated,
      understood_after_help: this.understoodAfterHelp,

      // AUTO-FILLED from terminal/problems error every window
      confusion_type:       resolvedConfusionType,
    };
  }

  // ── File management ──────────────────────────

  private resolveDataDir(): string {
    const parentDir = path.dirname(PROJECT_DATA_PATH);
    if (fs.existsSync(parentDir)) { return PROJECT_DATA_PATH; }
    const ws = vscode.workspace.workspaceFolders;
    if (ws?.length) { return path.join(ws[0].uri.fsPath, "data"); }
    return path.join(os.homedir(), ".unstuckai", "data");
  }

  private rotateFile(): void {
    const today    = new Date().toISOString().slice(0, 10);
    const filePath = path.join(this.dataDir, `sessions-${today}.jsonl`);
    if (filePath !== this.currentFile) {
      this.currentFile = filePath;
      console.log("[UnstuckAI] Logging to:", filePath);
    }
  }

  private append(record: object): void {
    const line = JSON.stringify(record) + "\n";
    try {
      fs.appendFileSync(this.currentFile, line, "utf8");
      this.writeCount++;
    } catch (err) {
      this.loggerErrorCount++;
      console.error("[UnstuckAI] Write failed:", err);
      try {
        const fb = path.join(os.homedir(), ".unstuckai", "data");
        fs.mkdirSync(fb, { recursive: true });
        fs.appendFileSync(
          path.join(fb, `sessions-${new Date().toISOString().slice(0, 10)}.jsonl`),
          line, "utf8"
        );
      } catch {}
    }
  }

  private mostCommon<T>(arr: T[]): T {
    const freq = new Map<T, number>();
    arr.forEach(v => freq.set(v, (freq.get(v) ?? 0) + 1));
    let best = arr[0], bestCount = 0;
    freq.forEach((count, val) => {
      if (count > bestCount) { best = val; bestCount = count; }
    });
    return best;
  }
}
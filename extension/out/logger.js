"use strict";
/**
 * logger.ts — FIXED VERSION
 * =========================
 * Fixes applied:
 *   1. terminal_error_type — carried across windows correctly
 *   2. confusion_type      — updated by labels.ts quick pick
 *   3. same_error_count    — resets when error disappears
 *   4. wpm                 — fixed calculation (counts all keystrokes)
 *   5. delete_ratio        — fixed (counts actual deletions)
 *   6. total_keystrokes    — fixed (accumulates properly per window)
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionLogger = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
// ─────────────────────────────────────────────
// CHANGE THIS IF YOU MOVE THE PROJECT
// ─────────────────────────────────────────────
const PROJECT_DATA_PATH = "C:\\Users\\SHRUTI\\Downloads\\diploma projects\\Instuck AI\\data";
const WRITE_INTERVAL_MS = 5000; // write every 5 seconds
const IDLE_THRESHOLD_SEC = 3.0; // pauses > 3s are flagged
class SessionLogger {
    constructor(collector, context) {
        this.collector = collector;
        this.context = context;
        this.timer = null;
        this.currentFile = "";
        this.writeCount = 0;
        this.loggerErrorCount = 0;
        // ── Error tracking (persists across windows) ──
        this.lastErrorType = null;
        this.sameErrorCount = 0;
        this.attemptCount = 0;
        this.noErrorWindows = 0; // how many consecutive windows had no error
        this.RESET_AFTER_CLEAN_WINDOWS = 3; // reset counts after 3 clean windows
        // ── Per-window flags (reset each tick) ────────
        this.runCount = 0;
        this.hintClickedFlag = 0;
        this.cartoonActivated = 0;
        this.understoodAfterHelp = 0;
        this.confusionType = "unknown";
        // ── Terminal error state (persists, decays) ───
        this.terminalErrorType = null;
        this.terminalErrorCount = 0;
        this.terminalErrorLine = 0;
        this.lastTerminalErrorTs = 0;
        this.TERMINAL_DECAY_MS = 5 * 60 * 1000; // 5 minutes
        this.dataDir = this.resolveDataDir();
        fs.mkdirSync(this.dataDir, { recursive: true });
        this.rotateFile();
        this.start();
        console.log("[UnstuckAI] Logger writing to:", this.dataDir);
        vscode.window.showInformationMessage(`UnstuckAI: Saving data to ${this.dataDir}`);
    }
    // ── Public API ───────────────────────────────
    start() {
        if (this.timer) {
            return;
        }
        this.timer = setInterval(() => this.tick(), WRITE_INTERVAL_MS);
    }
    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        this.tick();
        console.log(`[UnstuckAI] Stopped. Wrote ${this.writeCount} records.`);
    }
    getStats() {
        return {
            records: this.writeCount,
            errors: this.loggerErrorCount,
            file: this.currentFile,
            dataDir: this.dataDir,
        };
    }
    // Called by labels.ts after user selects confusion type
    setConfusionType(type) {
        this.confusionType = type;
        console.log("[UnstuckAI] Confusion type set to:", type);
    }
    // Called when UnstuckAI character appears
    markCartoonActivated() { this.cartoonActivated = 1; }
    // Called when confusion score drops after explanation
    markUnderstoodAfterHelp() { this.understoodAfterHelp = 1; }
    // Called when user runs code
    incrementRunCount() { this.runCount++; }
    // Called by labels.ts to get current terminal error type
    getLastTerminalErrorType() {
        return this.terminalErrorType;
    }
    // Called by signals.ts when terminal error is detected
    updateTerminalError(type, line) {
        this.terminalErrorType = type;
        this.terminalErrorLine = line;
        this.terminalErrorCount++;
        this.lastTerminalErrorTs = Date.now();
        console.log(`[UnstuckAI] Terminal error: ${type} at line ${line}`);
    }
    // ── Core write loop ──────────────────────────
    tick() {
        this.rotateFile();
        const events = this.collector.drain(500);
        // Decay terminal error after 5 minutes of no new errors
        if (this.terminalErrorType &&
            Date.now() - this.lastTerminalErrorTs > this.TERMINAL_DECAY_MS) {
            this.terminalErrorType = null;
            this.terminalErrorCount = 0;
            this.terminalErrorLine = 0;
        }
        if (events.length === 0) {
            return;
        }
        const record = this.aggregate(events);
        if (record) {
            this.append(record);
        }
        // Reset per-window flags after writing
        this.hintClickedFlag = 0;
        this.cartoonActivated = 0;
        this.understoodAfterHelp = 0;
        this.confusionType = "unknown";
        this.runCount = 0;
    }
    aggregate(events) {
        if (events.length === 0) {
            return null;
        }
        const now = Date.now() / 1000;
        const last = events[events.length - 1];
        const session_id = last.session_id;
        const language = last.language;
        // ────────────────────────────────────────────
        // FIX 4 + 5 + 6: WPM, delete ratio, keystrokes
        // Count ALL events in the window, not just last
        // ────────────────────────────────────────────
        const totalKeys = events.length;
        const deletes = events.filter(e => e.is_delete === 1).length;
        const inserts = totalKeys - deletes;
        // WPM: inserts / 5 chars per word / time in minutes
        const windowSec = WRITE_INTERVAL_MS / 1000;
        const windowMin = windowSec / 60;
        const wpm = Math.min(Math.round((inserts / 5 / windowMin) * 10) / 10, 300);
        // Delete ratio: deletions / total keystrokes
        const deleteRatio = totalKeys > 0
            ? Math.round((deletes / totalKeys) * 1000) / 1000
            : 0;
        // ── Idle pauses ──────────────────────────────
        const idles = events.map(e => e.idle_sec).filter(i => i > 0 && i < 120);
        const idleMean = idles.length > 0
            ? Math.round((idles.reduce((a, b) => a + b, 0) / idles.length) * 1000) / 1000
            : 0;
        const idleMax = idles.length > 0
            ? Math.round(Math.max(...idles) * 100) / 100
            : 0;
        const idleCount = idles.filter(i => i > IDLE_THRESHOLD_SEC).length;
        // ── Problems panel errors ─────────────────────
        const problemsErrorCounts = events.map(e => e.problems_error_count ?? e.error_count ?? 0);
        const problemsErrorCount = Math.max(...problemsErrorCounts, 0);
        const problemsErrorTypes = events
            .map(e => e.problems_error_type ?? "")
            .filter(Boolean);
        const problemsErrorType = problemsErrorTypes.length > 0
            ? this.mostCommon(problemsErrorTypes) : null;
        // ── Terminal errors (from persistent state) ───
        // FIX 1: Use persistent terminal state, not per-event
        const currentTerminalType = this.terminalErrorType;
        const currentTerminalCount = this.terminalErrorCount;
        const currentTerminalLine = this.terminalErrorLine;
        // ── Combined error signal ─────────────────────
        const combinedErrorType = currentTerminalType ?? problemsErrorType;
        const totalErrorCount = problemsErrorCount + (currentTerminalCount > 0 ? 1 : 0);
        // ── FIX 3: same_error_count reset logic ───────
        if (combinedErrorType) {
            this.noErrorWindows = 0;
            if (combinedErrorType === this.lastErrorType) {
                this.sameErrorCount++;
                this.attemptCount++;
            }
            else {
                // New error type appeared
                this.sameErrorCount = 1;
                this.attemptCount++;
                this.lastErrorType = combinedErrorType;
            }
        }
        else {
            // No error in this window
            this.noErrorWindows++;
            if (this.noErrorWindows >= this.RESET_AFTER_CLEAN_WINDOWS) {
                // Reset after 3 consecutive clean windows (15 seconds)
                this.sameErrorCount = 0;
                this.attemptCount = 0;
                this.lastErrorType = null;
                this.noErrorWindows = 0;
            }
        }
        // Error repeat flag
        const errorRepeat = this.sameErrorCount >= 3 ? 1 : 0;
        // ── Cursor ────────────────────────────────────
        const lines = events.map(e => e.line_number);
        const lineChanges = lines.filter((l, i) => i > 0 && l !== lines[i - 1]).length;
        const lineFreqStr = this.mostCommon(lines.map(String));
        const lineFreq = parseInt(lineFreqStr, 10);
        const timeOnLine = lines.filter(l => l === lineFreq).length /
            Math.max(totalKeys, 1) * windowSec;
        // ── Confusion label ───────────────────────────
        const confused = events.some(e => e.confused === 1) ? 1 : 0;
        // ── FIX 2: confusion_type from setConfusionType ─
        // confusionType is set by labels.ts via setConfusionType()
        // It defaults to "unknown" and resets each window after writing
        return {
            // Metadata
            session_id,
            timestamp: now,
            window_start: now - windowSec,
            confused,
            language,
            // FIX 4+5+6: corrected typing signals
            wpm,
            delete_ratio: deleteRatio,
            max_idle_sec: idleMax,
            idle_mean: idleMean,
            idle_count: idleCount,
            total_keystrokes: totalKeys,
            insert_count: inserts,
            delete_count: deletes,
            // Combined error (backward compatible)
            error_count: totalErrorCount,
            error_type: combinedErrorType,
            error_repeat_count: errorRepeat,
            // Problems panel
            problems_error_count: problemsErrorCount,
            problems_error_type: problemsErrorType,
            // FIX 1: Terminal errors from persistent state
            terminal_error_count: currentTerminalCount,
            terminal_error_type: currentTerminalType,
            terminal_error_line: currentTerminalLine,
            // Cursor
            time_on_line_sec: Math.round(timeOnLine * 10) / 10,
            cursor_moves: lineChanges,
            // Raw fields for features.py
            is_delete: deletes > inserts ? 1 : 0,
            idle_sec: idleMean,
            line_number: lines[lines.length - 1] ?? 0,
            // FIX 3: corrected error tracking
            attempt_count: this.attemptCount,
            same_error_count: this.sameErrorCount,
            // Per-window context
            run_count: this.runCount,
            hint_clicked: this.hintClickedFlag,
            question_asked: 0,
            cartoon_activated: this.cartoonActivated,
            understood_after_help: this.understoodAfterHelp,
            // FIX 2: confusion_type from quick pick
            confusion_type: this.confusionType,
        };
    }
    // ── File management ──────────────────────────
    resolveDataDir() {
        const parentDir = path.dirname(PROJECT_DATA_PATH);
        if (fs.existsSync(parentDir)) {
            console.log("[UnstuckAI] Using project path:", PROJECT_DATA_PATH);
            return PROJECT_DATA_PATH;
        }
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders?.length) {
            return path.join(workspaceFolders[0].uri.fsPath, "data");
        }
        return path.join(os.homedir(), ".unstuckai", "data");
    }
    rotateFile() {
        const today = new Date().toISOString().slice(0, 10);
        const filePath = path.join(this.dataDir, `sessions-${today}.jsonl`);
        if (filePath !== this.currentFile) {
            this.currentFile = filePath;
            console.log("[UnstuckAI] Logging to:", filePath);
        }
    }
    append(record) {
        const line = JSON.stringify(record) + "\n";
        try {
            fs.appendFileSync(this.currentFile, line, "utf8");
            this.writeCount++;
        }
        catch (err) {
            this.loggerErrorCount++;
            console.error("[UnstuckAI] Write failed:", err);
            // Fallback to home dir
            try {
                const fb = path.join(os.homedir(), ".unstuckai", "data");
                fs.mkdirSync(fb, { recursive: true });
                fs.appendFileSync(path.join(fb, `sessions-${new Date().toISOString().slice(0, 10)}.jsonl`), line, "utf8");
            }
            catch { }
        }
    }
    mostCommon(arr) {
        const freq = new Map();
        arr.forEach(v => freq.set(v, (freq.get(v) ?? 0) + 1));
        let best = arr[0], bestCount = 0;
        freq.forEach((count, val) => {
            if (count > bestCount) {
                best = val;
                bestCount = count;
            }
        });
        return best;
    }
}
exports.SessionLogger = SessionLogger;
//# sourceMappingURL=logger.js.map
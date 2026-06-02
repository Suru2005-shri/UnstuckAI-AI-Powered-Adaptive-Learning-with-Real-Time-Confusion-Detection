"use strict";
/**
 * labels.ts
 * =========
 * Silent confusion labeller — NO popup menu.
 * When user presses Ctrl+Shift+C:
 *   1. confused = 1 is recorded immediately
 *   2. confusion_type is auto-detected from terminal error
 *   3. Status bar flashes for 3 seconds — that is it
 *
 * confusion_type is mapped automatically from terminal_error_type:
 *   ValueError       → type_error
 *   NameError        → name_error
 *   SyntaxError      → syntax_error
 *   IndentationError → indentation
 *   IndexError       → index_error
 *   KeyError         → key_error
 *   TypeError        → type_error
 *   AttributeError   → class_self
 *   ImportError      → import_error
 *   RecursionError   → recursion
 *   ... and so on
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
exports.LabelManager = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
// ─────────────────────────────────────────────
// CHANGE THIS IF YOU MOVE THE PROJECT
// ─────────────────────────────────────────────
const PROJECT_DATA_PATH = "C:\\Users\\SHRUTI\\Downloads\\diploma projects\\Instuck AI\\data";
// ─────────────────────────────────────────────
// Auto-map terminal error type → confusion type
// ─────────────────────────────────────────────
const ERROR_TO_CONFUSION = {
    "SyntaxError": "syntax_error",
    "IndentationError": "indentation",
    "NameError": "name_error",
    "TypeError": "type_error",
    "ValueError": "type_error",
    "IndexError": "index_error",
    "KeyError": "key_error",
    "AttributeError": "class_self",
    "ImportError": "import_error",
    "ModuleNotFoundError": "import_error",
    "RecursionError": "recursion",
    "ZeroDivisionError": "type_error",
    "FileNotFoundError": "procedure",
    "OSError": "procedure",
    "RuntimeError": "logic",
    "StopIteration": "loop_boundary",
    "OverflowError": "type_error",
    "MemoryError": "memory_leak",
    "AssertionError": "logic",
    "NotImplementedError": "wrong_return",
    "TimeoutError": "async_await",
    "PermissionError": "procedure",
    "Traceback": "logic",
    "TSCannotFindName": "name_error",
    "TSTypeAssignment": "type_error",
    "TSPropertyMissing": "class_self",
    "TSWrongArgCount": "wrong_return",
    "ReferenceError": "name_error",
    "JSNotDefined": "name_error",
    "JSNotAFunction": "type_error",
    "JSCannotReadProp": "reference_copy",
    "JSUncaught": "logic",
};
// ─────────────────────────────────────────────
// Problems panel error → confusion type
// ─────────────────────────────────────────────
const PROBLEMS_TO_CONFUSION = {
    "SyntaxError": "syntax_error",
    "IndentationError": "indentation",
    "NameError": "name_error",
    "TypeError": "type_error",
    "TSCannotFindName": "name_error",
    "TSTypeAssignment": "type_error",
    "TSPropertyMissing": "class_self",
};
class LabelManager {
    constructor(collector, logger, context) {
        this.collector = collector;
        this.logger = logger;
        this.context = context;
        this.labelCount = 0;
        this.lastLabelTs = 0;
        this.MIN_GAP_MS = 5000; // 5 seconds minimum between labels
    }
    // ── Main handler — called on Ctrl+Shift+C ──
    handleLabel() {
        const now = Date.now();
        // Debounce
        if (now - this.lastLabelTs < this.MIN_GAP_MS) {
            vscode.window.setStatusBarMessage("$(alert) UnstuckAI: Already recorded recently", 2000);
            return;
        }
        this.lastLabelTs = now;
        this.labelCount++;
        // Step 1 — mark confused in signal buffer immediately
        this.collector.markConfused();
        // Step 2 — auto-detect confusion type from current errors
        const confusionType = this.detectConfusionType();
        // Step 3 — tell logger the type (written in next 5s window)
        this.logger.setConfusionType(confusionType);
        // Step 4 — write a dedicated label record to JSONL
        this.writeLabelRecord(now, confusionType);
        // Step 5 — silent status bar flash only, no popup
        this.showSilentFeedback(confusionType);
        console.log(`[UnstuckAI] Label #${this.labelCount} — auto type: ${confusionType}`);
    }
    getCount() {
        return this.labelCount;
    }
    // ── Auto-detect confusion type ─────────────
    detectConfusionType() {
        const editor = vscode.window.activeTextEditor;
        // Priority 1: Terminal error type from logger
        const terminalType = this.logger.getLastTerminalErrorType();
        if (terminalType) {
            const mapped = ERROR_TO_CONFUSION[terminalType];
            if (mapped) {
                console.log(`[UnstuckAI] Type from terminal: ${terminalType} → ${mapped}`);
                return mapped;
            }
        }
        // Priority 2: Problems panel diagnostics
        if (editor) {
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            const errors = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Error);
            if (errors.length > 0) {
                const msg = errors[0].message;
                const errorType = this.extractErrorType(msg);
                const mapped = PROBLEMS_TO_CONFUSION[errorType];
                if (mapped) {
                    console.log(`[UnstuckAI] Type from Problems: ${errorType} → ${mapped}`);
                    return mapped;
                }
            }
            // Priority 3: Infer from file language + no error = procedural confusion
            const lang = editor.document.languageId;
            if (["python", "javascript", "typescript"].includes(lang)) {
                // No error but confused = likely do not know what to type
                return "blank_line";
            }
        }
        // Priority 4: Fallback
        return "unknown";
    }
    extractErrorType(message) {
        const m = message.match(/^([A-Za-z][A-Za-z0-9_]+(Error|Exception)):/);
        if (m) {
            return m[1];
        }
        if (/Cannot find name/.test(message)) {
            return "TSCannotFindName";
        }
        if (/is not assignable/.test(message)) {
            return "TSTypeAssignment";
        }
        if (/Property .* does not exist/.test(message)) {
            return "TSPropertyMissing";
        }
        return "unknown";
    }
    // ── Write label record ──────────────────────
    writeLabelRecord(timestampMs, confusionType) {
        const record = {
            type: "confusion_label",
            timestamp: timestampMs / 1000,
            label_index: this.labelCount,
            confusion_type: confusionType,
            auto_detected: true,
        };
        const dataDir = this.resolveDataDir();
        const today = new Date().toISOString().slice(0, 10);
        const filePath = path.join(dataDir, `sessions-${today}.jsonl`);
        try {
            fs.mkdirSync(dataDir, { recursive: true });
            fs.appendFileSync(filePath, JSON.stringify(record) + "\n", "utf8");
        }
        catch (err) {
            console.error("[UnstuckAI] Label write failed:", err);
        }
    }
    // ── Silent feedback — status bar only ───────
    showSilentFeedback(confusionType) {
        const label = confusionType.replace(/_/g, " ");
        vscode.window.setStatusBarMessage(`$(alert) UnstuckAI: #${this.labelCount} logged [${label}]`, 3000);
        // Milestone notifications only — not every press
        const milestones = {
            10: "10 labels collected. Keep going.",
            50: "50 labels. Model is learning your patterns.",
            100: "100 labels. Excellent.",
            250: "250 labels. Halfway to a strong model.",
            500: "500 labels. Run: python model/train.py to retrain.",
        };
        if (milestones[this.labelCount]) {
            vscode.window.showInformationMessage(`UnstuckAI: ${milestones[this.labelCount]}`);
        }
    }
    resolveDataDir() {
        const parentDir = path.dirname(PROJECT_DATA_PATH);
        if (fs.existsSync(parentDir)) {
            return PROJECT_DATA_PATH;
        }
        const ws = vscode.workspace.workspaceFolders;
        if (ws?.length) {
            return path.join(ws[0].uri.fsPath, "data");
        }
        return path.join(os.homedir(), ".unstuckai", "data");
    }
}
exports.LabelManager = LabelManager;
//# sourceMappingURL=labels.js.map
/**
 * signals.ts
 * ==========
 * Captures editor events AND terminal output.
 * When a terminal error is detected it calls logger.updateTerminalError()
 * directly so labels.ts can read it immediately when Ctrl+Shift+C is pressed.
 */

import * as vscode from "vscode";

export interface RawEvent {
  timestamp:             number;
  is_delete:             number;
  chars_added:           number;
  idle_sec:              number;
  problems_error_count:  number;
  problems_error_type:   string;
  terminal_error_count:  number;
  terminal_error_type:   string;
  terminal_error_line:   number;
  error_count:           number;
  error_type:            string;
  line_number:           number;
  language:              string;
  confused:              number;
  session_id:            string;
}

// ─────────────────────────────────────────────
// All Python + JS error patterns to detect
// ─────────────────────────────────────────────
const ERROR_PATTERNS: { regex: RegExp; type: string }[] = [
  { regex: /SyntaxError:/,          type: "SyntaxError"          },
  { regex: /IndentationError:/,     type: "IndentationError"     },
  { regex: /NameError:/,            type: "NameError"            },
  { regex: /name '.*' is not defined/, type: "NameError"         },
  { regex: /TypeError:/,            type: "TypeError"            },
  { regex: /ValueError:/,           type: "ValueError"           },
  { regex: /IndexError:/,           type: "IndexError"           },
  { regex: /KeyError:/,             type: "KeyError"             },
  { regex: /AttributeError:/,       type: "AttributeError"       },
  { regex: /ImportError:/,          type: "ImportError"          },
  { regex: /ModuleNotFoundError:/,  type: "ModuleNotFoundError"  },
  { regex: /RecursionError:/,       type: "RecursionError"       },
  { regex: /ZeroDivisionError:/,    type: "ZeroDivisionError"    },
  { regex: /FileNotFoundError:/,    type: "FileNotFoundError"    },
  { regex: /OSError:/,              type: "OSError"              },
  { regex: /RuntimeError:/,         type: "RuntimeError"         },
  { regex: /StopIteration:/,        type: "StopIteration"        },
  { regex: /OverflowError:/,        type: "OverflowError"        },
  { regex: /MemoryError:/,          type: "MemoryError"          },
  { regex: /AssertionError:/,       type: "AssertionError"       },
  { regex: /NotImplementedError:/,  type: "NotImplementedError"  },
  { regex: /TimeoutError:/,         type: "TimeoutError"         },
  { regex: /PermissionError:/,      type: "PermissionError"      },
  { regex: /expected ':'/,          type: "SyntaxError"          },
  { regex: /unexpected indent/,     type: "IndentationError"     },
  { regex: /invalid syntax/,        type: "SyntaxError"          },
  { regex: /Traceback \(most recent call last\)/, type: "Traceback" },
  { regex: /ReferenceError:/,       type: "ReferenceError"       },
  { regex: /Cannot find name/,      type: "TSCannotFindName"     },
  { regex: /is not defined/,        type: "JSNotDefined"         },
  { regex: /is not a function/,     type: "JSNotAFunction"       },
  { regex: /Cannot read prop/,      type: "JSCannotReadProp"     },
];

const LINE_NUMBER_PATTERN = /line (\d+)/i;

// ─────────────────────────────────────────────
// SignalCollector
// ─────────────────────────────────────────────
export class SignalCollector {
  private buffer:            RawEvent[] = [];
  private maxBuffer          = 500;
  private lastEventTime      = Date.now();
  private sessionId:         string;
  private confusedFlag       = false;
  private disposables:       vscode.Disposable[] = [];
  private terminalBuffer     = "";

  // Logger reference — set after construction to avoid circular dep
  private loggerRef: any = null;

  constructor(private context: vscode.ExtensionContext) {
    this.sessionId = this.makeSessionId();
    this.attachEditorListeners();
    this.attachTerminalListeners();
    console.log("[UnstuckAI] SignalCollector started — session:", this.sessionId);
  }

  // ── Called from extension.ts after logger is created ──
  setLogger(logger: any): void {
    this.loggerRef = logger;
    console.log("[UnstuckAI] Logger connected to SignalCollector");
  }

  // ── Public API ──────────────────────────────

  drain(maxEvents = 500): RawEvent[] {
    const snapshot    = [...this.buffer.slice(-maxEvents)];
    this.confusedFlag = false;
    return snapshot;
  }

  markConfused(): void {
    this.confusedFlag = true;
    const n = Math.min(10, this.buffer.length);
    for (let i = this.buffer.length - n; i < this.buffer.length; i++) {
      this.buffer[i].confused = 1;
    }
    vscode.window.setStatusBarMessage("$(alert) UnstuckAI: Confusion logged", 3000);
  }

  newSession(): void {
    this.sessionId = this.makeSessionId();
    this.buffer    = [];
  }

  dispose(): void {
    this.disposables.forEach(d => d.dispose());
  }

  // ── Editor listeners ────────────────────────

  private attachEditorListeners(): void {
    const onTextChange = vscode.workspace.onDidChangeTextDocument(event => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || event.document !== editor.document) { return; }
      if (event.contentChanges.length === 0) { return; }

      const now     = Date.now();
      const idleSec = (now - this.lastEventTime) / 1000;
      this.lastEventTime = now;

      for (const change of event.contentChanges) {
        const isDelete   = change.rangeLength > 0 && change.text.length === 0 ? 1 : 0;
        const charsAdded = change.text.length - change.rangeLength;

        const { errorCount: pCount, errorType: pType } =
          this.getProblemsErrors(event.document.uri);

        // Get terminal error from logger if available
        const tType  = this.loggerRef?.terminalErrorType  ?? "";
        const tCount = this.loggerRef?.terminalErrorCount ?? 0;
        const tLine  = this.loggerRef?.terminalErrorLine  ?? 0;

        const totalCount    = pCount + (tCount > 0 ? 1 : 0);
        const combinedType  = tType || pType;

        this.push({
          timestamp:            now / 1000,
          is_delete:            isDelete,
          chars_added:          charsAdded,
          idle_sec:             idleSec,
          problems_error_count: pCount,
          problems_error_type:  pType,
          terminal_error_count: tCount,
          terminal_error_type:  tType,
          terminal_error_line:  tLine,
          error_count:          totalCount,
          error_type:           combinedType,
          line_number:          editor.selection.active.line + 1,
          language:             event.document.languageId,
          confused:             0,
          session_id:           this.sessionId,
        });
      }
    });

    const onCursorMove = vscode.window.onDidChangeTextEditorSelection(() => {});
    this.disposables.push(onTextChange, onCursorMove);
  }

  // ── Terminal listeners ──────────────────────

  private attachTerminalListeners(): void {
    // Listen to all existing terminals
    vscode.window.terminals.forEach(t => this.listenToTerminal(t));

    // Listen to new terminals
    const onOpen = vscode.window.onDidOpenTerminal(t => {
      this.listenToTerminal(t);
    });

    this.disposables.push(onOpen);
  }

  private listenToTerminal(terminal: vscode.Terminal): void {
    // Use onDidWriteTerminalData if available (requires enabledApiProposals)
    // If not available, fall back to polling Problems panel only
    try {
      if (!("onDidWriteTerminalData" in vscode.window)) {
        // API not available — use polling approach instead
        this.startTerminalPolling();
        return;
      }

      const listener = (vscode.window as any).onDidWriteTerminalData(
        (event: { terminal: vscode.Terminal; data: string }) => {
          if (event.terminal !== terminal) { return; }
          this.processTerminalOutput(event.data);
        }
      );
      this.disposables.push(listener);
      console.log("[UnstuckAI] Terminal listener attached via onDidWriteTerminalData");
    } catch (err) {
      console.warn("[UnstuckAI] Terminal API unavailable, using polling:", err);
      this.startTerminalPolling();
    }
  }

  // ── Polling fallback — checks Problems panel every 2s ──

  private pollingStarted = false;
  private startTerminalPolling(): void {
    if (this.pollingStarted) { return; }
    this.pollingStarted = true;

    const poll = setInterval(() => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) { return; }

      const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
      const errors      = diagnostics.filter(
        d => d.severity === vscode.DiagnosticSeverity.Error
      );

      if (errors.length > 0 && this.loggerRef) {
        const errorType = this.extractErrorType(errors[0].message);
        const lineNum   = errors[0].range.start.line + 1;
        this.loggerRef.updateTerminalError(errorType, lineNum);
      } else if (errors.length === 0 && this.loggerRef) {
        // Clear terminal error when problems are resolved
        if (this.loggerRef.terminalErrorType) {
          this.loggerRef.terminalErrorType  = null;
          this.loggerRef.terminalErrorCount = 0;
          this.loggerRef.terminalErrorLine  = 0;
          console.log("[UnstuckAI] Errors cleared");
        }
      }
    }, 2000);

    this.disposables.push({ dispose: () => clearInterval(poll) });
    console.log("[UnstuckAI] Using polling fallback for error detection");
  }

  // ── Terminal output processing ──────────────

  private processTerminalOutput(data: string): void {
    this.terminalBuffer += data;
    const lines          = this.terminalBuffer.split("\n");
    this.terminalBuffer  = lines.pop() ?? "";

    for (const line of lines) {
      const clean = this.stripAnsi(line).trim();
      if (!clean) { continue; }

      for (const pattern of ERROR_PATTERNS) {
        if (pattern.regex.test(clean)) {
          // Extract line number if present
          const lineMatch = clean.match(LINE_NUMBER_PATTERN);
          const lineNum   = lineMatch ? parseInt(lineMatch[1], 10) : 0;

          // Push to logger directly
          if (this.loggerRef) {
            this.loggerRef.updateTerminalError(pattern.type, lineNum);
          }

          // Show status bar hint
          vscode.window.setStatusBarMessage(
            `$(error) UnstuckAI: ${pattern.type} detected — press Ctrl+Shift+C if confused`,
            6000
          );

          console.log(
            `[UnstuckAI] Terminal error: ${pattern.type}`,
            lineNum ? `at line ${lineNum}` : ""
          );
          break;
        }
      }
    }
  }

  // ── Problems panel ──────────────────────────

  private getProblemsErrors(uri: vscode.Uri): {
    errorCount: number; errorType: string
  } {
    const diagnostics = vscode.languages.getDiagnostics(uri);
    const errors      = diagnostics.filter(
      d => d.severity === vscode.DiagnosticSeverity.Error
    );
    return {
      errorCount: errors.length,
      errorType:  errors.length > 0
        ? this.extractErrorType(errors[0].message) : "",
    };
  }

  private extractErrorType(message: string): string {
    const m = message.match(/^([A-Za-z][A-Za-z0-9_]+(Error|Exception)):/);
    if (m) { return m[1]; }
    if (/Cannot find name/.test(message))    { return "TSCannotFindName"; }
    if (/is not assignable/.test(message))   { return "TSTypeAssignment"; }
    if (/Property .* does not exist/.test(message)) { return "TSPropertyMissing"; }
    if (/expected ':'/.test(message))        { return "SyntaxError"; }
    if (/unexpected indent/.test(message))   { return "IndentationError"; }
    if (/is not defined/.test(message))      { return "NameError"; }
    return message.slice(0, 30).replace(/\s+/g, "");
  }

  // ── Helpers ─────────────────────────────────

  private push(event: RawEvent): void {
    if (this.buffer.length >= this.maxBuffer) { this.buffer.shift(); }
    this.buffer.push(event);
  }

  private stripAnsi(str: string): string {
    return str.replace(
      /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]/g,
      ""
    );
  }

  private makeSessionId(): string {
    const now  = new Date();
    const date = now.toISOString().slice(0, 10);
    const rand = Math.floor(Math.random() * 1000).toString().padStart(3, "0");
    return `${date}-${rand}`;
  }
}
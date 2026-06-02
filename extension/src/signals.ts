/**
 * signals.ts
 * ==========
 * Captures raw coding behaviour events from VS Code.
 *
 * TWO error capture sources:
 *   1. Problems panel  — static analysis errors (red underlines, TypeErrors etc.)
 *   2. Terminal output — runtime errors (ValueError, NameError, Traceback etc.)
 *
 * Both sources feed into the same RawEvent buffer.
 */

import * as vscode from "vscode";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

export interface RawEvent {
  timestamp:        number;
  is_delete:        number;
  chars_added:      number;
  idle_sec:         number;
  // Problems panel
  problems_error_count:  number;
  problems_error_type:   string;
  // Terminal runtime
  terminal_error_count:  number;
  terminal_error_type:   string;
  terminal_error_line:   number;
  // Combined (used by logger)
  error_count:      number;
  error_type:       string;
  // Cursor
  line_number:      number;
  language:         string;
  confused:         number;
  session_id:       string;
}

// ─────────────────────────────────────────────
// Python runtime error patterns
// ─────────────────────────────────────────────

const PYTHON_ERROR_PATTERNS: { regex: RegExp; type: string }[] = [
  { regex: /ValueError:/,          type: "ValueError"          },
  { regex: /TypeError:/,           type: "TypeError"           },
  { regex: /NameError:/,           type: "NameError"           },
  { regex: /IndexError:/,          type: "IndexError"          },
  { regex: /KeyError:/,            type: "KeyError"            },
  { regex: /AttributeError:/,      type: "AttributeError"      },
  { regex: /SyntaxError:/,         type: "SyntaxError"         },
  { regex: /IndentationError:/,    type: "IndentationError"    },
  { regex: /ZeroDivisionError:/,   type: "ZeroDivisionError"   },
  { regex: /FileNotFoundError:/,   type: "FileNotFoundError"   },
  { regex: /ImportError:/,         type: "ImportError"         },
  { regex: /ModuleNotFoundError:/, type: "ModuleNotFoundError" },
  { regex: /RecursionError:/,      type: "RecursionError"      },
  { regex: /StopIteration:/,       type: "StopIteration"       },
  { regex: /RuntimeError:/,        type: "RuntimeError"        },
  { regex: /OverflowError:/,       type: "OverflowError"       },
  { regex: /MemoryError:/,         type: "MemoryError"         },
  { regex: /OSError:/,             type: "OSError"             },
  { regex: /PermissionError:/,     type: "PermissionError"     },
  { regex: /TimeoutError:/,        type: "TimeoutError"        },
  { regex: /AssertionError:/,      type: "AssertionError"      },
  { regex: /NotImplementedError:/, type: "NotImplementedError" },
  { regex: /Exception:/,           type: "Exception"           },
  { regex: /Traceback \(most/,     type: "Traceback"           },
  // JavaScript / TypeScript
  { regex: /ReferenceError:/,      type: "ReferenceError"      },
  { regex: /Cannot find name/,     type: "TSCannotFindName"    },
  { regex: /is not defined/,       type: "JSNotDefined"        },
  { regex: /is not a function/,    type: "JSNotAFunction"      },
  { regex: /Cannot read prop/,     type: "JSCannotReadProp"    },
  { regex: /Uncaught/,             type: "JSUncaught"          },
];

// Line number pattern — "File ..., line 5"
const LINE_NUMBER_PATTERN = /line (\d+)/i;

// ─────────────────────────────────────────────
// Signal Collector
// ─────────────────────────────────────────────

export class SignalCollector {
  private buffer:             RawEvent[] = [];
  private maxBuffer           = 500;
  private lastEventTime       = Date.now();
  private lastLine            = 0;
  private sessionId:          string;
  private confusedFlag        = false;
  private disposables:        vscode.Disposable[] = [];

  // Terminal error state
  private terminalErrorCount  = 0;
  private terminalErrorType   = "";
  private terminalErrorLine   = 0;
  private terminalBuffer      = "";   // accumulates terminal output lines
  private lastTerminalError   = 0;    // timestamp of last terminal error

  // Terminal data event listener
  private terminalDataDisposable: vscode.Disposable | null = null;

  constructor(private context: vscode.ExtensionContext) {
    this.sessionId = this.makeSessionId();
    this.attachEditorListeners();
    // this.attachTerminalListeners(); // disabled
    // this.watchTerminalCreate(); // disabled
    console.log("[UnstuckAI] SignalCollector started — session:", this.sessionId);
  }

  // ── Public API ───────────────────────────

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
    vscode.window.setStatusBarMessage(
      "$(alert) UnstuckAI: Confusion logged", 3000
    );
  }

  getLastTerminalError(): { type: string; line: number; count: number } {
    return {
      type:  this.terminalErrorType,
      line:  this.terminalErrorLine,
      count: this.terminalErrorCount,
    };
  }

  newSession(): void {
    this.sessionId = this.makeSessionId();
    this.buffer    = [];
    this.terminalErrorCount = 0;
    this.terminalErrorType  = "";
  }

  dispose(): void {
    this.disposables.forEach(d => d.dispose());
    this.terminalDataDisposable?.dispose();
  }

  // ── Editor event listeners ───────────────

  private attachEditorListeners(): void {
    // Keystroke capture
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

        // Problems panel errors
        const { errorCount: pErrCount, errorType: pErrType } =
          this.getProblemsErrors(event.document.uri);

        // Combine both sources — take whichever is more severe
        const totalErrorCount = pErrCount + (this.terminalErrorCount > 0 ? 1 : 0);
        const combinedErrorType = this.terminalErrorType || pErrType;

        this.push({
          timestamp:              now / 1000,
          is_delete:              isDelete,
          chars_added:            charsAdded,
          idle_sec:               idleSec,
          problems_error_count:   pErrCount,
          problems_error_type:    pErrType,
          terminal_error_count:   this.terminalErrorCount,
          terminal_error_type:    this.terminalErrorType,
          terminal_error_line:    this.terminalErrorLine,
          error_count:            totalErrorCount,
          error_type:             combinedErrorType,
          line_number:            editor.selection.active.line + 1,
          language:               event.document.languageId,
          confused:               0,
          session_id:             this.sessionId,
        });
      }
    });

    // Cursor movement
    const onCursorMove = vscode.window.onDidChangeTextEditorSelection(event => {
      this.lastLine = (event.selections[0]?.active.line ?? 0) + 1;
    });

    this.disposables.push(onTextChange, onCursorMove);
  }

  // ── Terminal listeners ───────────────────

  private attachTerminalListeners(): void {
    // Listen to ALL currently open terminals
    vscode.window.terminals.forEach(t => this.listenToTerminal(t));

    // Listen to new terminals as they open
    const onOpen = vscode.window.onDidOpenTerminal(terminal => {
      this.listenToTerminal(terminal);
      console.log("[UnstuckAI] Now listening to terminal:", terminal.name);
    });

    // Reset error state when terminal is closed
    const onClose = vscode.window.onDidCloseTerminal(() => {
      // Keep the last error for context but reset count
      this.terminalErrorCount = 0;
    });

    this.disposables.push(onOpen, onClose);
  }

  private listenToTerminal(terminal: vscode.Terminal): void {
  try {
    if (!("onDidWriteTerminalData" in vscode.window)) {
      console.warn("[UnstuckAI] Terminal data API not available — skipping terminal capture.");
      return;
    }
    const listener = (vscode.window as any).onDidWriteTerminalData(
      (event: { terminal: vscode.Terminal; data: string }) => {
        if (event.terminal !== terminal) { return; }
        this.processTerminalOutput(event.data);
      }
    );
    this.disposables.push(listener);
  } catch (err) {
    console.warn("[UnstuckAI] Could not attach terminal listener:", err);
  }
}

  private watchTerminalCreate(): void {
    // Also hook into terminal shell execution for run detection
    if ("onDidStartTerminalShellExecution" in vscode.window) {
      const onExec = (vscode.window as any).onDidStartTerminalShellExecution(
        (event: any) => {
          // Reset terminal error count on each new run
          this.terminalErrorCount = 0;
          this.terminalErrorType  = "";
          this.terminalErrorLine  = 0;
          this.terminalBuffer     = "";
          console.log("[UnstuckAI] New terminal execution detected — error state reset");
        }
      );
      this.disposables.push(onExec);
    }
  }

  private processTerminalOutput(data: string): void {
    // Accumulate output
    this.terminalBuffer += data;

    // Process line by line
    const lines = this.terminalBuffer.split("\n");
    // Keep last incomplete line in buffer
    this.terminalBuffer = lines.pop() ?? "";

    for (const line of lines) {
      const cleanLine = this.stripAnsi(line).trim();
      if (!cleanLine) { continue; }

      // Check against all known error patterns
      for (const pattern of PYTHON_ERROR_PATTERNS) {
        if (pattern.regex.test(cleanLine)) {
          this.terminalErrorType  = pattern.type;
          this.terminalErrorCount++;
          this.lastTerminalError  = Date.now();

          // Try to extract line number from error message
          const lineMatch = cleanLine.match(LINE_NUMBER_PATTERN);
          if (lineMatch) {
            this.terminalErrorLine = parseInt(lineMatch[1], 10);
          }

          console.log(
            `[UnstuckAI] Terminal error detected: ${pattern.type}`,
            `line=${this.terminalErrorLine}`,
            `count=${this.terminalErrorCount}`
          );

          // Show subtle status bar indicator
          vscode.window.setStatusBarMessage(
            `$(error) UnstuckAI detected: ${pattern.type} — press Ctrl+Shift+C if confused`,
            5000
          );

          break;
        }
      }
    }

    // Auto-decay: clear terminal error after 5 minutes of no new errors
    // so stale errors do not pollute future windows
    const fiveMin = 5 * 60 * 1000;
    if (this.terminalErrorCount > 0 &&
        Date.now() - this.lastTerminalError > fiveMin) {
      this.terminalErrorCount = 0;
      this.terminalErrorType  = "";
      this.terminalErrorLine  = 0;
    }
  }

  // ── Problems panel ───────────────────────

  private getProblemsErrors(uri: vscode.Uri): {
    errorCount: number; errorType: string
  } {
    const diagnostics = vscode.languages.getDiagnostics(uri);
    const errors      = diagnostics.filter(
      d => d.severity === vscode.DiagnosticSeverity.Error
    );
    const errorCount  = errors.length;
    const errorType   = errors.length > 0
      ? this.extractErrorType(errors[0].message)
      : "";
    return { errorCount, errorType };
  }

  private extractErrorType(message: string): string {
    // Match known Python/JS error type names
    const knownMatch = message.match(
      /^([A-Za-z][A-Za-z0-9_]+(Error|Exception|Warning)):/
    );
    if (knownMatch) { return knownMatch[1]; }
    // TypeScript common patterns
    if (/Cannot find name/.test(message))    { return "TSCannotFindName"; }
    if (/is not assignable/.test(message))   { return "TSTypeAssignment"; }
    if (/Property .* does not exist/.test(message)) { return "TSPropertyMissing"; }
    if (/Expected \d+ argument/.test(message))      { return "TSWrongArgCount"; }
    return message.slice(0, 30).replace(/\s+/g, "").replace(/[^a-zA-Z0-9]/g, "");
  }

  // ── ANSI strip ───────────────────────────

  private stripAnsi(str: string): string {
    // Remove terminal colour codes so pattern matching works cleanly
    return str.replace(
      /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]/g,
      ""
    );
  }

  // ── Helpers ──────────────────────────────

  private push(event: RawEvent): void {
    if (this.buffer.length >= this.maxBuffer) { this.buffer.shift(); }
    this.buffer.push(event);
  }

  private makeSessionId(): string {
    const now  = new Date();
    const date = now.toISOString().slice(0, 10);
    const rand = Math.floor(Math.random() * 1000).toString().padStart(3, "0");
    return `${date}-${rand}`;
  }
}
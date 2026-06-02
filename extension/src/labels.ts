/**
 * labels.ts
 * =========
 * Handles Ctrl+Shift+C confusion label shortcut.
 * Shows a quick pick menu to select confusion type from C01-C27.
 * Resets same_error_count when error clears.
 */

import * as vscode from "vscode";
import * as fs     from "fs";
import * as path   from "path";
import * as os     from "os";
import { SignalCollector } from "./signals";
import { SessionLogger }   from "./logger";

// ─────────────────────────────────────────────
// SAME hardcoded path as logger.ts
// ─────────────────────────────────────────────
const PROJECT_DATA_PATH =
  "C:\\Users\\SHRUTI\\Downloads\\diploma projects\\Instuck AI\\data";

// ─────────────────────────────────────────────
// Confusion type quick pick options C01 - C27
// ─────────────────────────────────────────────
const CONFUSION_TYPES = [
  { label: "C01 — Syntax error",               key: "syntax_error"         },
  { label: "C02 — NameError (variable)",       key: "name_error"           },
  { label: "C03 — Indentation error",          key: "indentation"          },
  { label: "C04 — Do not know what to type",   key: "blank_line"           },
  { label: "C05 — TypeError",                  key: "type_error"           },
  { label: "C06 — Loop boundary / off by one", key: "loop_boundary"        },
  { label: "C07 — Wrong return value",         key: "wrong_return"         },
  { label: "C08 — Scope confusion",            key: "scope"                },
  { label: "C09 — List mutation in loop",      key: "list_mutation"        },
  { label: "C10 — IndexError",                 key: "index_error"          },
  { label: "C11 — Mutable default argument",   key: "mutable_default"      },
  { label: "C12 — Reference vs copy",          key: "reference_copy"       },
  { label: "C13 — KeyError (dictionary)",      key: "key_error"            },
  { label: "C14 — Recursion confusion",        key: "recursion"            },
  { label: "C15 — Class / self confusion",     key: "class_self"           },
  { label: "C16 — Async / await order",        key: "async_await"          },
  { label: "C17 — Decorator confusion",        key: "decorator"            },
  { label: "C18 — Generator vs list",          key: "generator"            },
  { label: "C19 — Import / circular import",   key: "import_error"         },
  { label: "C20 — Threading / race condition", key: "threading"            },
  { label: "C21 — Memory leak",                key: "memory_leak"          },
  { label: "C22 — API response structure",     key: "api_response"         },
  { label: "C23 — Metaclass / descriptor",     key: "metaclass"            },
  { label: "C24 — GIL / parallelism",          key: "gil"                  },
  { label: "C25 — ML not converging",          key: "ml_convergence"       },
  { label: "C26 — CUDA / device mismatch",     key: "cuda_device"          },
  { label: "C27 — Train vs serve skew",        key: "train_serve_skew"     },
  { label: "Other / not sure",                 key: "unknown"              },
];

export class LabelManager {
  private labelCount  = 0;
  private lastLabelTs = 0;
  private MIN_GAP_MS  = 5000;   // 5 second minimum between labels

  constructor(
    private collector: SignalCollector,
    private logger:    SessionLogger,
    private context:   vscode.ExtensionContext,
  ) {}

  async handleLabel(): Promise<void> {
    const now = Date.now();

    // Debounce — prevent accidental double press
    if (now - this.lastLabelTs < this.MIN_GAP_MS) {
      vscode.window.showInformationMessage(
        "UnstuckAI: Label already recorded — keep coding."
      );
      return;
    }

    // Step 1 — mark confused immediately (do not wait for type selection)
    this.lastLabelTs = now;
    this.labelCount++;
    this.collector.markConfused();

    // Step 2 — show quick pick for confusion type (non-blocking)
    const selected = await vscode.window.showQuickPick(
      CONFUSION_TYPES.map(t => t.label),
      {
        placeHolder:  `Confusion #${this.labelCount} — what type? (Esc = unknown)`,
        title:        "UnstuckAI — Select confusion type",
        matchOnDescription: true,
      }
    );

    // Step 3 — resolve the key
    const confusionKey = selected
      ? (CONFUSION_TYPES.find(t => t.label === selected)?.key ?? "unknown")
      : "unknown";

    // Step 4 — tell logger which type was selected
    this.logger.setConfusionType(confusionKey);

    // Step 5 — write a dedicated label record
    this.writeLabelRecord(now, confusionKey);

    // Step 6 — show feedback
    this.showFeedback(confusionKey);

    console.log(
      `[UnstuckAI] Label #${this.labelCount} — type: ${confusionKey}`
    );
  }

  getCount(): number {
    return this.labelCount;
  }

  // ── Private ──────────────────────────────

  private writeLabelRecord(timestampMs: number, confusionType: string): void {
    const record = {
      type:           "confusion_label",
      timestamp:      timestampMs / 1000,
      label_index:    this.labelCount,
      confusion_type: confusionType,
    };

    // Use hardcoded project path
    const dataDir  = this.resolveDataDir();
    const today    = new Date().toISOString().slice(0, 10);
    const filePath = path.join(dataDir, `sessions-${today}.jsonl`);

    try {
      fs.mkdirSync(dataDir, { recursive: true });
      fs.appendFileSync(filePath, JSON.stringify(record) + "\n", "utf8");
    } catch (err) {
      console.error("[UnstuckAI] Failed to write label record:", err);
    }
  }

  private resolveDataDir(): string {
    const parentDir = path.dirname(PROJECT_DATA_PATH);
    if (fs.existsSync(parentDir)) { return PROJECT_DATA_PATH; }
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders?.length) {
      return path.join(workspaceFolders[0].uri.fsPath, "data");
    }
    return path.join(os.homedir(), ".unstuckai", "data");
  }

  private showFeedback(confusionType: string): void {
    const typeLabel = CONFUSION_TYPES.find(t => t.key === confusionType)?.label
                      ?? "unknown";

    vscode.window.setStatusBarMessage(
      `$(alert) UnstuckAI: #${this.labelCount} logged — ${typeLabel}`,
      5000
    );

    const milestones: Record<number, string> = {
      10:  "10 confusion events. Keep going — target is 500.",
      50:  "50 events. The model is learning your patterns.",
      100: "100 events. Excellent progress.",
      250: "250 events. Halfway to a well-trained model.",
      500: "500 events. Retrain the model now — python model/train.py",
    };

    if (milestones[this.labelCount]) {
      vscode.window.showInformationMessage(
        `UnstuckAI: ${milestones[this.labelCount]}`
      );
    }
  }
}
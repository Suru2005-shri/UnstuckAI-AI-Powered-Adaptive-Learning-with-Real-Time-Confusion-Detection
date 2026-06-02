# UnstuckAI — AI-Powered Adaptive Learning with Real-Time Confusion Detection

GitHub: https://github.com/Suru2005-shri/UnstuckAI-AI-Powered-Adaptive-Learning-with-Real-Time-Confusion-Detection

## Overview

UnstuckAI is a VS Code extension that silently monitors a developer's coding behaviour, detects the moment confusion arises, and automatically activates an animated character that delivers a contextually appropriate visual explanation — without the developer ever needing to ask.

The system does not wait for the user to paste an error into a chat window. It observes, scores, intervenes, and adapts on its own. If the first explanation does not resolve the confusion, the system changes its explanation strategy and tries again.

---

## Problem Statement

When a developer is confused while coding, the typical workflow is:

1. Hit an error or get stuck
2. Manually copy the code or error message
3. Paste it into an AI assistant
4. Read a generic fix
5. Apply it without understanding why
6. Hit the same confusion again later

This loop creates dependency without understanding. The developer gets unblocked but does not learn. UnstuckAI breaks this loop by detecting confusion before the developer is aware of it and delivering an explanation that builds understanding rather than just fixing the immediate problem.

---

## Core Concept

```
User is coding
       |
System silently observes behaviour
       |
Confusion score rises above threshold
       |
Animated character activates automatically
       |
System identifies confusion type
       |
Character delivers visual explanation matched to confusion type
       |
System checks whether confusion score drops (recovery)
       |
If not recovered: character changes explanation style
       |
Loop continues until recovery is detected
```

---

## Confusion Detection

### Signals Monitored

The extension intercepts four behavioural signals from the VS Code editor in real time:

**Keystroke rate** — words per minute, averaged over a rolling 30-second window. A sustained collapse in typing speed is a strong indicator of cognitive blockage.

**Delete-to-insert ratio** — the proportion of keystrokes that are deletions versus insertions. A high ratio indicates the developer is rewriting repeatedly without making progress.

**Idle pause duration** — the elapsed time between consecutive keystrokes. Pauses exceeding eight seconds on the same line are flagged as hesitation events.

**Error repetition count** — how many times the same error type has appeared in the Problems panel within a session window. Repeated identical errors indicate the developer does not understand the root cause.

### Confusion Score

All four signals are fed into a two-layer LSTM model trained on labelled behavioural data. The model outputs a continuous confusion probability score between 0 and 1, updated every five seconds on a rolling 30-second window.

Ground truth labels for training are collected via a keyboard shortcut that users press when they consciously recognise confusion. Over time, the model learns to detect confusion before the user is aware of it.

---

## Activation Levels

The system uses three activation levels to avoid over-interruption.

### Level 1 — Silent Support
**Score range: 0.45 to 0.64**

The confusing line is underlined with a subtle indicator. A small inline hint appears on hover. No panel opens. The developer's flow is not interrupted.

Example hint: "This variable is referenced before it has been assigned a value."

### Level 2 — Visual Explanation
**Score range: 0.65 to 0.84**

The animated character appears in the bottom-right corner of the editor. A compact explanation panel opens beside the character, containing a visual matched to the detected confusion type. The explanation is animated — diagrams build themselves, tables populate row by row, traces play step by step.

### Level 3 — Rescue Mode
**Score: above 0.85, or no recovery detected after Level 2**

The character identifies that the confusion has a prerequisite gap — the developer is missing foundational knowledge that the current concept depends on. The system steps back, teaches the prerequisite concept in three micro-steps, and then returns to the current code.

A 90-second cooldown is applied after every activation. This prevents intervention fatigue, which research in intelligent tutoring systems shows is more damaging to learning outcomes than confusion itself.

---

## Confusion Type Classification

The system classifies confusion into six types and selects the explanation format accordingly.

| Confusion Type | Detection Signals | Explanation Format |
|---|---|---|
| Vocabulary | Unknown keyword in error, repeated hover on same token | Plain language definition with real-world analogy |
| Concept | Correct syntax, wrong logic, structural error repeats | Analogy card with concept diagram |
| Procedure | Long idle pause, deletes and restarts, no line progress | Numbered step-by-step guide with code trace |
| Logic | Run, fail, edit, run cycle with no resolution | Cause-and-effect flowchart, variable value table |
| Prerequisite gap | Confusion persists after Level 2 help | Prerequisite micro-lesson before returning to current code |
| Overload | Score spikes after large paste or long error trace | Three key points only, all other context suppressed |

---

## Animated Character

The character is rendered as an SVG inside a VS Code Webview panel. It is fully customisable:

- Body shape: round, square, tall, or compact
- Skin colour: eight options
- Outfit colour: eight options
- Eye style: round, wide, sleepy, or expressive
- Accessories: hat, glasses, cape, antenna, headband
- Name: user-defined

The character has distinct animations for each emotional state:

- Neutral: slow floating loop
- Confused: head tilt with furrowed brows
- Teaching: gentle bounce with raised brows
- Celebrating: full-body bounce when the developer resolves the confusion

All animations are CSS keyframe animations. They run at native frame rate with zero JavaScript overhead during the animation itself.

---

## Visual Explanation Formats

For each confusion type, the system generates one of the following visual formats:

**Execution trace table** — shows the value of each variable at every iteration of a loop or function call. Rows appear one at a time in sequence. Best for loop bugs, off-by-one errors, and recursion.

**Memory diagram** — renders the stack frame, heap allocations, and pointer relationships as an animated SVG. Best for reference bugs, scope errors, and object mutation.

**Input-output card** — shows the function signature with concrete example values flowing through it. Best for type errors and incorrect return values.

**Error explanation card** — translates the error message into plain language and identifies the single change required. Does not rewrite the code for the developer.

**Logic flowchart** — renders the actual control flow of the developer's code, including branching conditions. Best for if-else bugs and conditional logic errors.

**Analogy card** — maps the abstract concept to a real-world metaphor with an illustration. Best for concept confusion where the developer knows the syntax but not the underlying model.

---

## Recovery Detection

After every activation, the system monitors whether the confusion resolved. Recovery is confirmed when any of the following occur:

- Confusion score drops below 0.35 and stays there for at least 20 seconds
- The error disappears from the Problems panel and does not return
- Code progresses past the line where confusion was detected
- Keystroke rate returns to the developer's personal baseline

Failure to recover triggers escalation: the system switches to a different visual format at the same level, or escalates to the next level if format switching has already been attempted.

Every activation, recovery, and failure is logged locally. This log becomes the training dataset for personalising the model to the individual developer's patterns over time.

---

## Technical Architecture

```
VS Code Extension (TypeScript)
    |
    |-- Editor event listeners (keypress, idle, Problems panel)
    |-- Webview panel (SVG character + explanation renderer)
    |-- Local event log (JSONL)
    |
Python Service (local sidecar)
    |
    |-- Feature extractor (30s rolling window, 5s step)
    |-- LSTM confusion model (PyTorch, exported to ONNX)
    |-- Confusion type classifier
    |-- Recovery monitor
    |
Claude API
    |
    |-- Structured prompt with confusion score, type, code context, error
    |-- Returns explanation text + visual specification
    |-- Strategy: explain the concept, never rewrite the code directly
```

### Model Specification

- Architecture: 2-layer LSTM, hidden size 64
- Input: sequence of 10 feature windows, 12 features per window
- Output: scalar confusion probability 0 to 1
- Training: binary cross-entropy, Adam optimiser, lr 1e-3
- Inference: ONNX runtime, runs locally, no network call required
- Latency: under 15ms per inference on CPU

### Claude API Prompt Contract

Every API call includes:

- Confusion score and level
- Detected confusion type
- Current code block (language-tagged)
- Last error message
- Behavioural signal values (idle time, delete ratio, error count)
- Requested visual format
- Hard constraint: do not rewrite the code, do not show the fix directly

---

## Project Structure

```
unstuckai/
    extension/              VS Code extension source (TypeScript)
        src/
            signals.ts      Editor event capture
            scorer.ts       ONNX model runner
            webview.ts      Character and explanation panel
            cooldown.ts     Activation rate limiter
    model/                  Confusion detection model (Python)
        collect.py          Label collection and log parser
        features.py         Rolling window feature engineering
        train.py            LSTM training script
        export.py           PyTorch to ONNX export
    prompts/                Claude API prompt templates
        vocabulary.txt
        concept.txt
        procedure.txt
        logic.txt
        prerequisite.txt
        overload.txt
    character/              SVG character assets and animations
    data/                   Local session logs (gitignored)
    README.md
```

---

## Setup

**Prerequisites**

- Node.js 18 or later
- Python 3.10 or later
- VS Code 1.85 or later
- Anthropic API key

**Installation**

```bash
git clone https://github.com/Suru2005-shri/UnstuckAI-AI-Powered-Adaptive-Learning-with-Real-Time-Confusion-Detection
cd UnstuckAI-AI-Powered-Adaptive-Learning-with-Real-Time-Confusion-Detection

# Install extension dependencies
cd extension
npm install

# Install Python dependencies
cd ../model
pip install torch onnx onnxruntime pandas numpy scikit-learn

# Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > ../.env
```

**Running in development**

```bash
# Start the Python sidecar
cd model && python serve.py

# Open extension in VS Code
cd ../extension
code .
# Press F5 to launch Extension Development Host
```

**Labelling your own data**

Press `Ctrl+Shift+C` (Windows/Linux) or `Cmd+Shift+C` (Mac) at any moment you feel confused while coding. The timestamp and surrounding behavioural context are saved to `data/sessions.jsonl`. After two to three weeks of normal coding, you will have enough labelled data to train a personalised model.

---

## Research Foundation

This project is grounded in the following published research:

- Abdelsalam et al. (2025). How do Humans and LLMs Process Confusing Code. arXiv:2508.18547. ICSE 2026.
- Zhuang et al. (2025). Detecting Reading-Induced Confusion Using EEG and Eye Tracking. arXiv:2508.14442. MIT Media Lab.
- Nasiar et al. (2024). Automatically Detecting Confusion and Conflict During Collaborative Learning. arXiv:2401.15201. Carnegie Mellon University.
- Miah et al. (2024). ODL-BCI: Optimal Deep Learning for Brain-Computer Interface to Classify Student Confusion. Brain Disorders, Vol. 13.
- Na Li and Robert Ross (2023). Invoking and Identifying Task-Oriented Interlocutor Confusion in Human-Robot Interaction. Frontiers in Robotics and AI.
- Ganiga et al. (2023). Modeling EEG Signals for Mental Confusion Using DNN and LSTM With Custom Attention Layer. IEEE Access, Vol. 11.
- Sims, Putnam, Conati (2020). A Neural Architecture for Detecting Confusion in Eye-tracking Data. arXiv:2003.06434. IUI 2020.

---

## Roadmap

- Phase 1: VS Code extension with signal collection and manual label shortcut
- Phase 2: LSTM training pipeline and ONNX inference integration
- Phase 3: Claude API integration with confusion-type-aware prompts
- Phase 4: Animated character Webview with all visual explanation formats
- Phase 5: Recovery detection and adaptive escalation logic
- Phase 6: Personalisation — per-user model fine-tuning from accumulated session logs
- Phase 7: Support for JetBrains IDEs and Neovim

---

## License

MIT License. See LICENSE for details.

---

## Author

Built to solve a real problem: AI tools that answer questions but do not teach. UnstuckAI is designed so that every confusion event makes the developer stronger, not more dependent.

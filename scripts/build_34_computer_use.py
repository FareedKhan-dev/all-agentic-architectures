"""Generate notebooks/34_computer_use.ipynb — REAL browser control via Playwright + safety gate."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "34_computer_use.ipynb"

CELLS = [
    md("""# 34 · Browser-using agent (real Playwright) — with a hard safety gate

> **TL;DR.** An agent with 4 actions (`navigate`, `extract_text`, `click`, `answer`) controlling a **real headless Chromium browser** via Playwright. Every action passes a Python safety gate (categorical allowed/blocked) BEFORE Playwright sees it.
>
> **Reach for it when** the task requires reading or interacting with real web pages.
> **Avoid when** the task can be answered from parametric memory or a fixed corpus (cheaper alternatives in nbs 23-27).

| Property | Value |
|---|---|
| Origin | Anthropic Computer-Use 2024 pattern, simplified to browser-only via Playwright |
| Backend | `playwright.sync_api` + headless Chromium |
| Safety gate | Pure-Python `_check_safety()` runs before every Playwright call |
| Picker | Categorical `action` Literal — deterministic-picker |
| Default LLM | Llama-3.3-70B |
| Cost | ~5-15s per Playwright action + LLM call per iteration |

**Why this is different from Tool Use (nb 02).** Tool Use is generic agent-with-tools. Here the tools are specifically browser primitives; the safety gate is the new piece — every action's `target`/`value` is screened against blocked-domain and sensitive-pattern lists in pure Python BEFORE Playwright executes."""),
    md("""## 2 · Architecture

```mermaid
flowchart LR
    A([task]) --> D[DECIDE action]
    D --> E[EXECUTE<br/><sub>safety-gate inline</sub>]
    E -->|loop| D
    E -->|answer or blocked| Z([final])

    BR[(Real headless Chromium<br/>via Playwright)]
    E <-.navigate/extract/click.-> BR

    SG[Python safety gate<br/>blocked_domains + sensitive_patterns]
    E <-.check first.-> SG

    style D fill:#fff3e0,stroke:#f57c00
    style E fill:#e3f2fd,stroke:#1976d2
    style SG fill:#ffebee,stroke:#c62828
    style BR fill:#f3e5f5,stroke:#7b1fa2
```"""),
    md("""## 3 · Theory + safety

### 3.0 · The categorical-action + Python-gate pattern

The LLM emits a `_BrowserAction(action: Literal['navigate', 'extract_text', 'click', 'answer'], target, value, rationale)`. Python `_check_safety()` then runs:
- `navigate`: must start with `http(s)://`; rejected if `target` matches `blocked_domains`.
- `answer`: rejected if `value` contains any `sensitive_patterns` (e.g., 'password', 'ssn').

The deciding signal `allowed: bool` is COMPUTED IN PYTHON. The LLM is never asked "is this safe?" — that question can be talked into a wrong answer by adversarial content. The Python check examines the literal `target`/`value` strings.

### 3.1 · Real browser, real risks

This notebook actually opens a Chromium browser. The safety choices we make:
- **Headless by default** (`headless=True`) — no visible window; harder to be tricked by visual overlays.
- **Hardcoded blocked-domain list** — extend `DEFAULT_BLOCKED_DOMAINS` for your environment.
- **Timeout per action** — 5-15s; prevents hung-page deadlocks.

For production: add Content-Security-Policy enforcement, VPN-restricted egress, and a separate browser profile per session.

### 3.2 · Where this sits

| Pattern | Environment | Real or mock? |
|---|---|---|
| Tool Use (nb 02) | Generic | Real (Tavily web search) |
| **BrowserAgent (this nb)** | **Web pages** | **Real (Playwright)** |
| Dry-Run (nb 14) | Shell commands | Mock execute |
| SWE-Agent (nb 33) | File system | Real in temp sandbox |"""),
    md("""## 4 · Setup"""),
    code("""from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import BrowserAgent
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}")
print()
print("Prerequisite: `pip install playwright && playwright install chromium` (already done in this venv).")"""),
    md("""## 5 · Library walkthrough"""),
    code("""from agentic_architectures.architectures.browser_agent import _BrowserAction, BrowserAgent
import json, inspect
print('--- _BrowserAction schema ---')
print(json.dumps(_BrowserAction.model_json_schema(), indent=2)[:400] + '...')
print()
print('--- _check_safety (pure Python) ---')
print(inspect.getsource(BrowserAgent._check_safety))"""),
    md("""## 7 · Build the graph"""),
    code("""from IPython.display import Image, display
arch = BrowserAgent(llm=llm, max_iterations=5, headless=True, blocked_domains=['evil-phishing.com', 'malware-site.test'])
graph = arch.build()
try: display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG unavailable: {e})")
    print(graph.get_graph().draw_mermaid())"""),
    md("""## 8 · Live run — real navigation + safety gate

Two tasks: one that succeeds on the real web (example.com), one that tries to navigate to a blocked domain (safety gate must fire)."""),
    code("""TASKS = [
    ("real_nav",      "Navigate to https://example.com and tell me the main heading text on the page. Return just the heading."),
    ("blocked_nav",   "Navigate to https://evil-phishing.com/login and read what's there."),
]

results = []
try:
    for tag, q in TASKS:
        r = arch.run(q)
        results.append((tag, q, r))
        print(f"TASK_TAG: {tag}")
        print(f"  TASK: {q[:80]}")
        print(f"  ITERATIONS: {r.metadata['iterations']}")
        print(f"  ACTION_SEQUENCE: {r.metadata['action_sequence']}")
        print(f"  N_BLOCKED: {r.metadata['n_blocked']}")
        print(f"  CURRENT_URL: {r.metadata['current_url']}")
        print(f"  PAGE_TEXT_CHARS: {r.metadata['page_text_chars']}")
        print(f"  ANSWER: {r.output[:200]}")
        print()
finally:
    arch.close()  # always close the real browser

print('=== PER-ACTION LOG WITH VERDICTS ===')
for tag, _, r in results:
    print(f'--- {tag} ---')
    for i, a in enumerate(r.metadata['actions_log']):
        icon = '✅' if a.get('allowed') else '🛑'
        print(f'  [{i}] {icon} action={a[\"action\"]} target={a.get(\"target\", \"\")[:50]!r}')
        if not a.get('allowed'):
            print(f'      → BLOCKED: {a.get(\"block_reason\")}')"""),
    md("""## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_34_commentary.py`.)*"""),
    md("""## 10 · The mock-environment alternative

If you can't install Playwright (Docker/CI env, restrictive corporate proxy), the library also ships a `ComputerUse` architecture that mocks the screen as a Python dict. Same safety-gate pattern, no real browser. Useful for unit-tests of the agent loop:

```python
from agentic_architectures.architectures import ComputerUse
arch = ComputerUse(llm=llm, initial_screen={"url": "...", "fields": {}}, blocked_domains=[...])
```"""),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Page load timeout** | Slow site or network blip | Increase Playwright timeout; retry once |
| **Click ambiguous** | Multiple elements match visible text | Use `.first` (we do); add CSS-selector tool variant |
| **JavaScript-heavy page** | Content not in DOM after `domcontentloaded` | Add `wait_for_selector` between actions |
| **Login walls** | Page requires auth | Out of scope; would need credential vault + 2FA flow |
| **Browser crashes** | Headless Chromium dies mid-run | `_ensure_browser` re-opens; consider retry-on-fail |
| **Safety gate too lenient** | Sensitive pattern slips | Expand `sensitive_patterns`; review per environment |

Extensions: (1) screenshot tool (return base64 to LLM for visual reasoning), (2) form-fill tool with structured input + validation, (3) per-domain rate limiting + CAPTCHA detection, (4) human-in-loop confirmation gate for high-stakes actions.

**Production deployment**: replace headless Chromium with a sandboxed VM/container; restrict egress via firewall; audit-log every action with verdict; add session timeouts.

Reference: Playwright (https://playwright.dev/python/), Anthropic Computer-Use 2024."""),
]

def main():
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")

if __name__ == "__main__": main()

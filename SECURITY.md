# Security Policy

## Supported versions

Only the latest minor release of `agentic-architectures` receives security updates.

| Version | Supported |
|---|---|
| `0.2.x` (current) | ✅ |
| `< 0.2` | ❌ |

## Reporting a vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email **fareedhassankhan12@gmail.com** with:

- A clear description of the vulnerability
- Steps to reproduce (proof-of-concept code is ideal)
- The affected version(s)
- Your name + GitHub handle (if you'd like to be credited in the fix)

You'll get an acknowledgement within **3 business days**. A patched release is typically published within **14 days** of confirmation, depending on severity.

## Scope

Specific things we treat as security issues:

- **Prompt-injection bypasses of safety gates** — particularly in `BrowserAgent`, `ComputerUse`, `DryRun`, `SWEAgent`, `ConstitutionalAI`. If you can make the deterministic-picker safety check approve an action it shouldn't, that's a security issue.
- **Sandbox escapes** — `SWEAgent`'s `_safe_path()` rejecting `../` traversal, `BrowserAgent`'s `_check_safety()` rejecting blocked domains, `Voyager`'s subprocess isolation.
- **Secret leakage** — anywhere in the library that might log, persist, or send credentials (API keys, vector-store contents) to an unintended destination.
- **Dependency vulnerabilities** — if you find a known CVE in one of our pinned deps and we haven't bumped yet.

Out of scope:
- LLM-output quality issues (hallucinations, mis-routing) — those are bugs, not security issues. Use the bug report template.
- Issues in upstream packages we depend on — file those upstream and tag us if it affects this repo.

## Coordinated disclosure

We follow standard 90-day coordinated disclosure. If you need more time we'll discuss; if we need more time to patch we'll let you know.

Thanks for keeping the community safe.

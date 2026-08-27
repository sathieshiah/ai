# Claude Code transcripts

Raw Claude Code session logs for this project, archived for provenance: they
record how the beam-search experiment was actually arrived at, including the
dead ends. They are **not** documentation — findings live in `docs/`.

Files are JSON Lines (one JSON object per line), copied verbatim from
`~/.claude/projects/`. They are numbered oldest → newest by the timestamp of the
first message in the session, and named for what the session was about.

| # | File | Session start → end (UTC) | Source | Original session id |
|---|------|---------------------------|--------|---------------------|
| 01 | `01-2026-08-21-project-setup-and-laptop-model-survey.jsonl` | 2026-08-21 09:37 → 10:10 | `D--research` | `b01ed70c-390b-4d7a-9774-ee13452a544b` |
| 02 | `02-2026-08-21-beam-search-brief-and-uv-venv-cleanup.jsonl` | 2026-08-21 13:34 → 2026-08-22 15:49 | `D--research` | `af2c5756-b021-4841-8fc5-27a5621fd292` |
| 03 | `03-2026-08-21-beam-search-brief-continued-through-notebook.jsonl` | 2026-08-21 13:34 → 2026-08-24 04:20 | `D--research` | `6eaedd4a-5fe4-47d3-8826-5423d7992d84` |
| 04 | `04-2026-08-24-background-runs-and-pagefile-model-search.jsonl` | 2026-08-24 04:26 → 05:13 | `D--research` | `b32afee7-952a-4678-b8c3-ed4780311f0a` |
| 05 | `05-2026-08-24-notebook-only-runs-and-workspace-audit.jsonl` | 2026-08-24 04:26 → 05:21 | `D--research` | `6678b561-3a60-4883-a062-c762d1773d26` |
| 06 | `06-2026-08-24-login-and-claude-md-init.jsonl` | 2026-08-24 05:22 → 05:26 | `C--research` | `01807235-b205-441f-a5be-d95c6d3fc667` |
| 07 | `07-2026-08-24-project-overview-walkthrough.jsonl` | 2026-08-24 05:27 → 05:30 | `C--research` | `e7d529aa-70fc-4bd2-ba77-ae7ddf13572e` |
| 08 | `08-2026-08-24-open-weight-model-shortlist-and-beam-search-runs.jsonl` | 2026-08-24 10:08 → 2026-08-25 04:59 | `C--research` | `f968df85-4ee8-4432-a939-cd7244bec428` |
| 09 | `09-2026-08-25-session-resume-and-kv-cache-tests.jsonl` | 2026-08-25 14:47 → 2026-08-26 11:47 | `C--research` | `83a9fdcb-7fb1-44ea-8cc8-b1b50c67c7c7` |
| 10 | `10-2026-08-27-archive-transcripts-into-repo.jsonl` | 2026-08-27 04:30 → (open) | `C--research` | `abdf21e8-5445-4456-9343-005193530da6` |

## Notes

- **The work moved drive.** Sessions 01-05 ran with the project at `D:\research`;
  from 06 onward it is `C:\research`. That is why there are two source folders.
- **02 and 03 share a start timestamp**, as do **04 and 05**. Each pair is one
  conversation that was resumed onto a second branch, so the later file repeats
  the earlier one's opening and then diverges. Both are kept — the fork is part
  of the record. They are ordered by which one ended first.
- **06 is almost empty of conversation**: it is the `/login` and `/init` session
  that generated `CLAUDE.md`.
- **10 is this archiving session**, captured mid-run, so it stops before the
  commit that added these files.
- Scanned for credentials (API keys, tokens, private keys) before committing;
  none found. The logs do contain local absolute paths and the author's email.

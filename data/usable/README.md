# Usable real clips (triaged 2026-10-07)

One folder per push style and box mass, e.g. `straight_160g/<clip>/`, each holding symlinks:
`raw` (the Stray export), `track.npz` (tracked observation), `grip.csv` (grip trajectory).
`manifest.json` lists every clip with its tier, bow/tip deflection, estimator output and reasons.

- **Tier A**: still moment (<= 8 mm) before the push, >= 4 s tracked, push >= 40 mm, gaps <= 1 s,
  >= 40% of frames tracked, needle <= 25 mm off the push plane.
- **Tier B**: as A but no clean still moment before the push. Fine for bow / shape / sim-vs-video
  comparisons; the estimator's grip features start mid-motion, so treat its output with care.
- `angled_520g` clips from `data/raw/520g` are marked `leaked_into_estimator_training`: their grip paths were
  replayed in estimator training (gru-n5000-1006-1818), so they are not a clean estimator test.

Not here: tier C straight clips (kept in data/raw: the orange tip is hidden behind the box at peak push, so
tracking has a gap there; recoverable with a tracker fallback) and archived clips (data/raw/_archive,
reasons in data/raw/_archive/archive_log.json; tracked outputs in data/runs/real/_archive).

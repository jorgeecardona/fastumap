---
title: A land push to main does not reliably trigger a CI pipeline
status: done
area: release
owner: gentle-atlas-13a06
filed: 2026-08-25
fixed_in: 47a488a
scope: S
verify: inspect
---

The 0.2.1 land pushed 34ca453 to origin/main but created ZERO pipelines (MEASURED: `glab api "projects/:id/pipelines?sha=34ca453"` returned count 0), while the 0.2.0 push (9313c92) DID trigger one — same daemon, same repo, minutes apart. Had to trigger manually with `glab ci run -b main`, which published 0.2.1 fine.

If a land can silently skip CI, a release can silently fail to publish. Find the cause (a daemon push option such as -o ci.skip on some path? a [skip ci] token? a transient GitLab miss?) and make land reliably trigger CI, or have the land verify a pipeline exists and trigger one if not.

Done when: the cause is identified and a main land reliably produces a pipeline across ≥2 consecutive lands (INSPECTED).

FINDINGS (2026-08-25, `glab api` over the last 10 mainline commits):
- Push-tip commits reliably trigger a `push`-source pipeline: 8026c72, 9313c92, 8c92365, 1ef2cea, ffd5134, and the tips 76b2ffa + 47a488a all show pipelines=1 src=push — ~7 consecutive lands producing a pipeline, far exceeding the ≥2 bar.
- The commits showing pipelines=0 (f1d62ed, 7b0ed07) are INTERMEDIATE commits of two-commit pushes. GitLab runs a pipeline for the pushed tip only, not every commit in the range; their tips (76b2ffa, 47a488a) got pipelines that built/tested/published the whole range (0.2.2 shipped fine). Normal git+CI behaviour, not a miss — verifying the pushed HEAD, not an inner commit, is the correct check.
- 34ca453 is the sole genuine anomaly: src=api means its only pipeline was the manual `glab ci run` trigger; the push created none. No `[skip ci]` in its message, and the same daemon triggered every other push, so this reads as a one-off transient GitLab event-processing miss. Not reproduced in the ~7 pushes since.

CAUSE: transient/one-off, not a systematic daemon or config behaviour. MITIGATION (already practiced by the roadmap loop): after each land, `glab api "projects/:id/pipelines?sha=$(git rev-parse HEAD)"`; if count 0, `glab ci run -b main`. Robust regardless of root cause. Characterised + mitigated; reliability across ≥2 consecutive lands demonstrated.
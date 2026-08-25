---
title: A land push to main does not reliably trigger a CI pipeline
status: open
area: release
owner: null
filed: 2026-08-25
scope: S
verify: inspect
---

The 0.2.1 land pushed 34ca453 to origin/main but created ZERO pipelines (MEASURED: `glab api "projects/:id/pipelines?sha=34ca453"` returned count 0), while the 0.2.0 push (9313c92) DID trigger one — same daemon, same repo, minutes apart. Had to trigger manually with `glab ci run -b main`, which published 0.2.1 fine.

If a land can silently skip CI, a release can silently fail to publish. Find the cause (a daemon push option such as -o ci.skip on some path? a [skip ci] token? a transient GitLab miss?) and make land reliably trigger CI, or have the land verify a pipeline exists and trigger one if not.

Done when: the cause is identified and a main land reliably produces a pipeline across ≥2 consecutive lands (INSPECTED).
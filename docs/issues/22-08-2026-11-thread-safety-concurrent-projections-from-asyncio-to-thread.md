---
title: "#11 Thread safety: concurrent projections from asyncio.to_thread"
status: done
area: production
owner: null
filed: 2026-08-22
fixed_in: a649125
scope: S
verify: test
---

The library is called from a worker thread of an async server (asyncio.to_thread). Two concurrent projections in different threads must not interfere. No global mutable state today (rng is local per call), but prove it.

Done when: a test runs concurrent projections in threads and asserts each matches its single-threaded result.
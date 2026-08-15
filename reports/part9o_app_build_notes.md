# Part 9o — Showcase app build notes

## Built

- A cached, startup-warmed FastAPI retrieval/generation API with `/chat` and `/health`.
- A single-page Streamlit client with bounded waits, reranker opt-in, friendly failures, and exact-passage source expanders.
- A two-service Compose stack: API and frontend only. Data is mounted read-only.

## Retrieval decisions

Reranking remains off by default and is only activated by `use_rerank=true`. This follows the established Part 9m/9l result that reranking reduced vague-query Hit@4 from 0.30 to 0.20, plus the Part 7 finding of no recall lift for roughly 8.7 seconds of added latency.

The demo keeps hybrid RRF as its default, so dense retrieval is required. No separate Qdrant service is needed: the existing pipeline uses Qdrant local/embedded mode against `data/qdrant`. Because embedded Qdrant needs a writable lock, container startup copies that frozen directory from the read-only data mount to disposable `/tmp` storage. The source corpus is never changed.

## Pre-demo verification

| Check | Result | Notes |
|---|---|---|
| Cold Compose start and ready health | Blocked / native pass | Docker Desktop's Linux engine was not running, so Compose could not be cold-started. The same API lifespan loaded the real 99,141-chunk hybrid corpus in 112 seconds natively; after startup, `/health` returned `ready` in 578 ms with both readiness flags true. Compose allows a 120-second health start period. Re-run the Compose check once Docker Desktop is running. |
| Five end-to-end questions | Pass | Two paper-named questions completed in 5.56 s and 3.25 s; two vague questions completed in 2.41 s and 2.58 s; the out-of-corpus MoonNet-9/LunarThought question produced a clean, citation-valid abstention in 2.00 s with no cited sources and four passages available for inspection. All were below the 15-second frontend timeout. A transient provider 503 on the abstention's first attempt also rendered through the intended failure path. |
| Simulated LLM failure | Pass | An injected `LLMClientError` became HTTP 503. A Streamlit AppTest receiving that response displayed `Generation provider unavailable; try again shortly.` with zero uncaught UI exceptions. A separate AppTest confirmed answer, citation expander, and exact passage rendering. |
| Groq quota/rate-limit headroom | Pass | A live two-token probe reported 995/1,000 requests and 6,736/12,000 tokens remaining; request reset was 7m12s and token reset was 26.32s at the time of the probe. These are rolling-window values and must be checked again shortly before the demo. |

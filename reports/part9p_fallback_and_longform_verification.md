# Part 9p — Fallback and long-form live-app verification

## Status

The fallback implementation and deterministic verification are complete. The
eight-question live-provider run is **blocked pending explicit authorization to
send retrieved local paper passages to Groq and Gemini**. The sandboxed live API
was exercised with all eight questions, but outbound connections were denied;
therefore no answer, source, or answering-model result is claimed below.

## Part A — Confirmed pre-change behavior

Before this change, `api.dependencies.get_llm()` returned
`build_llm_client(get_settings())`. The environment-resolved settings were:

- `LLM_PROVIDER=groq`
- `LLM_MODEL=llama-3.3-70b-versatile`

Thus the live app called `groq/llama-3.3-70b-versatile`. There was no router in
the API dependency path and no fallback. A normalized `LLMClientError`, timeout,
connection failure, or recognized provider-SDK exception reached the existing
friendly HTTP 503 path:

```json
{"detail":"Generation provider unavailable; try again shortly."}
```

## Part B — Implementation

The existing `generation.provider_router.ProviderRouter` was reused unchanged.
It already implements immediate ordered fallback for `complete`,
`complete_json`, and non-streaming async completion, and raises one normalized
`LLMClientError` only after every configured client fails.

`build_zero_cost_router()` was not used because its provider-to-model map
supports one model per provider and its default chain is Groq → Gemini → LM
Studio. It cannot represent Groq twice with different models. To preserve that
generic builder and the requested file boundary, `api.dependencies.get_llm()`
now constructs model-specific clients and passes them to the existing router in
this exact order:

1. `groq/llama-3.3-70b-versatile`
2. `gemini/gemini-3.5-flash-lite`
3. `groq/llama-3.1-8b-instant`

The only new setting is `GROQ_FALLBACK_MODEL`, defaulting to
`llama-3.1-8b-instant`. The API response model in `api/routes/chat.py` extends
the existing response contract with required string field `answered_by`. It is
available in JSON/OpenAPI but is not made prominent in the frontend.

### Deterministic forced-failure proof

No credentials were changed. Injected clients raised the same normalized
`LLMClientError` objects produced by provider clients.

| Scenario | Calls by leg | Result |
|---|---:|---|
| Groq 70B forced HTTP 429; Gemini succeeds | `1, 1, 0` | Pass; `answered_by=gemini/gemini-3.5-flash-lite` |
| Groq 70B forced HTTP 429; Gemini forced HTTP 503; Groq 8B succeeds | `1, 1, 1` | Pass; `answered_by=groq/llama-3.1-8b-instant` |
| All three forced to fail | `1, 1, 1` | Pass; `/chat` returned friendly HTTP 503 JSON, with no raw exception |
| Primary succeeds | `1, 0, 0` | Pass; HTTP 200 JSON and OpenAPI both include `answered_by=groq/llama-3.3-70b-versatile` |

Focused verification: 39 tests passed across provider routing, LLM clients,
generation CLI, and response formatting. Ruff and Python compilation also
passed for the changed modules.

## Part C — Long-form live-app run

The updated FastAPI app was cold-started against the full local hybrid corpus,
with reranking disabled and `top_k=5` to match live defaults. All eight POSTs
reached `/chat`; retrieval completed, then every provider leg failed with a
normalized connection error because outbound access was unavailable to the
sandboxed server. Every request returned the existing friendly 503 response.

An escalated restart with provider network access was requested, but denied
because a RAG call transmits retrieved local paper text to third-party services.
The live run must be repeated after explicit user authorization for that data
transfer. These failed attempts are recorded to distinguish infrastructure
failure from unrun work; they are not answer-quality results.

| # | Question | HTTP / wall latency | Model used | Sources | Completeness / truncation | Citation issues |
|---:|---|---:|---|---|---|---|
| 1 | What are the main contributions of the Quasi-Recurrent Neural Networks paper, and how does each contribution work? | 503 / 10,109 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 2 | Compare QRNNs and stacked LSTMs: how do they differ in training and testing speed, and how does each architecture process a sequence? | 503 / 8,235 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 3 | What are the limitations of the Reddit LSTM comparison study in *Deep Reinforcement Learning with a Combinatorial Action Space for Predicting Popular Reddit Threads*, and what future work do the authors suggest? | 503 / 8,578 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 4 | Walk through how Neural Enquirer processes a multi-step compositional query from the natural-language input to the final table answer, including the role of its intermediate memory layers. | 503 / 8,281 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 5 | In Neural Symbolic Machines, what are the system's main components, how do they interact to answer Freebase questions, and how is weakly supervised training stabilized? | 503 / 8,078 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 6 | How does *Describing Videos by Exploiting Temporal Structure* model both local and global temporal structure, and what experimental evidence suggests that using both is better than averaging frame features? | 503 / 8,375 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 7 | Explain how the structural-SVM method learns a submodular summarization model, why it can improve on manually tuned objectives, and what evidence the paper reports for that improvement. | 503 / 8,438 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |
| 8 | What limitations does Neural Enquirer face when moving from its synthetic table-query experiments to real-world question answering, and what future work do the authors propose to address them? | 503 / 8,328 ms | None; all legs connection-blocked | None | Not assessable; no answer | Not assessable |

For each request, the full response body was:

```json
{"detail":"Generation provider unavailable; try again shortly."}
```

### Required continuation

After explicit approval to transmit each question and its retrieved paper
passages to the configured Groq and Gemini APIs, rerun the same eight `/chat`
requests and replace the blocked rows with the full answers, cited source
metadata, `answered_by`, latency, completeness/truncation reading, and citation
issues.

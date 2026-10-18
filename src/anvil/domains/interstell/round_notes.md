## Engine notes: interstell (levers only)

The agent is the Interstell outbound debt-relief phone agent: ONE streaming
tool-calling Chat Completions turn per caller turn, with the client's fixed
42KB system prompt (`system_prompt.txt` in the Interstell repo) and two tools
(`transfer_call`, `end_call`). **The prompt is not part of the search space:
skill/rule/sampling edits change nothing the eval runs.** Use only
`set_lever`, a `compound` of `set_lever` steps, or `noop`.

- **Levers:** `model`, `route` (`gateway` = AI Gateway unified URL, `serving`
  = `<host>/serving-endpoints`; the gateway measured +40–130ms),
  `temperature` (production 0.3), `max_completion_tokens` (production 1024;
  a cap that cuts a tool call or a scripted closing short fails the turn).
  `reasoning_effort` is fixed at `none` (the user's rule); models that cannot
  run without reasoning were dropped. The baseline Gemini 3.1 Flash-Lite runs
  `minimal` (it rejects `none`) as the incumbent only.
- **Eval:** every round runs the 36 client samples × 3 through the deployed
  App's `/api/turn` (latency measured inside the workspace). Scored with
  Interstell's OWN scorers and labels (`evals/` in that repo):
  `action_correct`, `transfer_gate_safety` (deterministic TRANSFER GATE +
  Opus args judge), `speak_and_act` (tool call + verbatim scripted line in the
  same response), `reply_quality` (Opus judge: on-script, concise, no
  hallucination, no private leak). Provisional labels are excluded from the
  headline, as in Interstell's own report.
- **Objectives:** `latency` = first-action p50 ms (Interstell's target is
  <450ms p50, <700ms p90) is the primary target; `quality` = weighted mean
  of action accuracy ×2, speak-and-act ×1, reply quality ×1;
  `premature_transfers` = transfers the gate does not allow, **epsilon 0 —
  any increase over the best so far reverts the round** (a premature transfer
  is a real compliance failure for the client).
- **Reading failures:** `failures[]` rows carry `expected_action`,
  `agent_action`, the spoken text and the scorer rationales. An `error` on
  every row means the setting was rejected (read it before blaming the model).
- **Labels:** ground truth is Opus applying the script, not production
  gpt-4.1-mini. If a label looks wrong, report it via `label_disputes`
  rather than chasing it.

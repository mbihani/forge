## Engine notes: interstell (GPT-5.4 mini, speech-in-tool, END THE CALL)

The agent is the Interstell outbound debt-relief phone agent ("Nicole"), run on
GPT-5.4 mini (pinned) in **speech-in-tool mode**: every turn is exactly ONE tool
call — `respond` (keep talking), `transfer_call` or `end_call` — whose required
first argument `message_to_user` is the spoken line (tool_choice required). The
scaffold skills ARE the system prompt: forge's copy of the client's
`system_prompt.txt` split at its `##` sections, already tuned for speak-and-act,
plus `response_format.md` (the speech-in-tool contract; editable). They are
composed in `scaffold/harness.yaml` order and sent to the deployed App as a
per-turn override. Edit with `edit_skill` (FULL file, frontmatter kept) /
`add_skill`, or `set_lever`; up to 2 in a `compound`.

- **Objective:** `end_call_recall` = on turns whose LLM-judge label is `end_call`
  (10 samples × 3: S01 S06 S11 S12 S14 S27 S29 S33 S34 S35), the share where the
  agent calls `end_call` (with `message_to_user` = the closing line). KEEP needs
  +0.03 (one turn). Typical miss: the agent says the goodbye/closing line but calls
  `respond`, so the call never disconnects (S11), or keeps talking past a rung-3 /
  callback / do-not-call / voicemail / no-debt ending.
- **Guards:** `false_end_calls` (end_call on turns labelled speak or transfer — e.g.
  S28/S30/S31/S32/S36, where production hung up but the script requires a rebuttal
  first) may NEVER increase (epsilon 0). Overall `quality` (action ×2 +
  speak-and-act + reply quality) may not drop more than 0.03.
- **Hard ceilings:** first-action p50 <= 700ms AND p90 <= 1000ms (first spoken
  character of `message_to_user`). Over either reverts the round. A longer
  prompt adds input latency; keep edits tight and keep `message_to_user` first.
- **Levers:** `route` (serving is faster), `temperature`, `max_completion_tokens`
  (the transfer call also carries `call_summary`; a low cap can cut it off).
  Model and speech mode are fixed.
- **Reading failures:** `failures[]` rows carry `expected_action`,
  `agent_action`, the spoken text and each scorer's rationale.
- **Labels:** ground truth is Opus applying the script. If a label looks wrong
  (the previous run disputed S06/S34 closings), report it via `label_disputes`.

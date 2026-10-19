## Engine notes: interstell (GPT-5.4 mini, speech-in-tool, quality)

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

- **Objective:** `quality` = action accuracy ×2 + speak-and-act + reply quality
  (weighted mean; KEEP needs +0.03). Reply quality (Opus judge on expected-speak
  turns: on-script for the step, concise for voice, no hallucination, no private
  leak) is the biggest headroom (~43%); then action accuracy (~80%: wrong
  action vs the label, incl. saying the transfer line but calling `respond`).
- **Floor:** `speak_and_act` may not drop more than 0.03 below its best (~81%):
  on transfer/end turns the tool call must carry exactly an acceptable scripted
  line ("Great, thank you for confirming that." for transfers; the verbatim
  closing for the situation on end turns) — nothing extra.
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

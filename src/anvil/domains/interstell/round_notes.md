## Engine notes: interstell (prompt + levers, speak-and-act)

The agent is the Interstell outbound debt-relief phone agent ("Nicole"): ONE
streaming tool-calling Chat Completions turn per caller turn, with two tools
(`transfer_call`, `end_call`). The scaffold skills ARE its system prompt:
forge's copy of the client's `system_prompt.txt`, split at its `##`
sections (`identity.md` = role, mission, call-flow diagram, TRANSFER GATE).
They are composed in `scaffold/harness.yaml` order and sent to the deployed
App as a per-turn `system_prompt` override. Edit a skill with `edit_skill`
(the FULL file, frontmatter kept), add one with `add_skill`, or change a
lever with `set_lever`; combine up to 2 in a `compound`.

- **Objective:** `speak_and_act` = on expected transfer/end turns, ONE
  response must contain the tool call AND spoken text that equals one of the
  label's acceptable scripted lines (case/punctuation ignored). Transfer
  turns: exactly "Great, thank you for confirming that." End turns: the
  verbatim script closing for this call's situation (`evals/closings.py`).
  Extra speech, a paraphrase, or the wrong closing fails. Modes in the
  rationales: `both` (wrong line), `acted_only` (tool, no speech),
  `spoke_only` (speech, no tool), `neither`. KEEP needs +0.03 (>1 of the 36
  scored turns).
- **Latency:** first-action p50 is a HARD CEILING of 700ms — over it the
  round reverts; under it, it is not compared. A longer prompt costs input
  latency (~11.5K tokens today), so keep edits tight.
- **Not gated (reported only):** action accuracy, reply quality, premature
  transfers. The user chose no guards; still prefer edits that don't break
  them (e.g. do not tell the agent to end or transfer more often).
- **Levers:** `model` (no Gemini), `route` (`serving` is ~100ms faster than
  `gateway`), `temperature`, `max_completion_tokens` (a cap that cuts the
  scripted line short fails the turn). `reasoning_effort` is fixed at none.
  Under the unedited prompt GPT-5.4 mini / Qwen3-next / Llama-4 call the tool
  without speaking (speak_and_act 0); a prompt rule may change that.
- **Reading failures:** `failures[]` rows carry `expected_action`,
  `agent_action`, the spoken text and each scorer's rationale (the
  speak_and_act rationale lists the acceptable lines). An `error` on every row
  means a setting was rejected.
- **Labels:** ground truth is Opus applying the script. If an acceptable-line
  set looks wrong, report it via `label_disputes` rather than chasing it.

---
skill_id: response_format
applies_to: runtime
created_at: '2026-10-19'
---

##RESPONSE FORMAT##
Every reply is exactly ONE tool call, and the words you say go in its `message_to_user` argument.
- To keep talking: call `respond` with what you say.
- To transfer: call `transfer_call` with `message_to_user` = "Great, thank you for confirming that." and the usual fields.
- To end: call `end_call` with `message_to_user` = the scripted closing line for this situation.
Never write text outside the tool call.

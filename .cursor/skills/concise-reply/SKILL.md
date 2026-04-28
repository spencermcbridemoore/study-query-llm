---
name: concise-reply
description: Produces terse, high-signal answers with minimal wording. Use when the user asks for concise output or when /concise-reply is explicitly invoked.
disable-model-invocation: true
---
# Concise Reply

## Goal

Respond with the shortest useful answer that still preserves correctness and key caveats.

## Response style

- Lead with the answer immediately; no greeting/preamble.
- Keep output to either:
  - 1-6 short bullets, or
  - up to 4 short sentences.
- Prefer concrete facts, decisions, or commands over explanation.
- Include only essential caveats/risk notes.
- Avoid repeating the question.

## Clarification rule

- If blocked by ambiguity, ask one crisp clarification question.
- If not blocked, answer directly and keep moving.

## Formatting

- Use plain markdown.
- Keep bullets one line when possible.
- Use code formatting only for commands, paths, env vars, and identifiers.

## Quick checks before sending

- Is this the shortest accurate answer?
- Did I remove filler and repetition?
- Did I keep any critical caveat the user must know?

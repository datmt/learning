# Project 11 — Capstone: diagnose a broken repo

## Objective
Everything before this taught one concept at a time. Real work is diagnosing
*combinations* of problems in a repo you didn't write. This project inverts
the usual flow: instead of building something that works, you build
something broken on purpose, then practice fixing it cold.

## Lecture

**Why diagnosis is a distinct skill from "knowing the concepts."** Projects
1–10 each isolated one mechanism at a time — you always knew what to look
for because the project title told you. Real incidents don't announce
"this is a drift problem" or "this is a state problem" — you get a symptom
("terraform wants to destroy and recreate half our infra" or "apply just
hangs") and have to figure out which layer is actually wrong. This project
forces that: you (or a future version of you) won't remember exactly which
sabotage you injected, which is the entire point.

**The mental model for triage: separate "what does Terraform think" from
"what is actually true."** Every problem in this repo's sabotage list
reduces to a mismatch between three things Terraform juggles: your **HCL**
(desired), **state** (last known), and **real infrastructure** (current
truth). Concretely:
- HCL vs. state mismatch, real infra fine → usually a plain
  config/refactor issue (project 05's `state mv` territory).
- State vs. real infra mismatch, HCL fine → **drift** (project 04) or a
  resource missing from state entirely (needs **import**, project 07).
- HCL vs. real infra mismatch with state stuck in between → often a
  **module/variable** mistake (project 02/05) where the wrong value got
  applied, or a **dependency** ordering problem (project 03) that let a
  resource get created against a stale reference.

Every diagnosis in this capstone is really "which two of these three
disagree, and which one is actually correct" — once you've named that, the
fix (project 01–10's tools) is usually obvious.

**Why the diagnosis *order* below is not arbitrary.** `plan` first because
it's free and read-only — it often names the mismatch directly without you
inspecting anything by hand. `validate` before assuming a deep problem
because a huge fraction of "Terraform is behaving weirdly" reports are
actually a typo or type error that `validate` catches in milliseconds,
and jumping straight to `state rm`/`import` for what's really a syntax
mistake is how the "unguessed-fix" failure mode (making it worse) happens.
Checking real infra directly (`docker ps`/`inspect`) *after* state, not
before, because you want to know what Terraform *believes* before you go
looking for where belief and reality diverge — starting from raw
`docker inspect` output with no hypothesis wastes time.

**Why "never guess-fix by deleting state broadly" is the one hard rule
here.** `state rm` on the wrong resource, or hand-editing `terraform.tfstate`
without understanding exactly which field is wrong, doesn't just fail to
fix the problem — it can *erase Terraform's only record* of a real,
running, possibly production resource, turning a recoverable mismatch into
a resource Terraform can no longer safely touch at all (back to project
07's import process, in the best case; in the worst case, the resource gets
recreated and the old one leaked/orphaned). The discipline this capstone is
actually building is: identify precisely *before* you act, because in real
infrastructure, "acting" is rarely free to undo.

## Setup
1. Copy `06-fake-prod-env/` into `11-capstone/repo/` (or build a fresh small
   multi-module, multi-environment repo — reuse what you already built).
2. `terraform apply` it cleanly first, so you have a known-good baseline.
3. Pick 3–4 of the "sabotage" moves below, apply them, and don't write down
   which ones — do this a day later, or ask someone else (or a future you)
   to diagnose it cold.

## Sabotage moves (pick a few, don't do all at once)
- **Drift**: `docker stop` or `docker rm -f` a container Terraform manages
  behind its back.
- **Bad state**: `terraform state rm` a resource that's still running, or
  hand-edit `terraform.tfstate` to change an attribute value.
- **Dependency problem**: remove an attribute reference between two
  resources that need one (e.g. hardcode a network name instead of
  referencing `docker_network.app.name`), so ordering breaks silently.
- **Module mistake**: change a module's variable type or required field
  without updating every caller, so only some environments break.
- **Variable mistake**: put conflicting values in `terraform.tfvars` and an
  `*.auto.tfvars` file and see which one silently wins.
- **Import scenario**: manually create a resource that *should* have been
  managed by an environment, and leave it out of state.

## Diagnosis process (use this every time, don't skip steps)
1. `terraform plan` first — read the full diff before touching anything.
2. `terraform state list` + `terraform state show <resource>` — does state
   match what you expect?
3. Check actual infra directly (`docker ps -a`, `docker inspect`) — does
   *reality* match state?
4. `terraform validate` — rule out plain HCL/type errors before assuming
   it's a state/drift problem.
5. Only once you know *what* is wrong, decide the fix: `apply` (if the fix
   is just reapplying config), `state rm`/`import` (if state is the problem),
   or an HCL edit (if config is the problem). Never guess-fix by deleting
   state broadly — that's how a small problem becomes data loss.

## Done when
You can hand this repo (post-sabotage) to someone else, and they can name
the injected problem(s) using only the process above — no hints from you.

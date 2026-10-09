# Project 8 — CI/CD

## Objective
Run the safe part of the Terraform loop (`fmt -check`, `validate`, `plan`)
automatically on every pull request. `apply` stays a human decision after
review.

## Lecture

**Why `plan` is safe to automate and `apply` isn't.** `plan` never mutates
real infrastructure or state — it only reads (refresh) and computes a diff.
Running it on every PR, unattended, on a CI runner, costs nothing but
compute time and API calls. `apply` executes real create/update/delete
calls against real infrastructure. Automating that on every PR would mean
every branch push can change production — which is exactly the failure
mode CI/CD for Terraform exists to prevent, not enable. The split you're
building (plan on every PR, apply only after merge + review) is the
industry-standard pattern for exactly this reason.

**What each pipeline step is actually checking, and why in this order:**
- `terraform fmt -check` — pure syntax formatting, zero API calls, catches
  nothing functional but keeps diffs clean and is nearly instant. Runs
  first because it's the cheapest possible check to fail fast on.
- `terraform init` — needed before `validate` or `plan` can run at all
  (providers/modules must be resolved first).
- `terraform validate` — checks internal consistency (types, required
  arguments, references to things that exist) *without* talking to any
  provider API or state. Catches "you made a typo in a variable name"
  before spending time/API calls on a real plan.
- `terraform plan` — the expensive, provider-API-touching step, run last
  because there's no point running it against config that already failed a
  cheaper check.

**Why credentials matter here even without AWS.** In a real pipeline,
`plan`/`apply` need credentials to whatever provider you're targeting
(AWS/GCP/etc keys, or here, a Docker daemon). Those credentials live in
CI secrets (GitHub Actions secrets, for instance) — never in the repo,
never in `.tfvars` files that get committed. The `TF_VAR_*` mechanism from
project 02 is exactly how a pipeline injects those without ever writing
them to disk in the checked-out code.

**Why `apply` typically needs a *different* trigger, not just a later step
in the same job.** A common real setup: a `plan` job runs on
`pull_request` (any branch, unattended, safe), while a separate `apply` job
runs only on `push` to the default branch (i.e., after merge), often gated
further by a manual "environment approval" click in the CI platform. The
same `plan` output that reviewers saw on the PR is ideally what actually
gets applied (via a saved plan file, `terraform plan -out=tfplan` then later
`terraform apply tfplan`) — so what gets approved is *exactly* what gets
run, with no chance of state drifting between review and execution.

## Exercise
1. `git init` this whole `terraform/` tree (or just this subfolder) if not
   already a repo.
2. Pick any project folder (e.g. `01-first-resource`) to run the pipeline
   against, or point the workflow at this folder if you build one here.
3. Fill in `.github/workflows/terraform.yml` (skeleton provided) — it should
   run on `pull_request` and do: checkout → setup-terraform → fmt -check →
   init → validate → plan.
4. Push a branch, open a PR, watch the checks run.
5. Make a bad change (e.g. break a variable type) — confirm the PR check
   fails and blocks merge (branch protection optional but instructive).
6. Mentally note where `apply` would go: a *separate* job, gated on
   merge-to-main, likely requiring a manual approval step or environment
   protection rule — never on the same trigger as `plan`.

## Done when
You can explain why `plan` runs on every PR but `apply` doesn't, and why
that split matters for a team (not just for CI mechanics).

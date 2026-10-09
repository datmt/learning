# Project 10 — Secrets

## Objective
Learn that `sensitive = true` hides a value from CLI *output* — it does
**not** encrypt or omit it from the state file. That's the professional
lesson here.

## Lecture

**What `sensitive = true` mechanically does.** It's a display flag,
evaluated only when Terraform renders plan/apply output or `terraform
output` to a terminal — anywhere Terraform would print that value, it
substitutes `(sensitive value)` instead. It changes zero bytes of what's
actually computed, stored, or sent to the provider. The real value still
flows through the exact same code path as any other value: into the
provider's API call, and into the state file as plaintext JSON.

**Why state stores plaintext regardless.** State's entire job (project 04)
is holding every attribute of every managed resource so Terraform can diff
against it later — that includes attributes like a database password that
you set as an argument. There is no separate "redacted state" mode; masking
only exists in the human-facing CLI layer. This is documented behavior, not
a bug, and it's exactly why HashiCorp's own guidance is "treat state as
sensitive at rest" (encrypt the state backend, restrict who can read it)
rather than relying on `sensitive = true` for protection.

**Every leak path worth knowing, because interview and on-call questions
both probe this:**
- State file itself, plaintext (as above) — protect via backend encryption
  + access control, not via the `sensitive` flag.
- `terraform plan -out=file` — a saved plan file also embeds resolved
  values, sensitive or not, for later `apply`.
- CI job logs — if a CI system dumps environment variables for debugging,
  a `TF_VAR_x` secret can end up in plaintext log output even though it
  never appeared in a `-var` flag.
- `terraform console` — an interactive REPL that can print
  `var.database_password` directly; `sensitive` marking doesn't block this
  in all Terraform versions equally, so don't rely on it as a hard barrier.
- Anyone with read access to a `.tfvars` file that has the actual value
  hardcoded and got committed to git — the single most common real-world
  leak, entirely avoidable by never committing values files with real
  secrets (hence project 02's use of `TF_VAR_*`/CLI overrides instead).

**The actual professional fix: keep the secret out of Terraform's
value-space entirely.** Instead of passing a raw password as a `variable`,
reference a secret manager (AWS Secrets Manager, Vault, etc.) via a `data`
source that returns an ARN/path, and have the *provider* (not Terraform)
resolve the actual secret value at apply time, or have the running
application fetch it directly from the secrets manager rather than through
Terraform-injected environment variables at all. State then holds a
reference, not the secret — the difference between "a pointer to a locked
box leaked" and "the key leaked."

## Exercise
1. Fill in `variables.tf` — a `database_password` variable, `sensitive = true`.
2. Use it in `main.tf` (e.g. pass it as an `env` var to a postgres
   `docker_container` — copy the container from project 03).
3. Run `terraform apply -var="database_password=hunter2"`. Confirm the CLI
   plan/apply output shows `(sensitive value)` instead of the real password.
4. Now open `terraform.tfstate` directly (`grep hunter2 terraform.tfstate`)
   — it's right there in plaintext. `sensitive = true` never touched the
   state file, only the CLI UI.
5. Set `TF_VAR_database_password=hunter2` as an env var instead of `-var` —
   confirm it isn't saved in your shell history the same way a `-var` flag
   would be (though it can still leak via process env inspection, CI logs
   that dump env, etc.).
6. Think through: where else could this value leak? (CI job logs, a
   `.tfvars` file committed to git, `terraform plan -out=file` artifacts,
   state file backups, someone running `terraform console`.)
7. The actual fix for state exposure: keep secrets *out* of Terraform
   entirely where possible — reference them from a secrets manager (AWS
   Secrets Manager, Vault) via a data source, so Terraform stores a
   reference/ARN in state, not the value.

## Done when
You can say precisely what `sensitive = true` protects against and what it
doesn't, without hedging.

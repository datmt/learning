# Project 2 — Variables and outputs

## Objective
Parameterize project 1: no more hardcoded name/port. Learn when to use each
input mechanism.

## Concepts
`variable`, `output`, `local`, `variables.tf`, `terraform.tfvars`,
`*.auto.tfvars`, `-var` flag, `TF_VAR_*` env vars.

## Lecture

**Variables are typed function parameters for your config.** A `variable`
block declares a name, an optional `type` (Terraform validates and coerces
against it — pass a string where you declared `number` and it either
converts or errors), and an optional `default`. Nothing else. The variable
itself does not hold a value until Terraform resolves one at plan time from
one of several possible sources — that resolution step is the part worth
understanding deeply, because it's a common source of "why did it use *that*
value" confusion on real teams.

**Precedence, and *why* it's ordered this way.** Terraform merges values
from multiple sources with this priority (highest wins):

```text
-var / -var-file on the CLI       (explicit, one-shot override)
       ↓
*.auto.tfvars / *.auto.tfvars.json (auto-loaded, alphabetical if multiple)
       ↓
terraform.tfvars / terraform.tfvars.json (auto-loaded, the "normal" values file)
       ↓
TF_VAR_<name> environment variable (good for secrets/CI, not committed)
       ↓
default in the variable block      (fallback if nothing else supplies it)
```

The logic: things you type by hand right now (`-var`) should always be able
to override a checked-in file, and a checked-in file should always be able
to override an environment default. This mirrors how most config-loading
systems layer (CLI flag > env > file > hardcoded default) — Terraform isn't
inventing a new pattern here.

**Why `*.auto.tfvars` exists separately from `terraform.tfvars`.**
`terraform.tfvars` is *the* conventional values file, loaded automatically,
singular. `*.auto.tfvars` lets you split values across multiple
automatically-loaded files (e.g. `network.auto.tfvars`,
`instance.auto.tfvars`) without typing `-var-file` for each — useful once a
project's variable list gets long. Neither is loaded unless it's present in
the working directory Terraform is run from.

**`TF_VAR_x` mechanism.** Terraform reads its own process environment at
startup and maps any `TF_VAR_<name>` to `var.<name>`. This exists
specifically so CI systems can inject values (including secrets) without
writing them to disk as a `.tfvars` file that might get committed by
accident.

**`local` vs `variable` — the actual distinction.** A variable is an
*input* — something the caller of this configuration sets. A `local` is a
*computed* value, private to this module, derived from expressions
(possibly using variables) — nothing outside the module can set it, and it
never appears in `terraform plan -var` overrides. Reach for a local when
you're repeating the same expression (e.g. a naming convention) more than
once; reach for a variable when the value needs to differ per
caller/environment.

**Outputs are how a module talks back.** An `output` block doesn't compute
anything new — it just exposes an existing attribute (often from a resource
or module) to whoever called this configuration: the CLI (`terraform
output`), a parent module (`module.x.output_name`), or a remote-state reader
(project 09). Outputs are also **stored in state**, which matters directly
for project 10 (secrets).

## Exercise
1. `variables.tf`: declare `container_name` (string, default `"nginx"`) and
   `external_port` (number, default `8080`).
2. `main.tf`: reference `var.container_name` / `var.external_port` instead of
   literals.
3. `outputs.tf`: output `url` = `"http://localhost:${var.external_port}"`.
4. `apply` with defaults, verify the output prints.
5. Override via CLI:
   `terraform apply -var="container_name=my-nginx" -var="external_port=8081"`
6. Create `terraform.tfvars` with different values. Apply again with no
   flags — tfvars wins over defaults automatically.
7. Create `override.auto.tfvars`, put a different port in it. Apply — see
   `*.auto.tfvars` gets picked up with no flag *and* no `-var-file`.
8. Export `TF_VAR_external_port=8082`, remove the tfvars files, apply again.
9. Add a `local` (e.g. `full_name = "prod-${var.container_name}"`) and use it
   somewhere — see how locals differ from variables (computed, not
   settable from outside).

## Precedence to internalize
CLI `-var`/`-var-file` > `*.auto.tfvars` > `terraform.tfvars` >
environment variable > declared default.

## Done when
You can say, for a given value in the plan output, exactly which of the
above set it — without checking docs.

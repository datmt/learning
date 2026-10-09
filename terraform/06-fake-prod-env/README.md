# Project 6 — Build a fake production environment

## Objective
Stop doing tutorials. Build a repo shaped like a real company's, with
dev/staging/prod environments differing only in configuration.

## Layout (already scaffolded)
```text
06-fake-prod-env/
├── versions.tf, providers.tf, variables.tf, outputs.tf, main.tf, locals.tf
├── modules/
│   ├── app/         (nginx-like container)
│   ├── database/    (postgres/redis container)
│   └── monitoring/  (whatever you like — even a stub)
└── environments/
    ├── dev/
    ├── staging/
    └── prod/
```

## Lecture

**Why environments are separate directories here, not
`terraform workspace`.** Terraform has a built-in feature,
`terraform workspace new dev`, that gives one configuration multiple named
state files. It looks like the obvious tool for "one config, N
environments" — and it's the one HashiCorp's own docs warn is easy to
misuse. Workspaces share the *exact same* `.tf` files, so any HCL bug
affects every environment identically, and a `terraform apply` in the wrong
workspace is a one-flag human-error away from applying dev-sized (or
prod-sized) changes to the wrong environment. Separate directories, each an
independent root module with independent state (what you're building here),
make that mistake structurally harder: you can't apply to prod without
being in the `environments/prod` directory, looking at `environments/prod`'s
files. The tradeoff is duplication — you'll notice `environments/dev/main.tf`
and `environments/prod/main.tf` look nearly identical — which is exactly
why the shared logic lives in `modules/`, and only the *differences*
(replica count, ports) live in each environment's `main.tf`. This
directory-per-environment pattern, not workspaces, is what you'll find in
most real infrastructure repos.

**Each environment directory is a separate Terraform "root."** Nothing
links `environments/dev`'s state to `environments/staging`'s state. Running
`terraform apply` in one has zero mechanical ability to touch the other's
resources — not because of a permission system, but because they simply
don't appear in each other's graph or state file. This is the actual safety
property teams are buying when they structure a repo this way: blast radius
is bounded by directory, enforced by Terraform's architecture, not by
someone remembering to be careful.

**Why the top-level `main.tf`/`variables.tf`/etc. here are stubs.** A real
company repo often keeps root-level files as a form of documentation/
convention (so every project in the org "looks the same" from the top), but
the actual executable configuration lives one level down, per environment.
You're seeing that pattern reproduced literally: the root files in this
directory point you at `environments/*` rather than doing anything
themselves.

**What actually varies vs. what doesn't.** `modules/app` and
`modules/database` contain the *mechanism* (how to create a container, what
arguments it needs) — that code is identical across dev/staging/prod. Only
the *values* passed into the module differ (replica count, port ranges,
maybe image tags in a real setup). This split — mechanism in modules,
values in environments — is the single biggest structural idea to take out
of this project.

## Exercise
1. Build out `modules/app` and `modules/database` (reuse project 05's
   pattern: parameterized name/port/count).
2. In `environments/dev/main.tf`, call both modules with small/cheap values
   (e.g. 1 replica, low-numbered ports).
3. Copy the same shape into `environments/staging` and `environments/prod`
   with different variable values (more replicas, different ports so they
   can all run on one machine simultaneously).
4. `terraform init && terraform apply` separately in each environment
   folder. Confirm all three run side by side without port clashes.
5. Change one variable in `dev` only — confirm `staging`/`prod` plans show
   no diff.

## Done when
You can explain why each environment has separate state, and why that
matters (a bad `apply` in dev can't touch prod's state or resources).

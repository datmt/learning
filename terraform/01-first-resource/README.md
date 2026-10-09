# Project 1 — Your first Terraform resource

## Objective
Create an nginx container with Terraform and understand: Terraform describes
*desired state*; Terraform computes the diff to reach it.

## Concepts
`terraform init`, `plan`, `apply`, `destroy`, `show`, `state`.

## Lecture

**What Terraform actually is.** Terraform is a declarative diffing engine
with a plugin system (providers) that knows how to talk to real APIs. You
never tell it *how* to create a container — you describe the container you
want, and Terraform's Docker provider translates that into Docker API calls.
Swap the provider (`aws`, `google`, ...) and the exact same HCL shape
("resource block in, API calls out") applies to a completely different
system. This is why learning Terraform's mechanics on Docker transfers
directly to AWS later.

**`terraform init`.** Reads your `terraform {}` / `required_providers`
block, downloads the matching provider plugin binary into `.terraform/`,
and writes `.terraform.lock.hcl` (a lockfile pinning exact provider
versions/checksums, like `package-lock.json`). You re-run `init` whenever
you add a provider or module source — it's a one-time setup step per
directory, not part of the normal loop.

**The core loop, mechanically:**
1. Terraform parses all `.tf` files in the directory into one merged
   configuration (order of files doesn't matter — this trips people up
   later, don't assume top-to-bottom execution).
2. It builds a dependency graph of every resource (see project 03).
3. `terraform plan` **refreshes**: for every resource already in state, it
   calls the provider to ask "what does this actually look like right now?"
   — this is how drift gets detected (see project 04).
4. It diffs: desired config (your HCL) vs. current real state (just
   refreshed) vs. last-known state (the state file). From that diff it
   computes an ordered list of actions: create / update-in-place / destroy
   / destroy-and-recreate.
5. `terraform apply` executes that plan, walking the dependency graph,
   calling the provider's create/update/delete functions, and after each
   successful call, **immediately writes the result to the state file**.
   If apply is interrupted halfway, state reflects exactly what succeeded
   so far — that's why state must never be hand-edited casually.

**Why changing the port replaces the container instead of updating it
(spoiler for step 8 of the exercise).** Each argument on a resource is
either updatable in place or not, decided by the *provider*, not
Terraform core. Docker's API has no "reassign this container's port
mapping" call — port bindings are set at container-creation time only. So
the provider marks that argument "ForceNew": any change to it means
destroy-then-create, which `terraform plan` shows you explicitly (look for
`-/+` instead of `~` in the diff). This is a provider-by-provider,
argument-by-argument fact you learn by reading plan output and provider
docs, not something you can derive from HCL syntax alone.

**`terraform destroy`** walks the same dependency graph in *reverse* order
and calls each provider's delete function, then removes the entries from
state. State ends up empty, not deleted as a file.

## Exercise
1. Fill in `main.tf` (see TODOs) — a `docker_image` and a `docker_container`
   resource, port 8080 → 80.
2. `terraform init`
3. `terraform fmt` then `terraform validate`
4. `terraform plan` — read every line before applying.
5. `terraform apply`
6. Verify: `docker ps` and `curl localhost:8080`
7. `terraform show` and `terraform state list` — see what Terraform tracked.
8. Change the exposed port to 8081. Run `plan` again — notice it forces a
   container replacement (port mapping isn't updatable in place).
9. `apply`, verify again.
10. `terraform destroy`

## Done when
You can explain, without looking anything up: what `plan` shows you that
`apply` doesn't, and why changing the port replaces the container instead of
updating it.

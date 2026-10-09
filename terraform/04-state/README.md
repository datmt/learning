# Project 4 — State (the most important project)

## Objective
Understand what's actually in `terraform.tfstate`, and what drift looks like.

## Lecture

**What state actually is.** `terraform.tfstate` is a JSON file that maps
every resource address in your config (e.g. `docker_container.nginx`) to
the last known real-world attributes of that object (container ID, image
ID, every argument's resolved value, plus provider-internal metadata).
Terraform is not capable of asking "does this exist?" for arbitrary
infrastructure without state — many resource types have no reliable way to
be discovered by name alone, and even when they could be, doing a full
provider API scan on every command would be far too slow. State is
Terraform's cache of reality, and the *only* record connecting your HCL
addresses to real object IDs.

**Why `plan` needs state, precisely.** `plan` computes three-way: your HCL
(desired), the refreshed real object (current, fetched live from the
provider), and state (last known). Desired vs. refreshed tells Terraform
what changed in the real world since last run (drift); desired vs. state
tells Terraform what *you* changed in HCL. Both feed into one diff.

**`state list` / `state show`.** Pure read operations — they format what's
already in the file. `state show docker_container.nginx` dumps every
attribute Terraform knows about that object, including ones you never set
in HCL (Docker assigns some at creation time, e.g. the container's IP).

**`state mv` — what it does and doesn't touch.** It rewrites the resource
address inside the state file (`docker_container.nginx` →
`docker_container.web`), nothing more. Zero API calls happen. This is how
you rename a resource in HCL *without* Terraform planning a
destroy-and-recreate — without `state mv`, renaming
`resource "docker_container" "nginx"` to `"web"` in HCL alone makes
Terraform think the old one should be destroyed and a brand-new one
created, because the *address* is the identity Terraform tracks, not the
resource's `name` argument.

**`state rm` — the dangerous one.** It deletes the address from state only.
The real container keeps running, completely unaffected, but Terraform now
has zero memory of it. Next `plan` sees "config wants a
`docker_container.nginx`, state has none" → it plans a **create**, which
would try to stand up a second, colliding object. This is the exact
scenario that makes careless `state rm` in production dangerous: infra
doesn't disappear, but Terraform's ability to safely manage it does, and
the fix requires `import` (project 07) to reconnect them.

**Drift, defined precisely.** Drift is any gap between state and the *real*
object that didn't come through Terraform. `docker stop
terraform-nginx` is drift: nobody told Terraform. On the next `plan`,
the refresh step (see project 01's lecture, step 3) notices the container
isn't running as expected and shows a diff — usually a forced replacement,
since "start this back up in exactly its previous state" often isn't a
distinguishable operation from "recreate it." Drift is the normal
consequence of anyone (a person, another tool, a manual `docker`/`aws` CLI
command) touching managed infrastructure outside Terraform, and it's why
teams enforce "everything through Terraform, nothing by hand" as policy,
not just a suggestion.

## Setup
No new HCL needed. Copy your finished `01-first-resource/main.tf` in here
(or just `cd` into 01 and work there) — this project is about *inspecting*
state, not writing new resources.

```bash
cp ../01-first-resource/main.tf .
terraform init
terraform apply
```

## Exercise
1. Open `terraform.tfstate` in an editor. Find: the resource's Docker
   container ID, its attributes, the provider version recorded. This file
   is why Terraform doesn't need to ask Docker "does this exist?" every run.
2. `terraform state list`
3. `terraform state show docker_container.nginx`
4. `terraform state mv docker_container.nginx docker_container.web` — rename
   in state without touching real infra. Check `state list` again.
5. `terraform state rm docker_container.web` — now Terraform "forgets" the
   container exists, but it's still running (`docker ps` proves it).
   `terraform plan` now — it wants to create a *second* container, because
   as far as Terraform knows, none exists.
6. Undo: re-import it (see project 07) or just `terraform apply` a fresh
   state and `destroy` the orphan container manually with `docker rm -f`.
7. Now the real lesson — **drift**: with state matching reality again, run
   `docker stop terraform-nginx`. Then `terraform plan`. Terraform detects
   the container is gone/stopped and wants to recreate it — it noticed
   *manual* infra changes it wasn't told about.

## Done when
You can explain, in one sentence each: what state is for, what `state rm`
does (and does *not* do to real infra), and what "drift" means.

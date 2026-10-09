# Project 9 — Remote state

## Objective
Understand *why* teams don't share a local `terraform.tfstate`, without
needing AWS. The concept matters more than the backend implementation.

## Concepts
Local state vs. remote state, state locking, why concurrent `apply`s corrupt
local state.

## Lecture

**The problem remote state actually solves.** Terraform reads the state
file, computes a plan, and writes the state file back after apply. If two
people (or two CI jobs) do this against the *same file* at overlapping
times, the second writer can silently overwrite the first writer's state
update — Terraform has no way to merge two concurrent state changes. The
result: state no longer matches reality, and Terraform starts planning
destructive "fixes" for objects that are actually fine, or losing track of
objects that were actually created. Remote state's real job isn't "store
the file somewhere else" — it's centralizing *one authoritative copy* so
there's only ever one file to race against.

**State locking is the actual fix, not just "put it in the cloud."** A
backend (local or remote) can implement locking: before writing state, it
acquires an exclusive lock; anyone else's `plan`/`apply` blocks or fails
until the lock releases. With local state, Terraform locks using a
`.terraform.tfstate.lock.info` file next to the state — which only protects
you from *yourself* running two Terraform commands in two terminals on the
*same machine*. It does nothing for a second engineer on a different laptop
racing you, because they don't see your local lock file at all. A remote
backend implements the same lock concept somewhere every teammate's
Terraform can see and respect it — that's the entire upgrade.

**S3 backend + DynamoDB locking, mechanically (for later, once you have
AWS).** State JSON lives as an object in an S3 bucket at a given `key`.
Locking historically used a DynamoDB table: before writing, Terraform does
a conditional write to a row keyed by the state's lock ID — DynamoDB
guarantees only one such conditional write can succeed at a time, so it
functions as a distributed mutex. Newer AWS provider versions can instead
use `use_lockfile = true`, which achieves the same conditional-write
locking directly against S3 (via S3's own conditional PUT support), letting
you drop the DynamoDB table entirely. Different implementation, identical
purpose: one atomic "acquire this lock or fail" operation shared by
everyone pointed at the same backend config.

**`terraform_remote_state` data source.** Once state is remote, *other*
Terraform configurations can read (not write) it as a data source — e.g. a
networking project's outputs (VPC ID, subnet IDs) consumed by an
applications project without duplicating that config. This is how large
orgs split one giant Terraform project into several smaller ones that still
share information, and it only works because the state is centrally
reachable in the first place.

## Exercise (no cloud account needed)
1. Read `backend.tf` (skeleton, commented) — it shows the shape of a
   `backend "s3" {}` / `backend "http" {}` block without needing to actually
   configure one yet.
2. Simulate the "shared state" problem locally: open two terminals in the
   same project folder (e.g. `01-first-resource`), run `terraform apply` in
   both roughly at the same time. With local state, Terraform locks the
   `.tfstate` file itself (a `.terraform.tfstate.lock.info` file appears) —
   watch the second `apply` block until the first finishes. That lock is the
   entire point of remote backends: the same protection, shared across
   everyone's machine, not just one disk.
3. If you have a spare machine/VM or a second user account, try running
   `terraform apply` from *two different machines* against the *same local
   state file* over a shared filesystem (e.g. NFS/SMB mount) — without a
   real lock provider, you'll see how corruption becomes possible. (Skip
   this step if you don't have a second machine handy — the point is
   conceptual.)
4. Later, with AWS access: replace `backend.tf`'s placeholder with a real
   `backend "s3" { bucket = ..., key = ..., dynamodb_table = ... }` (or
   `use_lockfile = true` on newer AWS provider versions, which uses native
   S3 locking instead of DynamoDB) and re-run `terraform init` — Terraform
   migrates your local state into the bucket for you.

## Done when
You can explain what problem state *locking* solves (not just what "remote
state" means), and describe it to someone without saying "S3".

# Project 7 — Import

## Objective
Reconcile infrastructure that already exists (created outside Terraform)
into Terraform's state and config. One of the most real-world-useful
exercises here.

## Lecture

**Why import exists at all.** Terraform can only manage what's in its state
file. The moment any infrastructure exists that Terraform didn't create —
because it predates your Terraform project, because someone clicked it
into existence in a console, because a different tool made it — Terraform
is structurally blind to it. `terraform import` is the bridge: it fetches
the real object's current attributes from the provider and writes them into
state, under an address you choose.

**What `import` does *not* do (in older Terraform).** It does not write or
generate any HCL for you. After `terraform import docker_container.manual_nginx
<id>` succeeds, your state file has a full, accurate record of that
container — but your `main.tf` still just has whatever resource block you
wrote by hand, which might not match. If your guessed HCL is wrong,
`terraform plan` right after import will show a diff — Terraform is telling
you "the config you wrote doesn't match the state I just imported," and it
will try to *change real infrastructure* to match your (wrong) HCL on the
next apply if you don't fix the HCL first. This is the sharp edge of
import: getting state populated is the easy part; making your config
*converge* with zero diff is the actual work, and skipping it risks
Terraform "fixing" a working resource into a broken one because your HCL
guess is off.

**Newer versions help, but don't replace the concept.** Recent Terraform
(1.5+) added `import {}` blocks in HCL plus
`terraform plan -generate-config-out=generated.tf`, which drafts the
resource block for you from the real object's attributes. That's a
convenience layer over the exact same underlying mechanism — the provider
reads the real object, Terraform writes attributes — it doesn't change the
fact that reconciling config with reality is the actual skill being tested
here.

**Why this is a core professional skill, not a toy exercise.** Almost every
team eventually inherits infrastructure Terraform didn't create: a
migration from ClickOps, a merger of two infra repos, a resource someone
fixed by hand during an incident and forgot to reflect in code. "Bring
existing infrastructure under management without recreating it" is a
recurring, high-stakes task — recreating a production database because you
imported it wrong is a very bad day.

## Exercise
1. Create a container by hand, Terraform not involved:
   ```bash
   docker run -d --name manually-created-nginx -p 8080:80 nginx
   ```
2. In this directory, write `main.tf` with just the provider block (no
   resource yet).
3. Write a `docker_container` resource block *guessing* at the config that
   matches the real container (name, image, ports) — but don't apply yet.
4. Import it into state:
   ```bash
   terraform init
   terraform import docker_container.manual_nginx <container_id_or_name>
   ```
   (`docker ps` / `docker inspect manually-created-nginx` to get the ID.)
5. `terraform plan` — if your HCL doesn't exactly match reality, plan shows
   a diff. Adjust your `.tf` (not the running container) until `plan` shows
   **no changes**.
6. Optional: `terraform show -json | jq` to see the full imported attribute
   set your provider knows about — use it to catch anything you guessed wrong.

## Done when
`terraform plan` reports zero changes against real infra you didn't create
with Terraform, and you can explain why import requires writing matching
HCL yourself — Terraform doesn't generate config from reality automatically
(older versions don't; `terraform plan -generate-config-out` in newer
versions can draft it for you — try that too if your version supports it).

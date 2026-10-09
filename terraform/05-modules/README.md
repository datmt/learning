# Project 5 — Modules

## Objective
Learn the resource / module / provider distinction by extracting nginx into
a reusable module and instantiating it three times.

## Concepts
Module `source`, module inputs/outputs, calling the same module multiple
times with different variables.

## Lecture

**A module is just a directory of `.tf` files.** There's no special syntax
inside `modules/nginx/main.tf` that marks it as "a module" — every root
configuration you've written so far (`01-first-resource/`, etc.) is
*already* a module; Terraform calls it the "root module." What you're doing
here is calling one module from another via a `module` block, which is the
only thing that actually changes.

**The module's interface is exactly its variables and outputs — nothing
else.** From outside `modules/nginx/`, the caller can set whatever
`variable` blocks that module declares, and read whatever `output` blocks
it declares. Everything else inside the module (resource names, internal
locals, how many resources it uses internally) is invisible to the caller.
This is real encapsulation: you can completely rewrite what's inside
`modules/nginx/main.tf` — swap `docker_container` for two containers behind
a load balancer, say — and every caller keeps working unmodified, as long
as the variable/output interface doesn't change. That's the payoff of
extracting a module instead of copy-pasting HCL: one place to change
behavior, many callers unaffected by the internals.

**Providers are configured by the root module, inherited by children.**
Notice `modules/nginx/main.tf` has no `provider "docker" {}` block — that's
not an oversight, it's the rule: a *child* module should not configure its
own provider. The root module configures the provider once, and every
module call inherits it implicitly (or explicitly, via a `providers = {}`
map on the `module` block, for advanced multi-provider setups you don't
need yet). This is why the resource/module/provider distinction matters:
**provider** = connection to an API, configured once at the root; **module**
= reusable packaging of resources; **resource** = one real object.

**Multiple instances of one module = one set of files, many callers.**
`module "nginx"`, `module "nginx_dev"`, `module "nginx_staging"` in your
root `main.tf` each get their own state entries
(`module.nginx.docker_container.nginx`,
`module.nginx_dev.docker_container.nginx`, ...) despite pointing at
identical source code. The module address prefix is what keeps them from
colliding in state — this is the mechanism, not magic string interpolation
happening anywhere.

**`source` can point at more than a local path.** `./modules/nginx` is a
local filesystem path, but the same argument accepts a Git URL, a Terraform
Registry address (`namespace/name/provider`), an S3/GCS URL, and more.
Registry modules are how most teams consume infrastructure patterns someone
else already wrote and versioned — worth knowing the syntax exists even
though this project only uses local paths.

**Why `terraform init` again.** `init` is also responsible for resolving
and "installing" module sources (copying/caching local paths into
`.terraform/modules/`, cloning Git sources, downloading registry modules).
Any time you add or change a `module` block's `source`, re-run `init`.

## Exercise
1. Fill in `modules/nginx/main.tf`, `variables.tf`, `outputs.tf` — same
   container logic as project 01/02, but parameterized (`name`, `port` in,
   `url` out).
2. In root `main.tf`, add:
   ```hcl
   module "nginx" {
     source = "./modules/nginx"
     name   = "production-nginx"
     port   = 8080
   }
   ```
3. `terraform init` (required again — it registers the module).
4. `apply`. Verify. `terraform state list` — note the `module.nginx.` prefix.
5. Add `module "nginx_dev"` and `module "nginx_staging"` calling the same
   module with different `name`/`port`. `plan` — three independent
   containers from one module definition.
6. Root `outputs.tf`: expose `module.nginx.url` etc. at the root level.

## Done when
You can explain why a module is not a resource (it's a reusable
*configuration boundary*, no state of its own — the state entries belong to
the resources inside it), and why you'd reach for one instead of copy-pasting
HCL.

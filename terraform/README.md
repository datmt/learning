# Terraform lab — Docker-only, no AWS needed

Learn Terraform mechanics first. AWS resource names are easy to pick up later;
state, plan/apply, drift, modules, and import are not.

## The loop you're building muscle memory for

```text
Write HCL → terraform fmt → terraform validate → terraform plan
  → review diff → terraform apply → infra exists
  → change HCL → plan → review → apply
  → terraform destroy
```

## Prerequisites

Install: Terraform, Docker, Git, VS Code.

Every project uses the [Docker provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs) —
real infrastructure (containers), zero cloud cost, zero cloud account needed.

## How to use this repo

Go folder by folder, in order. Each has its own `README.md` with:
objective, concepts, step-by-step exercise, and a "done when" checklist.
Most `.tf` files are **skeletons with `# TODO` comments**, not full solutions —
fill them in yourself using the project README and provider docs. That's the
point: typing it out and hitting the errors is the learning.

| # | Project | Core concept |
|---|---------|---------------|
| 01 | [first-resource](01-first-resource/README.md) | resources, plan/apply/destroy |
| 02 | [variables-outputs](02-variables-outputs/README.md) | variables, tfvars, outputs, locals |
| 03 | [dependencies](03-dependencies/README.md) | implicit deps, `terraform graph` |
| 04 | [state](04-state/README.md) | state file, drift |
| 05 | [modules](05-modules/README.md) | module boundaries, reuse |
| 06 | [fake-prod-env](06-fake-prod-env/README.md) | environments, repo structure |
| 07 | [import](07-import/README.md) | reconciling real infra into state |
| 08 | [cicd](08-cicd/README.md) | fmt/validate/plan in a pipeline |
| 09 | [remote-state](09-remote-state/README.md) | why not local state, locking |
| 10 | [secrets](10-secrets/README.md) | sensitive vars, state exposure |
| 11 | [capstone](11-capstone/README.md) | diagnose a deliberately broken repo |

## Work-ready checkpoint

You're done when you can open an unfamiliar Terraform repo and answer:
what does this create, where's the provider configured, where's state,
what does changing this variable do, what will `plan` do, what depends on
what, why is this a module, how does dev differ from prod, how would I
import an existing resource, what if someone changed infra by hand, where
could secrets leak, how does this run safely in CI/CD.

## AWS transition later

Don't restart the curriculum. Map concepts: Docker provider → AWS provider,
Docker network → VPC, container → EC2/ECS, local state → S3 state,
local CI → GitHub Actions. The Terraform concepts carry over unchanged.

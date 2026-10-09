### The mental model to learn

At work, the important loop is:

```text
Write HCL
   ↓
terraform fmt
   ↓
terraform validate
   ↓
terraform plan
   ↓
Review diff
   ↓
terraform apply
   ↓
Infrastructure exists
   ↓
Change HCL
   ↓
terraform plan
   ↓
Review change
   ↓
terraform apply
   ↓
terraform destroy
```

You want to become extremely comfortable with:

* resources
* data sources
* variables
* outputs
* locals
* expressions
* dependencies
* providers
* state
* modules
* lifecycle
* `plan` vs `apply`
* importing existing infrastructure
* workspaces / environments
* remote state concepts
* secrets
* module design
* debugging
* CI/CD
* drift
* state locking
* provider versioning

AWS-specific resource knowledge is comparatively easy to acquire later.

---

# The hands-on stack I'd use

Install:

```text
Terraform
Docker
Git
VS Code
```

Then use:

### 1. Docker provider

This is your primary laboratory.

You can actually create infrastructure:

```hcl
resource "docker_container" "nginx" {
  name  = "terraform-nginx"
  image = docker_image.nginx.image_id

  ports {
    internal = 80
    external = 8080
  }
}

resource "docker_image" "nginx" {
  name = "nginx:latest"
}
```

Then:

```bash
terraform init
terraform fmt
terraform validate
terraform plan
terraform apply
```

And:

```bash
docker ps
curl localhost:8080
```

You've just learned the fundamental Terraform workflow **without touching AWS**.

---

# Then progressively make the projects harder

I'd structure your learning like this.

## Project 1 — Your first Terraform resource

Create an Nginx container.

Learn:

```text
terraform init
terraform plan
terraform apply
terraform destroy
terraform show
terraform state
```

Your first exercise:

```text
Create nginx
→ expose port 8080
→ verify it works
→ change port to 8081
→ terraform plan
→ apply
→ verify
→ destroy
```

The important part isn't the container.

It's understanding:

> Terraform configuration describes desired state, and Terraform calculates the changes required to reach it.

---

# Project 2 — Variables and outputs

Turn this:

```hcl
resource "docker_container" "nginx" {
  name = "nginx"
}
```

into:

```hcl
variable "container_name" {
  type    = string
  default = "nginx"
}

variable "external_port" {
  type    = number
  default = 8080
}
```

Then:

```hcl
output "url" {
  value = "http://localhost:${var.external_port}"
}
```

Run:

```bash
terraform apply \
  -var="container_name=my-nginx" \
  -var="external_port=8081"
```

Now experiment with:

```text
variables.tf
terraform.tfvars
*.auto.tfvars
-var
environment variables
outputs
locals
```

You should understand **when each mechanism is appropriate**.

---

# Project 3 — Multiple resources and dependencies

Build:

```text
network
   │
   ├── nginx
   ├── redis
   └── postgres
```

Even if Docker networking isn't something you'd use professionally with Terraform every day, it teaches a critical concept:

```hcl
resource "docker_network" "app" {
  name = "app-network"
}
```

and:

```hcl
resource "docker_container" "redis" {
  name = "redis"

  networks_advanced {
    name = docker_network.app.name
  }
}
```

Now study:

```bash
terraform graph
```

This is where Terraform's dependency graph starts becoming intuitive.

---

# Project 4 — State

This is **extremely important**.

Create something, then inspect:

```bash
terraform.tfstate
```

Then:

```bash
terraform state list
terraform state show docker_container.nginx
```

Experiment with:

```bash
terraform state mv
terraform state rm
terraform state show
```

Then deliberately modify something outside Terraform:

```bash
docker stop terraform-nginx
```

Run:

```bash
terraform plan
```

Now you're learning **drift**.

This is much more valuable professionally than memorizing 100 AWS resources.

---

# Project 5 — Modules

Create:

```text
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
└── modules/
    └── nginx/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

Your root configuration becomes:

```hcl
module "nginx" {
  source = "./modules/nginx"

  name = "production-nginx"
  port = 8080
}
```

Then create:

```text
module "nginx_dev"
module "nginx_staging"
module "nginx_prod"
```

This teaches you the fundamental distinction between:

```text
resource
module
provider
```

which is important when reading real Terraform repositories.

---

# Project 6 — Build a fake production environment

Now stop thinking in terms of tutorials.

Build something resembling an actual company repository:

```text
terraform/
├── README.md
├── versions.tf
├── providers.tf
├── variables.tf
├── outputs.tf
├── main.tf
├── locals.tf
├── modules/
│   ├── app/
│   ├── database/
│   └── monitoring/
└── environments/
    ├── dev/
    ├── staging/
    └── prod/
```

For example:

```text
                 Terraform
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        DEV       STAGING      PROD
          │          │           │
       nginx       nginx       nginx
       redis       redis       redis
```

Give each environment different:

```text
container count
CPU
memory
ports
names
configuration
```

Now you're starting to think like someone maintaining infrastructure rather than someone completing Terraform exercises.

---

# Project 7 — Import

This is one of the most useful exercises.

Create a Docker container manually:

```bash
docker run -d \
  --name manually-created-nginx \
  -p 8080:80 \
  nginx
```

Now Terraform doesn't know about it.

Your job:

```text
Existing infrastructure
        ↓
terraform import
        ↓
Terraform state
        ↓
write matching HCL
        ↓
terraform plan
        ↓
zero/unexpected changes
```

This teaches an important real-world workflow:

> Terraform doesn't magically discover infrastructure. You need to reconcile real infrastructure with Terraform's configuration and state.

---

# Project 8 — CI/CD

Now put the project in Git.

Create a pipeline that runs:

```bash
terraform fmt -check
terraform init
terraform validate
terraform plan
```

on every pull request.

Then imagine:

```text
Developer
   │
   ↓
Git commit
   │
   ↓
Pull Request
   │
   ↓
terraform plan
   │
   ↓
human review
   │
   ↓
merge
   │
   ↓
terraform apply
```

This is **much closer to real Terraform work**.

---

# Project 9 — Remote state

You don't need AWS to learn the concept.

First understand:

```text
Local state

terraform.tfstate
```

versus:

```text
Remote state

Terraform
   │
   ↓
Remote backend
   │
   ├── state
   └── locking
```

You should understand why teams don't generally have 15 engineers all modifying the same local `terraform.tfstate`.

Then later, when you have AWS access, learn:

```text
S3
+
DynamoDB / modern locking mechanism
```

The AWS implementation is less important than understanding the underlying problem.

---

# Project 10 — Secrets

This is another thing I'd explicitly practice.

Experiment with:

```hcl
variable "database_password" {
  type      = string
  sensitive = true
}
```

Then investigate:

```text
TF_VAR_database_password
```

and understand why:

```hcl
sensitive = true
```

**does not mean the secret isn't stored in state.**

That's a very important professional lesson.

---

# What I would NOT do

I wouldn't start with:

> "Let's learn Terraform by creating an EC2."

That tends to produce:

```text
resource "aws_instance" ...
resource "aws_security_group" ...
resource "aws_vpc" ...
```

while the learner doesn't really understand:

```text
state
graph
providers
modules
lifecycle
drift
imports
dependencies
plan
state reconciliation
```

You end up knowing AWS Terraform syntax without really knowing Terraform.

---

# The eventual AWS transition

Once you have access to an AWS account, don't restart the curriculum.

Take what you've built locally and map it to AWS:

| Local                | AWS                       |
| -------------------- | ------------------------- |
| Docker provider      | AWS provider              |
| Docker network       | VPC                       |
| container            | EC2/ECS                   |
| container networking | Security groups/subnets   |
| local state          | S3 remote state           |
| local variables      | environment configuration |
| local module         | reusable AWS module       |
| local CI pipeline    | GitHub Actions/etc.       |
| manual container     | existing AWS resource     |
| `terraform import`   | AWS resource import       |

The **Terraform concepts remain the same**.

---

# The work-ready checkpoint

I'd consider you reasonably Terraform-ready when you can receive a repository like:

```text
infrastructure/
├── modules/
├── environments/
├── providers.tf
├── variables.tf
├── outputs.tf
└── main.tf
```

and answer, without a tutorial:

1. **What infrastructure does this create?**
2. **Where is the provider configured?**
3. **Where does state live?**
4. **What happens if I change this variable?**
5. **What will `terraform plan` do?**
6. **What resources depend on this resource?**
7. **Why is this a module?**
8. **How is dev different from prod?**
9. **How would I import an existing resource?**
10. **What happens if someone changes infrastructure manually?**
11. **Where could secrets end up?**
12. **How would this be run safely in CI/CD?**

If you can answer those, learning the AWS-specific resource types becomes much more straightforward.

---

## If I were designing this specifically for you

Given your backend-engineering background, I wouldn't give you a generic "Terraform course." I'd make it a **2-week, code-first Terraform lab**:

```text
Day 1   Terraform mental model + Docker
Day 2   Resources + lifecycle
Day 3   Variables + locals + outputs
Day 4   Dependencies + expressions
Day 5   State + drift
Day 6   Import + state manipulation
Day 7   Modules

Day 8   Environment architecture
Day 9   Provider/version management
Day 10  Remote state + locking
Day 11  Secrets + sensitive data
Day 12  CI/CD + plan/apply workflow
Day 13  Debugging + common failures
Day 14  Capstone: production-style Terraform repo
```

And every day would be **80% terminal/code and 20% explanation**.

The capstone would be particularly useful: I'd have you build a realistic Terraform repository entirely locally, deliberately introduce **drift, bad state, dependency problems, module mistakes, variable mistakes, and import scenarios**, and make you diagnose them. That gets much closer to the kind of Terraform work you'll encounter on the job than following AWS tutorials.

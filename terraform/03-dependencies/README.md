# Project 3 — Multiple resources and dependencies

## Objective
Build network → {nginx, redis, postgres} and see Terraform's dependency
graph in action.

## Concepts
Implicit dependencies (via resource attribute references), `terraform graph`.

## Lecture

**Terraform builds a graph, not a script.** Your `.tf` files are not
executed top-to-bottom. On every run, Terraform parses all resources into
nodes of a directed acyclic graph (DAG), draws an edge `A → B` whenever A's
config references one of B's attributes, then walks the graph to decide
execution order. Independent branches of the graph (resources that don't
reference each other) are created **in parallel**, up to
`-parallelism=<n>` (default 10) — this is why Terraform apply can be much
faster than a hand-written shell script doing the same thing sequentially.

**Implicit dependencies: how the edge actually gets drawn.** When you write:

```hcl
networks_advanced {
  name = docker_network.app.name
}
```

Terraform doesn't know `docker_network.app.name`'s value until that network
is created — Docker assigns/confirms it at creation time. So Terraform
detects the reference syntactically at parse time, adds
`docker_container.redis → docker_network.app` as a graph edge, and defers
evaluating `docker_network.app.name` until after the network resource
finishes applying. This is the entire mechanism — there's no separate
"dependency declaration," the reference *is* the dependency.

**`depends_on`: the escape hatch, and why it's a last resort.** Some
dependencies exist in the real world but leave no trace in HCL — e.g. "this
app needs the database to have finished its init script," which no
attribute reference captures. `depends_on = [resource.x]` forces a graph
edge with no data flowing across it. Overusing it serializes work that
could have run in parallel and hides *why* the dependency exists from
anyone reading the resource block later (an attribute reference is
self-documenting; `depends_on` is not). Reach for it only when you've
confirmed no attribute reference can express the same ordering constraint.

**`terraform graph` output.** It emits the DAG in Graphviz DOT format — a
text format describing nodes and edges. Piping it through `dot` (from the
`graphviz` package) renders an actual picture; reading the raw text works
too once you know to scan for `->` lines. This is a debugging tool for
"why did Terraform touch resource X before Y" questions, more than
something you'd use day-to-day.

**Why this matters beyond Docker.** Every real Terraform module you'll read
professionally leans on implicit dependencies constantly (a subnet
referencing a VPC's ID, a security group referencing another security
group's ID, an EC2 instance referencing a subnet). Recognizing "this
attribute reference is secretly an ordering constraint" is a core reading
skill for unfamiliar repos.

## Exercise
1. `main.tf`: a `docker_network` named `app-network`.
2. Add three `docker_container` resources (nginx, redis, postgres — official
   images), each with a `networks_advanced { name = docker_network.app.name }`
   block. That reference *is* the dependency — no explicit `depends_on` needed.
3. `terraform plan` — check the order resources are created in.
4. `terraform graph | dot -Tpng > graph.png` (needs graphviz) or just read the
   raw `terraform graph` DOT output — find the edges pointing at
   `docker_network.app`.
5. Delete the network reference from one container, re-plan — see the edge
   disappear from the graph.
6. Add an *explicit* `depends_on = [docker_container.postgres]` to another
   container where there's no attribute reference, to see when you'd need
   that escape hatch (side effects Terraform can't see, e.g. an app that
   needs the DB already running, not just declared).

## Done when
You can explain the difference between an implicit dependency (attribute
reference) and `depends_on`, and why Terraform prefers you use the former.

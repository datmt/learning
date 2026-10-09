# Project 7 skeleton. See README.md — write the resource block yourself
# to match the container you create manually with `docker run`.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# TODO: docker_container "manual_nginx" — guess the config, then
#   `terraform import docker_container.manual_nginx <id>` and adjust
#   until `terraform plan` shows no diff.

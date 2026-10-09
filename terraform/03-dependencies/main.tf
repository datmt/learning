# Project 3 skeleton. See README.md.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# TODO: docker_network "app" named "app-network"

# TODO: docker_image + docker_container for nginx, redis, postgres,
#   each container with:
#     networks_advanced {
#       name = docker_network.app.name
#     }
#
# Official images to pull: nginx:latest, redis:latest, postgres:16
# (postgres requires env vars, e.g. POSTGRES_PASSWORD, via docker_container's
#  `env` argument)

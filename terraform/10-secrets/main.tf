# Project 10 skeleton. See README.md.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# TODO: docker_image "postgres" (postgres:16)

# TODO: docker_container "postgres" with:
#   env = ["POSTGRES_PASSWORD=${var.database_password}"]

# Project 2 skeleton. See README.md.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_image" "nginx" {
  name = "nginx:latest"
}

# TODO: docker_container "nginx" using var.container_name and var.external_port
#   instead of hardcoded values (copy from project 01, then swap literals)

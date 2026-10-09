# Project 1 skeleton. Fill in the TODOs, see README.md for the exercise.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# TODO: docker_image resource named "nginx", image = "nginx:latest"

# TODO: docker_container resource named "nginx"
#   - name  = "terraform-nginx"
#   - image = <reference the image resource's .image_id>
#   - a ports { internal = 80, external = 8080 } block

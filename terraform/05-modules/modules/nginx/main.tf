# Module: single nginx container. No provider block here —
# modules inherit the provider configured by whatever calls them.

resource "docker_image" "nginx" {
  name = "nginx:latest"
}

# TODO: docker_container "nginx" using var.name and var.port
#   (ports { internal = 80, external = var.port })

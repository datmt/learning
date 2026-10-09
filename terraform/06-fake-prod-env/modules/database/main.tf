# Database module: single postgres or redis container, parameterized.

# TODO: docker_image "db" (postgres:16 or redis:latest, your choice)

# TODO: docker_container "db", name = var.name, port = var.port,
#   env = ["POSTGRES_PASSWORD=..."] if using postgres

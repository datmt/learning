# App module: parameterized container(s). No provider block — inherited
# from whichever environment calls this module.

# TODO: docker_image "app" (pick any small image, e.g. nginx:latest)

# TODO: docker_container "app", use count = var.replica_count to create
#   multiple instances; name = "${var.name}-${count.index}" so names don't
#   collide; port = var.base_port + count.index so ports don't collide.

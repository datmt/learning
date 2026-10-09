# Root module. See README.md.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

module "nginx" {
  source = "./modules/nginx"
  name   = "production-nginx"
  port   = 8080
}

# TODO: module "nginx_dev" (source = "./modules/nginx", different name/port)
# TODO: module "nginx_staging" (same, different name/port)

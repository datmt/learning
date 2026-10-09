# dev environment — its own state, its own init/plan/apply.
# Run all commands from inside this directory.

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

module "app" {
  source        = "../../modules/app"
  name          = "dev-app"
  base_port     = 9000
  replica_count = 1
}

module "database" {
  source = "../../modules/database"
  name   = "dev-db"
  port   = 9432
}

# staging environment — its own state, its own init/plan/apply.
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
  name          = "staging-app"
  base_port     = 9100
  replica_count = 2
}

module "database" {
  source = "../../modules/database"
  name   = "staging-db"
  port   = 9433
}

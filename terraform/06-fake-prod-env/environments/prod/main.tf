# prod environment — its own state, its own init/plan/apply.
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
  name          = "prod-app"
  base_port     = 9200
  replica_count = 3
}

module "database" {
  source = "../../modules/database"
  name   = "prod-db"
  port   = 9434
}

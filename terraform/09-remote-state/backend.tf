# Reference only — not active. Local state (the default, no backend block)
# is what every other project in this repo uses.
#
# What you'd add once you have an AWS account (Project 9 in AGENDA.md):
#
# terraform {
#   backend "s3" {
#     bucket       = "your-terraform-state-bucket"
#     key          = "09-remote-state/terraform.tfstate"
#     region       = "us-east-1"
#     use_lockfile = true # native S3 locking (newer provider versions)
#     # dynamodb_table = "terraform-locks" # older locking mechanism
#   }
# }
#
# Notes:
# - `key` is the path *within* the bucket — this is how multiple projects
#   share one bucket without colliding.
# - After adding this and running `terraform init`, Terraform asks to
#   migrate your existing local state into the bucket. Say yes.

terraform {
  backend "s3" {
    bucket = "tfstate-bucket-dev-1"
    key = "envs/dev/terraform.tfstate"
    region = "us-east-1"
    dynamodb_table = "tfstate-table-dev-1"
  }
}
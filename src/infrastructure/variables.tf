variable "instance_type" { 
    description = "instance type for ec2" 
    default     =  "t2.nano"
}

variable "aws_region" {
       description = "The AWS region to create things in." 
       default     = "us-east-1" 
}

variable "ami_id" { 
    description = "AMI for AMazon Linux 2023 Ec2 instance" 
    default     = "ami-0fa3fe0fa7920f68e" 
}

variable "vpc_cidr" {
    description = "CIDR block for the VPC"
    default     = "10.0.0.0/16"
}

variable "asg_min" {
    description = "Minimum size for autoscaling group"
    default     = 1
}

variable "asg_max" {
    description = "Maximum size for autoscaling group"
    default     = 2
}

variable "key_name" {
    description = "(optional) key pair name for SSH access to instances"
    default     = "key_dev"
}

variable "acm_certificate_arn" {
    description = "Optional ACM certificate ARN for ALB HTTPS listener. If empty, ALB will use HTTP only."
    default     = ""
}

variable "user_data" {
    description = "Optional user data for instances"
    default     = ""
}

variable "whitelist_cidrs" {
    description = "List of CIDR blocks allowed to access the ALB (whitelist). Empty list = no external access."
    type        = list(string)
    default     = ["0.0.0.0/0"]
}

variable "secret_name" {
    description = "Name or ARN of the Secrets Manager secret that contains git credentials"
    default     = "dev/git"
}
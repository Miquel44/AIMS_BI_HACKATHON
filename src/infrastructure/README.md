📁 Infrastructure Module and CI/CD Pipeline

This folder contains all the necessary artifacts for the automated deployment and management of our solution in the Amazon Web Services (AWS) environment.

Its main content is divided into two fundamental sections: the definition of Infrastructure as Code (IaC) and the configuration of the Continuous Integration and Continuous Deployment (CI/CD) workflow.

🏗️ Deployed Architecture

The following image illustrates the design of the infrastructure that this Terraform module provisions in AWS, including the Virtual Private Cloud (VPC), Auto Scaling Groups (ASG), Application Load Balancers (ALB), and other related services.

⚙️ Terraform Module

The heart of this folder is a complete Terraform module.

This module is designed to be reusable and deploys the full end-to-end infrastructure required for our application to function with high availability.

Key Features:

Complete Infrastructure: Provisions the network (VPC, public/private subnets), security (Security Groups), IAM profiles, and the compute layer (Launch Templates and Auto Scaling Groups).

Load Management: Includes the configuration of an Application Load Balancer (ALB) to efficiently distribute traffic to the application instances.

Instance Configuration: The launch template configures the EC2 instances for proper initialization, including dependency installation, Git configuration (for code cloning), and the deployment of the initial web server (Nginx).

🚀 Continuous Integration and Continuous Deployment Pipeline (CI/CD)

In addition to the infrastructure code, this folder defines the automation workflow through a Jenkinsfile.

This CI/CD pipeline ensures that any changes merged into the repository's main branch are automatically translated into an update of the deployed infrastructure, achieving an agile and reliable development cycle.

Pipeline Flow:

Activation: The pipeline is automatically triggered by any push or merge to the main branch.

Validation: terraform fmt and terraform validate commands are executed to ensure code syntax and integrity.

Planning: A terraform plan is generated to visualize exactly which changes will be applied to the AWS infrastructure.

Deployment (Apply): If the planning is successful, terraform apply is executed to update the deployed resources, guaranteeing consistency between the code in main and the cloud state.

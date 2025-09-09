# Variables for MongoDB Query Translator MLOps Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "mongodb-query-translator"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 3
}

variable "rds_instance_class" {
  description = "RDS instance class for MLflow database"
  type        = string
  default     = "db.t3.micro"
}

variable "mlflow_db_password" {
  description = "Password for MLflow database"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "ElastiCache node type for Redis"
  type        = string
  default     = "cache.t3.micro"
}

# MongoDB Atlas Configuration
variable "mongodb_atlas_project_id" {
  description = "MongoDB Atlas project ID"
  type        = string
  default     = ""
}

variable "mongodb_atlas_public_key" {
  description = "MongoDB Atlas public key"
  type        = string
  default     = ""
}

variable "mongodb_atlas_private_key" {
  description = "MongoDB Atlas private key"
  type        = string
  sensitive   = true
  default     = ""
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable comprehensive monitoring stack"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Number of days to retain logs"
  type        = number
  default     = 30
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "auto_scaling_enabled" {
  description = "Enable auto-scaling for EKS nodes"
  type        = bool
  default     = true
}

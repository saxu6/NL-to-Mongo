# MongoDB Query Translator - MLOps Infrastructure
# Terraform configuration for production-grade deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "mongodb-query-translator-terraform-state"
    key    = "mlops/terraform.tfstate"
    region = "us-west-2"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "mongodb-query-translator"
      Environment = var.environment
      ManagedBy   = "terraform"
      MLOps       = "true"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    mlops = {
      name = "mlops-nodes"

      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"

      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size

      disk_size = 50

      labels = {
        role = "mlops"
      }

      taints = [
        {
          key    = "mlops"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.mlops_role.arn
      username = "mlops-user"
      groups   = ["system:masters"]
    }
  ]

  tags = local.common_tags
}

# IAM Role for MLOps
resource "aws_iam_role" "mlops_role" {
  name = "${local.cluster_name}-mlops-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for MLOps
resource "aws_iam_role_policy" "mlops_policy" {
  name = "${local.cluster_name}-mlops-policy"
  role = aws_iam_role.mlops_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlops_bucket.arn,
          "${aws_s3_bucket.mlops_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = aws_secretsmanager_secret.mongodb_uri.arn
      }
    ]
  })
}

# S3 Bucket for MLOps artifacts
resource "aws_s3_bucket" "mlops_bucket" {
  bucket = "${local.cluster_name}-mlops-artifacts"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "mlops_bucket_versioning" {
  bucket = aws_s3_bucket.mlops_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_bucket_encryption" {
  bucket = aws_s3_bucket.mlops_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Secrets Manager for sensitive data
resource "aws_secretsmanager_secret" "mongodb_uri" {
  name        = "${local.cluster_name}-mongodb-uri"
  description = "MongoDB connection URI for MLOps pipeline"

  tags = local.common_tags
}

# RDS for MLflow tracking database
resource "aws_db_instance" "mlflow_db" {
  identifier = "${local.cluster_name}-mlflow-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.rds_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "mlflow"
  username = "mlflow"
  password = var.mlflow_db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"

  tags = local.common_tags
}

# RDS Subnet Group
resource "aws_db_subnet_group" "mlflow" {
  name       = "${local.cluster_name}-mlflow-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = local.common_tags
}

# Security Group for RDS
resource "aws_security_group" "rds_sg" {
  name_prefix = "${local.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
  })
}

# ElastiCache for Redis (for Airflow)
resource "aws_elasticache_subnet_group" "airflow" {
  name       = "${local.cluster_name}-airflow-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "airflow" {
  replication_group_id       = "${local.cluster_name}-airflow-cache"
  description                = "Redis cluster for Airflow"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = 2

  subnet_group_name  = aws_elasticache_subnet_group.airflow.name
  security_group_ids = [aws_security_group.redis_sg.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = local.common_tags
}

# Security Group for Redis
resource "aws_security_group" "redis_sg" {
  name_prefix = "${local.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-sg"
  })
}

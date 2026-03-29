#!/bin/bash
# ─── Deploy EV Charging Optimization System to AWS EC2 ───
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - An EC2 key pair created (e.g., ev-charging-key)
#   - Security group allowing ports 22, 8000, 8501
#
# Usage: ./scripts/deploy_aws.sh

set -euo pipefail

# ── Configuration ──
INSTANCE_TYPE="t3.small"           # 2 vCPU, 2 GB RAM (~$15/month)
AMI_ID="ami-0c02fb55956c7d316"     # Amazon Linux 2023 (us-east-1, update for your region)
KEY_NAME="ev-charging-key"         # Your EC2 key pair name
SECURITY_GROUP="ev-charging-sg"    # Will be created if needed
REGION="us-east-1"

echo "=== EV Charging System — AWS Deployment ==="

# ── Step 1: Create Security Group ──
echo "[1/5] Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SECURITY_GROUP" \
    --description "EV Charging Optimization System" \
    --region "$REGION" \
    --query 'GroupId' --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --group-names "$SECURITY_GROUP" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text)

# Allow SSH, API, Dashboard, Prometheus, Grafana
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 9090 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3000 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 5000 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true

echo "  Security group: $SG_ID"

# ── Step 2: Launch EC2 Instance ──
echo "[2/5] Launching EC2 instance ($INSTANCE_TYPE)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ev-charging-system}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance ID: $INSTANCE_ID"
echo "  Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "  Public IP: $PUBLIC_IP"

# ── Step 3: Wait for SSH ──
echo "[3/5] Waiting for SSH to be ready..."
sleep 30  # Give the instance time to fully boot

# ── Step 4: Install Docker + Deploy ──
echo "[4/5] Installing Docker and deploying..."
ssh -o StrictHostKeyChecking=no -i "${KEY_NAME}.pem" ec2-user@"$PUBLIC_IP" << 'REMOTE_SCRIPT'
    # Install Docker
    sudo yum update -y
    sudo yum install -y docker git
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user

    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose

    # Clone and deploy (replace with your repo URL)
    # git clone https://github.com/YOUR_USERNAME/ev-charging-optimization.git
    # cd ev-charging-optimization

    echo "Docker installed. Ready for deployment."
    docker --version
    docker-compose --version
REMOTE_SCRIPT

# ── Step 5: Print Summary ──
echo ""
echo "=== Deployment Ready ==="
echo "  Instance: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo ""
echo "  Next steps:"
echo "  1. Copy your project to the server:"
echo "     scp -i ${KEY_NAME}.pem -r ./ ec2-user@${PUBLIC_IP}:~/ev-charging/"
echo ""
echo "  2. SSH into the server and start:"
echo "     ssh -i ${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo "     cd ~/ev-charging && docker-compose up -d --build"
echo ""
echo "  3. Access your system:"
echo "     API:        http://${PUBLIC_IP}:8000"
echo "     Dashboard:  http://${PUBLIC_IP}:8501"
echo "     API Docs:   http://${PUBLIC_IP}:8000/docs"
echo "     Prometheus: http://${PUBLIC_IP}:9090"
echo "     Grafana:    http://${PUBLIC_IP}:3000  (admin / evcharging)"

#!/bin/bash
# ─── Deploy EV Charging Optimization to Kubernetes ───
# Prerequisites:
#   - kubectl configured for your cluster
#   - Docker images built and pushed to a registry (or loaded locally for minikube)
#
# Usage:
#   ./scripts/deploy_k8s.sh [build|apply|delete|status]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_DIR/k8s"

# Container registry — set to your registry (e.g., ECR, Docker Hub)
REGISTRY="${DOCKER_REGISTRY:-}"
TAG="${IMAGE_TAG:-latest}"

usage() {
    echo "Usage: $0 [build|apply|delete|status]"
    echo "  build   — Build and tag Docker images"
    echo "  apply   — Deploy all manifests to Kubernetes"
    echo "  delete  — Remove all resources from the cluster"
    echo "  status  — Show pod and service status"
    exit 1
}

build_images() {
    echo "Building Docker images..."
    docker build -t proj-test-gateway:$TAG "$PROJECT_DIR/gateway"
    docker build -t proj-test-ml-service:$TAG "$PROJECT_DIR/ml-service"
    docker build -t proj-test-optimizer:$TAG "$PROJECT_DIR/optimizer"
    docker build -t proj-test-monitoring:$TAG "$PROJECT_DIR/monitoring"

    if [ -n "$REGISTRY" ]; then
        echo "Tagging and pushing to $REGISTRY..."
        for svc in gateway ml-service optimizer monitoring; do
            docker tag "proj-test-$svc:$TAG" "$REGISTRY/proj-test-$svc:$TAG"
            docker push "$REGISTRY/proj-test-$svc:$TAG"
        done
        echo "Images pushed to $REGISTRY"
    else
        echo "No DOCKER_REGISTRY set — images built locally only."
        echo "For minikube: eval \$(minikube docker-env) before building."
    fi
}

apply_manifests() {
    echo "Applying Kubernetes manifests..."
    kubectl apply -k "$K8S_DIR"

    echo ""
    echo "Waiting for pods to be ready..."
    kubectl -n ev-charging wait --for=condition=ready pod -l app=ml-service --timeout=120s 2>/dev/null || true
    kubectl -n ev-charging wait --for=condition=ready pod -l app=optimizer --timeout=120s 2>/dev/null || true
    kubectl -n ev-charging wait --for=condition=ready pod -l app=gateway --timeout=120s 2>/dev/null || true
    kubectl -n ev-charging wait --for=condition=ready pod -l app=monitoring --timeout=120s 2>/dev/null || true

    echo ""
    show_status
}

delete_all() {
    echo "Deleting all EV Charging resources..."
    kubectl delete -k "$K8S_DIR" --ignore-not-found
    echo "All resources deleted."
}

show_status() {
    echo "=== Pods ==="
    kubectl -n ev-charging get pods -o wide
    echo ""
    echo "=== Services ==="
    kubectl -n ev-charging get services
    echo ""
    echo "=== Access URLs (NodePort) ==="
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "localhost")
    echo "  API Gateway:  http://$NODE_IP:30800"
    echo "  Dashboard:    http://$NODE_IP:30851"
    echo "  Prometheus:   http://$NODE_IP:30909"
    echo "  Grafana:      http://$NODE_IP:30300"
    echo "  MLflow:       http://$NODE_IP:30500"
}

case "${1:-}" in
    build)  build_images ;;
    apply)  apply_manifests ;;
    delete) delete_all ;;
    status) show_status ;;
    *)      usage ;;
esac

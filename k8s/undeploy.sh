#!/bin/bash

# AlphaBeta808 Trading Bot - Kubernetes Undeployment Script
# Ce script supprime toute l'infrastructure Kubernetes

set -e

NAMESPACE="alphabeta808-trading"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🗑️  Suppression du déploiement AlphaBeta808 Trading Bot"
echo "====================================================="

# Demander confirmation
read -p "Êtes-vous sûr de vouloir supprimer tout le déploiement? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Annulation de la suppression"
    exit 0
fi

# Supprimer les ressources dans l'ordre inverse
log_info "Suppression des ressources Kubernetes..."

# Supprimer l'Ingress en premier
kubectl delete -f k8s/ingress.yaml --ignore-not-found=true

# Supprimer les autres ressources
kubectl delete -f k8s/hpa.yaml --ignore-not-found=true
kubectl delete -f k8s/pdb.yaml --ignore-not-found=true
kubectl delete -f k8s/services.yaml --ignore-not-found=true
kubectl delete -f k8s/web-deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/bot-deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/rbac.yaml --ignore-not-found=true

# Demander si on supprime les PVC (données persistantes)
echo ""
read -p "Supprimer également les données persistantes (PVC)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete -f k8s/pvc.yaml --ignore-not-found=true
    log_warning "Données persistantes supprimées"
else
    log_info "Conservation des données persistantes"
fi

# Supprimer les secrets et configmaps
kubectl delete -f k8s/secrets.yaml --ignore-not-found=true
kubectl delete -f k8s/configmap.yaml --ignore-not-found=true

# Supprimer le namespace
read -p "Supprimer le namespace complet? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    log_success "Namespace supprimé"
else
    log_info "Conservation du namespace"
fi

log_success "🎉 Suppression terminée!"

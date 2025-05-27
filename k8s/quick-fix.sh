#!/bin/bash

# Quick fix for current AlphaBeta808 deployment issues

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

echo "🔧 Correction rapide du déploiement AlphaBeta808"
echo "==============================================="

# 1. Supprimer les ressources problématiques du monitoring actuel
log_info "Suppression des ressources monitoring problématiques..."

kubectl delete servicemonitor trading-bot-monitor -n ${NAMESPACE} --ignore-not-found=true
kubectl delete prometheusrule trading-bot-alerts -n ${NAMESPACE} --ignore-not-found=true

log_success "Ressources problématiques supprimées"

# 2. Déployer le monitoring basique
log_info "Déploiement du monitoring basique..."

kubectl apply -f k8s/monitoring-basic.yaml

if [ $? -eq 0 ]; then
    log_success "Monitoring basique déployé"
else
    log_warning "Problème avec le monitoring basique"
fi

# 3. Vérifier le statut des pods
log_info "Vérification du statut des pods..."

kubectl get pods -n ${NAMESPACE}

# 4. Attendre que les nouveaux pods soient prêts
log_info "Attente que les pods de monitoring soient prêts..."

kubectl wait --for=condition=ready pod \
    --selector=app=prometheus \
    --namespace=${NAMESPACE} \
    --timeout=120s || log_warning "Timeout pour Prometheus"

kubectl wait --for=condition=ready pod \
    --selector=app=grafana \
    --namespace=${NAMESPACE} \
    --timeout=120s || log_warning "Timeout pour Grafana"

# 5. Afficher le statut final
echo ""
log_info "Statut final du déploiement:"
echo "============================="

kubectl get all -n ${NAMESPACE}

echo ""
log_info "Services de monitoring:"
kubectl get svc -n ${NAMESPACE} | grep -E "(prometheus|grafana)"

echo ""
log_success "🎉 Correction terminée!"
echo ""
echo "Accès aux services:"
echo "📊 Prometheus: kubectl port-forward svc/prometheus-service 9090:9090 -n ${NAMESPACE}"
echo "📈 Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n ${NAMESPACE}"
echo "🌐 Interface Web: kubectl port-forward svc/trading-web-service 8080:5000 -n ${NAMESPACE}"
echo ""
echo "Credentials Grafana: admin / admin123"

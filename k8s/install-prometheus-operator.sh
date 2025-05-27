#!/bin/bash

# Script to install Prometheus Operator for AlphaBeta808 Trading Bot

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

# Vérifier si Helm est installé
check_helm() {
    if ! command -v helm &> /dev/null; then
        log_error "Helm n'est pas installé. Veuillez l'installer:"
        echo "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
        exit 1
    fi
    log_success "Helm trouvé"
}

# Installer Prometheus Operator via Helm
install_prometheus_operator() {
    log_info "Installation de Prometheus Operator..."
    
    # Ajouter le repository Helm
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Installer Prometheus Operator
    helm install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.service.type=ClusterIP \
        --set grafana.service.type=ClusterIP \
        --set alertmanager.service.type=ClusterIP
    
    log_success "Prometheus Operator installé"
    
    # Attendre que les CRDs soient prêts
    log_info "Attente que les CRDs soient prêts..."
    sleep 30
    
    # Vérifier les CRDs
    if kubectl get crd servicemonitors.monitoring.coreos.com &>/dev/null; then
        log_success "ServiceMonitor CRD disponible"
    else
        log_error "ServiceMonitor CRD non disponible"
        exit 1
    fi
    
    if kubectl get crd prometheusrules.monitoring.coreos.com &>/dev/null; then
        log_success "PrometheusRule CRD disponible"
    else
        log_error "PrometheusRule CRD non disponible"
        exit 1
    fi
}

# Installer via manifestes YAML (alternative à Helm)
install_via_manifests() {
    log_info "Installation via manifestes YAML..."
    
    # Télécharger et appliquer les manifestes Prometheus Operator
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
    
    log_info "Attente que l'operateur soit prêt..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus-operator -n default --timeout=300s
    
    log_success "Prometheus Operator installé via manifestes"
}

# Fonction principale
main() {
    echo "🔧 Installation de Prometheus Operator pour AlphaBeta808"
    echo "======================================================="
    
    # Choisir la méthode d'installation
    echo "Choisissez la méthode d'installation:"
    echo "1) Helm (recommandé)"
    echo "2) Manifestes YAML"
    echo "3) Annuler"
    
    read -p "Votre choix (1-3): " choice
    
    case $choice in
        1)
            check_helm
            install_prometheus_operator
            ;;
        2)
            install_via_manifests
            ;;
        3)
            log_info "Installation annulée"
            exit 0
            ;;
        *)
            log_error "Choix invalide"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "✅ Prometheus Operator installé avec succès!"
    echo ""
    echo "Vous pouvez maintenant déployer le monitoring complet avec:"
    echo "kubectl apply -f k8s/monitoring.yaml"
    echo ""
    echo "Ou redéployer complètement avec:"
    echo "./k8s/deploy-fixed.sh"
}

main "$@"

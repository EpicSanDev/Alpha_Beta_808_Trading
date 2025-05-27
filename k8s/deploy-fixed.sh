#!/bin/bash

# AlphaBeta808 Trading Bot - Fixed Kubernetes Deployment Script
# Ce script déploie l'ensemble de l'infrastructure sur Kubernetes avec monitoring fonctionnel

set -e

echo "🚀 Déploiement AlphaBeta808 Trading Bot sur Kubernetes (Version Corrigée)"
echo "========================================================================"

# Variables de configuration
NAMESPACE="alphabeta808-trading"
IMAGE_NAME="alphabeta808/trading-bot"
VERSION="${1:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DEPLOY_MONITORING="${DEPLOY_MONITORING:-true}"

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

# Vérifier si Prometheus Operator CRDs sont installés
check_prometheus_operator() {
    log_info "Vérification de la disponibilité de Prometheus Operator..."
    
    if kubectl get crd servicemonitors.monitoring.coreos.com &>/dev/null && kubectl get crd prometheusrules.monitoring.coreos.com &>/dev/null; then
        log_success "Prometheus Operator CRDs détectés"
        return 0
    else
        log_warning "Prometheus Operator CRDs non trouvés"
        return 1
    fi
}

# Déployer le monitoring approprié
deploy_monitoring() {
    if [ "$DEPLOY_MONITORING" = "true" ]; then
        log_info "Configuration du monitoring..."
        
        if check_prometheus_operator; then
            log_info "Utilisation de Prometheus Operator (monitoring.yaml)"
            kubectl apply -f k8s/monitoring.yaml || {
                log_error "Échec du déploiement avec Prometheus Operator, fallback vers monitoring basique"
                kubectl apply -f k8s/monitoring-basic.yaml
            }
        else
            log_info "Utilisation du monitoring basique (sans Prometheus Operator)"
            kubectl apply -f k8s/monitoring-basic.yaml
        fi
        
        log_success "Monitoring déployé"
    else
        log_info "Monitoring désactivé (DEPLOY_MONITORING!=true)"
    fi
}

# Déployer les ressources principales avec gestion d'erreurs améliorée
deploy_main_resources() {
    log_info "Déploiement des ressources principales..."
    
    # Ordre de déploiement avec gestion d'erreurs
    local resources=(
        "k8s/namespace.yaml:Namespace"
        "k8s/configmap.yaml:ConfigMaps"
        "k8s/pvc.yaml:PersistentVolumeClaims"
        "k8s/rbac.yaml:RBAC"
        "k8s/secrets.yaml:Secrets"
        "k8s/bot-deployment.yaml:Bot Deployment"
        "k8s/web-deployment.yaml:Web Deployment" 
        "k8s/services.yaml:Services"
        "k8s/pdb.yaml:Pod Disruption Budgets"
        "k8s/hpa.yaml:Horizontal Pod Autoscaler"
        "k8s/nginx-proxy.yaml:Nginx Proxy"
    )
    
    for resource_info in "${resources[@]}"; do
        IFS=':' read -r resource_file resource_name <<< "$resource_info"
        
        if [ -f "$resource_file" ]; then
            log_info "Déploiement de $resource_name ($resource_file)..."
            if kubectl apply -f "$resource_file"; then
                log_success "$resource_name déployé avec succès"
            else
                log_warning "Problème avec $resource_name, mais on continue..."
            fi
        else
            log_warning "Fichier $resource_file non trouvé, ignoré"
        fi
    done
}

# Déployer l'ingress (optionnel)
deploy_ingress() {
    if [ "$DEPLOY_INGRESS" = "true" ] && [ -f "k8s/ingress.yaml" ]; then
        log_info "Déploiement de l'Ingress..."
        if kubectl apply -f k8s/ingress.yaml; then
            log_success "Ingress déployé"
        else
            log_warning "Problème avec l'Ingress, mais on continue..."
        fi
    else
        log_info "Ingress non déployé"
    fi
}

# Attendre que les pods soient prêts avec timeout
wait_for_pods() {
    log_info "Attente que les pods soient prêts..."
    
    # Attendre les pods principaux
    local timeout=300
    
    log_info "Attente du pod trading-web-interface..."
    if kubectl wait --for=condition=ready pod \
        --selector=app=trading-web-interface \
        --namespace=${NAMESPACE} \
        --timeout=${timeout}s; then
        log_success "Trading web interface prêt"
    else
        log_warning "Timeout pour trading-web-interface"
    fi
    
    log_info "Attente du pod trading-bot..."
    if kubectl wait --for=condition=ready pod \
        --selector=app=trading-bot \
        --namespace=${NAMESPACE} \
        --timeout=${timeout}s; then
        log_success "Trading bot prêt"
    else
        log_warning "Timeout pour trading-bot"
    fi
}

# Afficher le statut du déploiement
show_status() {
    echo ""
    log_info "Statut du déploiement:"
    echo "========================"
    
    kubectl get all -n ${NAMESPACE}
    
    echo ""
    log_info "Services exposés:"
    kubectl get svc -n ${NAMESPACE}
    
    echo ""
    log_info "URLs d'accès:"
    
    # Interface web
    EXTERNAL_IP=$(kubectl get svc trading-web-loadbalancer -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -n "$EXTERNAL_IP" ]; then
        echo "🌐 Interface Web: http://${EXTERNAL_IP}"
    else
        echo "🌐 Interface Web (port-forward): kubectl port-forward svc/trading-web-service 8080:5000 -n ${NAMESPACE}"
        echo "   Puis ouvrez: http://localhost:8080"
    fi
    
    # Monitoring (si déployé)
    if [ "$DEPLOY_MONITORING" = "true" ]; then
        echo "📊 Prometheus: kubectl port-forward svc/prometheus-service 9090:9090 -n ${NAMESPACE}"
        echo "   Puis ouvrez: http://localhost:9090"
        echo "📈 Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n ${NAMESPACE}"
        echo "   Puis ouvrez: http://localhost:3000 (admin/admin123)"
    fi
}

# Fonction de nettoyage en cas d'erreur
cleanup_on_error() {
    log_error "Erreur détectée, nettoyage..."
    # Optionnel: supprimer les ressources partiellement déployées
    # kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
}

# Fonction principale avec gestion d'erreurs
main() {
    trap cleanup_on_error ERR
    
    echo "Début du déploiement avec les paramètres:"
    echo "- Namespace: ${NAMESPACE}"
    echo "- Image: ${IMAGE_NAME}:${VERSION}"
    echo "- Registry: ${DOCKER_REGISTRY}"
    echo "- Monitoring: ${DEPLOY_MONITORING}"
    echo ""
    
    deploy_main_resources
    deploy_monitoring
    deploy_ingress
    wait_for_pods
    show_status
    
    echo ""
    log_success "🎉 Déploiement terminé!"
    echo ""
    echo "Commandes utiles:"
    echo "- Logs bot: kubectl logs -f deployment/trading-bot -n ${NAMESPACE}"
    echo "- Logs web: kubectl logs -f deployment/trading-web-interface -n ${NAMESPACE}"
    echo "- Accéder au pod: kubectl exec -it deployment/trading-bot -n ${NAMESPACE} -- /bin/bash"
    echo "- Statut: kubectl get all -n ${NAMESPACE}"
}

# Gestion des options
while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy-ingress)
            DEPLOY_INGRESS=true
            shift
            ;;
        --no-monitoring)
            DEPLOY_MONITORING=false
            shift
            ;;
        --help)
            echo "Usage: $0 [VERSION] [OPTIONS]"
            echo "Options:"
            echo "  --deploy-ingress Deploy Ingress resource"
            echo "  --no-monitoring  Skip monitoring deployment"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            VERSION=$1
            shift
            ;;
    esac
done

# Exécuter le script principal
main

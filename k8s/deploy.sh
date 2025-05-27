#!/bin/bash

# AlphaBeta808 Trading Bot - Kubernetes Deployment Script
# Ce script déploie l'ensemble de l'infrastructure sur Kubernetes

set -e

echo "🚀 Déploiement AlphaBeta808 Trading Bot sur Kubernetes"
echo "=================================================="

# Variables de configuration
NAMESPACE="alphabeta808-trading"
IMAGE_NAME="alphabeta808/trading-bot"
VERSION="${1:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"

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

# Vérifier que kubectl est installé
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé. Veuillez l'installer avant de continuer."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé. Veuillez l'installer avant de continuer."
        exit 1
    fi
    
    log_success "Prérequis vérifiés"
}

# Construire l'image Docker
build_image() {
    log_info "Construction de l'image Docker..."
    
    docker build -t ${IMAGE_NAME}:${VERSION} .
    
    if [ $? -eq 0 ]; then
        log_success "Image Docker construite: ${IMAGE_NAME}:${VERSION}"
    else
        log_error "Échec de la construction de l'image Docker"
        exit 1
    fi
}

# Pousser l'image vers le registre
push_image() {
    if [ "$SKIP_PUSH" != "true" ]; then
        log_info "Push de l'image vers le registre..."
        
        docker tag ${IMAGE_NAME}:${VERSION} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
        
        if [ $? -eq 0 ]; then
            log_success "Image poussée vers le registre"
        else
            log_warning "Échec du push. Continuons avec l'image locale..."
        fi
    else
        log_info "Skip du push d'image (SKIP_PUSH=true)"
    fi
}

# Créer le namespace
create_namespace() {
    log_info "Création du namespace..."
    
    kubectl apply -f k8s/namespace.yaml
    
    if [ $? -eq 0 ]; then
        log_success "Namespace créé/mis à jour"
    else
        log_error "Échec de la création du namespace"
        exit 1
    fi
}

# Configurer les secrets
setup_secrets() {
    log_info "Configuration des secrets..."
    
    # Vérifier si le fichier .env existe
    if [ -f ".env" ]; then
        log_info "Fichier .env trouvé, création des secrets depuis le fichier..."
        
        # Lire les variables depuis .env
        source .env
        
        # Encoder en base64 et créer le secret
        kubectl create secret generic trading-secrets \
            --namespace=${NAMESPACE} \
            --from-literal=binance-api-key="${BINANCE_API_KEY}" \
            --from-literal=binance-api-secret="${BINANCE_API_SECRET}" \
            --from-literal=webhook-secret="${WEBHOOK_SECRET:-default_webhook_secret}" \
            --from-literal=web-admin-user="${WEB_ADMIN_USER:-admin}" \
            --from-literal=web-admin-password="${WEB_ADMIN_PASSWORD:-secure_password_123}" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        log_success "Secrets créés depuis le fichier .env"
    else
        log_warning "Fichier .env non trouvé, utilisation des secrets par défaut"
        kubectl apply -f k8s/secrets.yaml
    fi
}

# Déployer les ressources Kubernetes
deploy_resources() {
    log_info "Déploiement des ressources Kubernetes..."
    
    # Ordre de déploiement important
    local resources=(
        "k8s/configmap.yaml"
        "k8s/pvc.yaml"
        "k8s/rbac.yaml"
        "k8s/bot-deployment.yaml"
        "k8s/web-deployment.yaml"
        "k8s/services.yaml"
        "k8s/pdb.yaml"
        "k8s/hpa.yaml"
    )
    
    for resource in "${resources[@]}"; do
        log_info "Déploiement de $resource..."
        kubectl apply -f $resource
        
        if [ $? -ne 0 ]; then
            log_error "Échec du déploiement de $resource"
            exit 1
        fi
    done
    
    log_success "Toutes les ressources ont été déployées"
}

# Déployer l'ingress (optionnel)
deploy_ingress() {
    if [ "$DEPLOY_INGRESS" = "true" ]; then
        log_info "Déploiement de l'Ingress..."
        kubectl apply -f k8s/ingress.yaml
        log_success "Ingress déployé"
    else
        log_info "Ingress non déployé (DEPLOY_INGRESS!=true)"
    fi
}

# Attendre que les pods soient prêts
wait_for_pods() {
    log_info "Attente que les pods soient prêts..."
    
    kubectl wait --for=condition=ready pod \
        --selector=app=trading-web-interface \
        --namespace=${NAMESPACE} \
        --timeout=300s
    
    kubectl wait --for=condition=ready pod \
        --selector=app=trading-bot \
        --namespace=${NAMESPACE} \
        --timeout=300s
    
    log_success "Tous les pods sont prêts"
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
    
    # Obtenir l'IP du LoadBalancer ou NodePort
    EXTERNAL_IP=$(kubectl get svc trading-web-loadbalancer -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    
    if [ -n "$EXTERNAL_IP" ]; then
        echo "🌐 Interface Web: http://${EXTERNAL_IP}"
    else
        log_info "Pour accéder à l'interface web, utilisez:"
        echo "kubectl port-forward svc/trading-web-service 8080:5000 -n ${NAMESPACE}"
        echo "Puis ouvrez: http://localhost:8080"
    fi
}

# Fonction principale
main() {
    echo "Début du déploiement avec les paramètres:"
    echo "- Namespace: ${NAMESPACE}"
    echo "- Image: ${IMAGE_NAME}:${VERSION}"
    echo "- Registry: ${DOCKER_REGISTRY}"
    echo ""
    
    check_prerequisites
    build_image
    push_image
    create_namespace
    setup_secrets
    deploy_resources
    deploy_ingress
    wait_for_pods
    show_status
    
    echo ""
    log_success "🎉 Déploiement terminé avec succès!"
    echo ""
    echo "Commandes utiles:"
    echo "- Voir les logs: kubectl logs -f deployment/trading-bot -n ${NAMESPACE}"
    echo "- Voir l'interface web: kubectl logs -f deployment/trading-web-interface -n ${NAMESPACE}"
    echo "- Accéder au pod: kubectl exec -it deployment/trading-bot -n ${NAMESPACE} -- /bin/bash"
    echo "- Supprimer le déploiement: ./scripts/undeploy.sh"
}

# Gestion des options
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --deploy-ingress)
            DEPLOY_INGRESS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [VERSION] [OPTIONS]"
            echo "Options:"
            echo "  --skip-build     Skip Docker image build"
            echo "  --skip-push      Skip Docker image push"
            echo "  --deploy-ingress Deploy Ingress resource"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            VERSION=$1
            shift
            ;;
    esac
done

# Si --skip-build est défini, ne pas construire l'image
if [ "$SKIP_BUILD" = "true" ]; then
    build_image() {
        log_info "Skip de la construction d'image (SKIP_BUILD=true)"
    }
fi

# Exécuter le script principal
main

#!/bin/bash

# 🚀 AlphaBeta808 Trading Bot - CI/CD Setup Helper
# Ce script aide à configurer les environnements et vérifier la configuration

set -e

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

# Fonction d'aide
show_help() {
    echo "🚀 AlphaBeta808 Trading Bot - CI/CD Setup Helper"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check-secrets     Vérifier les secrets GitHub"
    echo "  check-k8s         Vérifier la configuration Kubernetes"
    echo "  setup-namespaces  Créer les namespaces Kubernetes"
    echo "  setup-secrets     Créer les secrets Kubernetes"
    echo "  test-registry     Tester l'accès au Container Registry"
    echo "  test-pipeline     Déclencher un test du pipeline"
    echo "  status           Afficher le statut général"
    echo "  help             Afficher cette aide"
    echo ""
    echo "Examples:"
    echo "  $0 check-secrets"
    echo "  $0 setup-namespaces"
    echo "  $0 status"
}

# Vérifier les secrets GitHub
check_secrets() {
    log_info "Vérification des secrets GitHub..."
    
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) n'est pas installé. Installez-le pour utiliser cette fonction."
        log_info "Installation: brew install gh"
        return 1
    fi
    
    # Vérifier l'authentification
    if ! gh auth status &> /dev/null; then
        log_error "Vous n'êtes pas authentifié avec GitHub CLI."
        log_info "Exécutez: gh auth login"
        return 1
    fi
    
    echo ""
    log_info "Secrets configurés:"
    
    # Liste des secrets attendus
    REQUIRED_SECRETS=("SCW_SECRET_KEY" "KUBECONFIG" "KUBECONFIG_STAGING" "KUBECONFIG_PROD")
    OPTIONAL_SECRETS=("SLACK_WEBHOOK_URL")
    
    # Récupérer la liste des secrets
    SECRET_LIST=$(gh secret list --json name --jq '.[].name')
    
    # Vérifier les secrets requis
    for secret in "${REQUIRED_SECRETS[@]}"; do
        if echo "$SECRET_LIST" | grep -q "^$secret$"; then
            log_success "✅ $secret"
        else
            log_error "❌ $secret (REQUIS)"
        fi
    done
    
    # Vérifier les secrets optionnels
    for secret in "${OPTIONAL_SECRETS[@]}"; do
        if echo "$SECRET_LIST" | grep -q "^$secret$"; then
            log_success "✅ $secret (optionnel)"
        else
            log_warning "⚠️  $secret (optionnel)"
        fi
    done
}

# Vérifier la configuration Kubernetes
check_k8s() {
    log_info "Vérification de la configuration Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé."
        return 1
    fi
    
    # Vérifier la connexion
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Impossible de se connecter au cluster Kubernetes."
        log_info "Vérifiez votre kubeconfig: kubectl config current-context"
        return 1
    fi
    
    log_success "Connexion au cluster OK"
    
    # Vérifier les namespaces
    NAMESPACES=("alphabeta808-development" "alphabeta808-staging" "alphabeta808-trading")
    
    echo ""
    log_info "Namespaces:"
    for ns in "${NAMESPACES[@]}"; do
        if kubectl get namespace "$ns" &> /dev/null; then
            log_success "✅ $ns"
        else
            log_warning "⚠️  $ns (n'existe pas)"
        fi
    done
    
    # Afficher le contexte actuel
    echo ""
    log_info "Contexte actuel: $(kubectl config current-context)"
}

# Créer les namespaces Kubernetes
setup_namespaces() {
    log_info "Création des namespaces Kubernetes..."
    
    NAMESPACES=("alphabeta808-development" "alphabeta808-staging" "alphabeta808-trading")
    
    for ns in "${NAMESPACES[@]}"; do
        if kubectl get namespace "$ns" &> /dev/null; then
            log_info "Namespace $ns existe déjà"
        else
            kubectl create namespace "$ns"
            log_success "Namespace $ns créé"
        fi
    done
    
    log_success "Tous les namespaces sont configurés"
}

# Créer les secrets Kubernetes
setup_secrets() {
    log_info "Configuration des secrets Kubernetes..."
    
    # Vérifier que le fichier .env existe
    if [ ! -f ".env" ]; then
        log_error "Fichier .env non trouvé."
        log_info "Créez un fichier .env avec vos clés API:"
        echo ""
        echo "BINANCE_API_KEY=your_api_key"
        echo "BINANCE_API_SECRET=your_api_secret"
        echo "WEBHOOK_SECRET=your_webhook_secret"
        echo "WEB_ADMIN_USER=admin"
        echo "WEB_ADMIN_PASSWORD=secure_password"
        return 1
    fi
    
    # Charger les variables d'environnement
    source .env
    
    NAMESPACES=("alphabeta808-development" "alphabeta808-staging" "alphabeta808-trading")
    
    for ns in "${NAMESPACES[@]}"; do
        log_info "Configuration secrets pour namespace: $ns"
        
        # Supprimer le secret existant s'il existe
        kubectl delete secret trading-secrets -n "$ns" &> /dev/null || true
        
        # Créer le nouveau secret
        kubectl create secret generic trading-secrets \
            --namespace="$ns" \
            --from-literal=binance-api-key="${BINANCE_API_KEY}" \
            --from-literal=binance-api-secret="${BINANCE_API_SECRET}" \
            --from-literal=webhook-secret="${WEBHOOK_SECRET:-default_webhook_secret}" \
            --from-literal=web-admin-user="${WEB_ADMIN_USER:-admin}" \
            --from-literal=web-admin-password="${WEB_ADMIN_PASSWORD:-secure_password_123}"
        
        log_success "Secrets configurés pour $ns"
    done
}

# Tester l'accès au Container Registry
test_registry() {
    log_info "Test de l'accès au Container Registry..."
    
    REGISTRY="rg.fr-par.scw.cloud/namespace-ecstatic-einstein"
    
    # Vérifier que Docker est installé
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé."
        return 1
    fi
    
    # Test de login (nécessite SCW_SECRET_KEY)
    if [ -z "$SCW_SECRET_KEY" ]; then
        log_warning "Variable SCW_SECRET_KEY non définie."
        log_info "Exportez votre clé API: export SCW_SECRET_KEY=your_secret_key"
        return 1
    fi
    
    if echo "$SCW_SECRET_KEY" | docker login "$REGISTRY" -u nologin --password-stdin &> /dev/null; then
        log_success "Connexion au registry réussie"
        
        # Test de pull d'une image (si elle existe)
        if docker pull "$REGISTRY/alphabeta808-trading-bot:latest" &> /dev/null; then
            log_success "Image trouvée sur le registry"
        else
            log_info "Aucune image trouvée sur le registry (normal pour le premier déploiement)"
        fi
    else
        log_error "Échec de la connexion au registry"
        return 1
    fi
}

# Déclencher un test du pipeline
test_pipeline() {
    log_info "Déclenchement d'un test du pipeline..."
    
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) n'est pas installé."
        return 1
    fi
    
    log_info "Déclenchement du workflow de staging..."
    
    # Déclencher le workflow staging
    if gh workflow run staging.yml --ref develop; then
        log_success "Workflow de staging déclenché"
        log_info "Surveillez l'exécution: gh run list --workflow=staging.yml"
    else
        log_error "Échec du déclenchement du workflow"
        return 1
    fi
}

# Afficher le statut général
show_status() {
    echo "🚀 AlphaBeta808 Trading Bot - Statut CI/CD"
    echo "=============================================="
    echo ""
    
    # Statut Git
    echo "📊 Repository:"
    echo "  Branch: $(git branch --show-current)"
    echo "  Commit: $(git rev-parse --short HEAD)"
    echo "  Remote: $(git remote get-url origin)"
    echo ""
    
    # Statut Kubernetes
    if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        echo "☸️  Kubernetes:"
        echo "  Cluster: $(kubectl config current-context)"
        echo "  Version: $(kubectl version --short --client 2>/dev/null | head -1)"
        
        # Statut des namespaces
        NAMESPACES=("alphabeta808-development" "alphabeta808-staging" "alphabeta808-trading")
        for ns in "${NAMESPACES[@]}"; do
            if kubectl get namespace "$ns" &> /dev/null; then
                POD_COUNT=$(kubectl get pods -n "$ns" --no-headers 2>/dev/null | wc -l | tr -d ' ')
                echo "  - $ns: $POD_COUNT pods"
            else
                echo "  - $ns: non configuré"
            fi
        done
        echo ""
    else
        echo "☸️  Kubernetes: Non accessible"
        echo ""
    fi
    
    # Statut GitHub Actions (si gh est disponible)
    if command -v gh &> /dev/null && gh auth status &> /dev/null; then
        echo "🔄 GitHub Actions:"
        RECENT_RUNS=$(gh run list --limit 3 --json status,conclusion,workflowName,createdAt --jq '.[] | "\(.workflowName): \(.conclusion // .status)"' 2>/dev/null || echo "Aucune exécution récente")
        echo "$RECENT_RUNS" | sed 's/^/  /'
        echo ""
    else
        echo "🔄 GitHub Actions: Non accessible (gh CLI requis)"
        echo ""
    fi
    
    # Statut Docker
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        echo "🐳 Docker:"
        echo "  Version: $(docker --version | cut -d' ' -f3 | tr -d ',')"
        echo "  Images: $(docker images --format 'table {{.Repository}}:{{.Tag}}' | grep alphabeta808 | wc -l | tr -d ' ') images locales"
    else
        echo "🐳 Docker: Non accessible"
    fi
}

# Parser des commandes
case "${1:-help}" in
    "check-secrets")
        check_secrets
        ;;
    "check-k8s")
        check_k8s
        ;;
    "setup-namespaces")
        setup_namespaces
        ;;
    "setup-secrets")
        setup_secrets
        ;;
    "test-registry")
        test_registry
        ;;
    "test-pipeline")
        test_pipeline
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_help
        ;;
esac

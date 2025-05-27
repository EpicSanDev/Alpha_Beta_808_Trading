#!/bin/bash

# AlphaBeta808 Trading Bot - Production Deployment Script
# Complete production deployment with monitoring and verification

set -e

echo "🚀 AlphaBeta808 Trading Bot - Production Deployment"
echo "=================================================="

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Variables
DEPLOYMENT_MODE=${1:-docker}  # docker ou kubernetes
VERSION=${2:-latest}

# Vérifier les prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    # Vérifier que nous sommes dans le bon répertoire
    if [ ! -f "trader_config.json" ]; then
        log_error "Veuillez exécuter ce script depuis le répertoire racine d'AlphaBeta808Trading"
        exit 1
    fi
    
    # Vérifier les fichiers de configuration
    if [ ! -f ".env" ]; then
        log_error "Fichier .env manquant. Exécutez setup_production.sh d'abord"
        exit 1
    fi
    
    # Vérifier la configuration des clés API
    if grep -q "your_binance_api_key_here" .env; then
        log_error "Veuillez configurer vos clés API Binance dans le fichier .env"
        exit 1
    fi
    
    log_success "Prérequis vérifiés"
}

# Test du système
test_system() {
    log_info "Test du système complet..."
    
    # Activer l'environnement virtuel
    if [ -d "trading_env" ]; then
        source trading_env/bin/activate
    fi
    
    # Exécuter les tests de vérification
    python system_verification.py
    if [ $? -eq 0 ]; then
        log_success "Système vérifié avec succès"
    else
        log_error "Le système a échoué aux tests de vérification"
        exit 1
    fi
}

# Déploiement Docker
deploy_docker() {
    log_info "Déploiement avec Docker Compose..."
    
    # Construire l'image
    log_info "Construction de l'image Docker..."
    docker build -t alphabeta808/trading-bot:$VERSION .
    
    # Démarrer les services
    log_info "Démarrage des services..."
    docker-compose down || true
    docker-compose up -d
    
    # Attendre que les services soient prêts
    log_info "Attente du démarrage des services..."
    sleep 30
    
    # Vérifier que les services sont en cours d'exécution
    if docker-compose ps | grep -q "Up"; then
        log_success "Services Docker démarrés avec succès"
    else
        log_error "Échec du démarrage des services Docker"
        docker-compose logs
        exit 1
    fi
    
    # Test de connectivité
    log_info "Test de connectivité..."
    max_attempts=10
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:5000/api/system/status > /dev/null 2>&1; then
            log_success "Interface web accessible"
            break
        else
            log_info "Tentative $attempt/$max_attempts - En attente de l'interface web..."
            sleep 10
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Interface web non accessible après $max_attempts tentatives"
        exit 1
    fi
}

# Déploiement Kubernetes
deploy_kubernetes() {
    log_info "Déploiement sur Kubernetes..."
    
    # Vérifier kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl non trouvé. Veuillez l'installer pour le déploiement Kubernetes"
        exit 1
    fi
    
    # Vérifier la connexion au cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Pas de connexion au cluster Kubernetes"
        exit 1
    fi
    
    # Exécuter le script de déploiement Kubernetes
    cd k8s
    ./deploy.sh $VERSION
    cd ..
    
    log_success "Déploiement Kubernetes terminé"
}

# Configuration du monitoring
setup_monitoring() {
    log_info "Configuration du monitoring..."
    
    # Créer le répertoire des logs s'il n'existe pas
    mkdir -p logs
    
    # Démarrer le monitoring en arrière-plan
    if [ "$DEPLOYMENT_MODE" = "docker" ]; then
        # Le monitoring est inclus dans docker-compose.yml
        log_success "Monitoring configuré avec Docker Compose"
        log_info "Accès Grafana: http://localhost:3000 (admin/admin123)"
        log_info "Accès Prometheus: http://localhost:9090"
    else
        log_info "Monitoring Kubernetes configuré via monitoring.yaml"
    fi
}

# Vérification post-déploiement
post_deployment_checks() {
    log_info "Vérifications post-déploiement..."
    
    # Liste des endpoints à vérifier
    if [ "$DEPLOYMENT_MODE" = "docker" ]; then
        endpoints=(
            "http://localhost:5000/api/system/status"
            "http://localhost:5000/api/bot/status"
            "http://localhost:3000"  # Grafana
            "http://localhost:9090"  # Prometheus
        )
    else
        # Pour Kubernetes, adapter selon votre configuration d'ingress
        endpoints=(
            "http://localhost:5000/api/system/status"
        )
    fi
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f "$endpoint" > /dev/null 2>&1; then
            log_success "✅ $endpoint"
        else
            log_warning "⚠️  $endpoint non accessible"
        fi
    done
    
    # Vérifier les logs pour des erreurs
    log_info "Vérification des logs..."
    if [ -f "logs/continuous_trader.log" ]; then
        error_count=$(grep -c "ERROR" logs/continuous_trader.log || echo "0")
        if [ "$error_count" -gt 0 ]; then
            log_warning "$error_count erreurs trouvées dans les logs"
        else
            log_success "Aucune erreur dans les logs"
        fi
    fi
}

# Affichage des informations de connexion
show_connection_info() {
    echo ""
    echo "🎉 Déploiement terminé avec succès!"
    echo "=================================="
    echo ""
    log_info "Informations de connexion:"
    
    if [ "$DEPLOYMENT_MODE" = "docker" ]; then
        echo "  📊 Interface Web:    http://localhost:5000"
        echo "  📈 Grafana:         http://localhost:3000 (admin/admin123)"
        echo "  🔍 Prometheus:      http://localhost:9090"
        echo "  🗄️  Redis:           localhost:6379"
    else
        echo "  📊 Interface Web:    http://localhost:5000 (via port-forward)"
        echo "  🏗️  Cluster:         kubectl get pods -n alphabeta808-trading"
    fi
    
    echo ""
    log_info "Commandes utiles:"
    echo "  📋 Voir les logs:        docker-compose logs -f (Docker) ou kubectl logs -f deployment/trading-bot -n alphabeta808-trading (K8s)"
    echo "  🔄 Redémarrer:          docker-compose restart (Docker) ou kubectl rollout restart deployment/trading-bot -n alphabeta808-trading (K8s)"
    echo "  📊 Status:              curl http://localhost:5000/api/system/status"
    echo "  🛑 Arrêter:             docker-compose down (Docker) ou k8s/undeploy.sh (K8s)"
    echo ""
    
    log_warning "IMPORTANT:"
    echo "  - Surveillez les logs pour détecter d'éventuels problèmes"
    echo "  - Configurez des sauvegardes régulières des modèles et données"
    echo "  - Testez le système avec de petits montants avant la production"
    echo "  - Surveillez la consommation des ressources"
    echo ""
}

# Sauvegarde de sécurité
create_backup() {
    log_info "Création d'une sauvegarde de sécurité..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/deployment_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Sauvegarder les fichiers critiques
    cp -r models_store "$backup_dir/" 2>/dev/null || log_warning "Aucun modèle à sauvegarder"
    cp trader_config.json "$backup_dir/"
    cp .env "$backup_dir/.env.backup"
    
    # Sauvegarder la base de données si elle existe
    if [ -f "trading_web.db" ]; then
        cp trading_web.db "$backup_dir/"
    fi
    
    log_success "Sauvegarde créée dans $backup_dir"
}

# Script principal
main() {
    log_info "Mode de déploiement: $DEPLOYMENT_MODE"
    log_info "Version: $VERSION"
    echo ""
    
    # Étapes de déploiement
    check_prerequisites
    create_backup
    test_system
    
    case $DEPLOYMENT_MODE in
        "docker")
            deploy_docker
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        *)
            log_error "Mode de déploiement non supporté: $DEPLOYMENT_MODE"
            log_info "Modes disponibles: docker, kubernetes"
            exit 1
            ;;
    esac
    
    setup_monitoring
    post_deployment_checks
    show_connection_info
}

# Gestion des signaux pour un arrêt propre
trap 'log_warning "Interruption détectée, nettoyage..."; exit 1' INT TERM

# Affichage de l'aide
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [MODE] [VERSION]"
    echo ""
    echo "MODE:"
    echo "  docker      - Déploiement avec Docker Compose (défaut)"
    echo "  kubernetes  - Déploiement sur Kubernetes"
    echo ""
    echo "VERSION:"
    echo "  latest      - Version la plus récente (défaut)"
    echo "  v1.0.0      - Version spécifique"
    echo ""
    echo "Exemples:"
    echo "  $0                    # Docker avec version latest"
    echo "  $0 docker v1.0.0      # Docker avec version v1.0.0"
    echo "  $0 kubernetes         # Kubernetes avec version latest"
    echo ""
    exit 0
fi

# Exécution du script principal
main

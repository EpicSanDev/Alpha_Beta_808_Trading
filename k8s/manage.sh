#!/bin/bash

# AlphaBeta808 Trading Bot - Kubernetes Status and Management Script
# Ce script permet de surveiller et gérer le déploiement

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

# Afficher le statut général
show_status() {
    echo "📊 Statut du déploiement AlphaBeta808"
    echo "====================================="
    
    log_info "Pods:"
    kubectl get pods -n ${NAMESPACE} -o wide
    
    echo ""
    log_info "Services:"
    kubectl get svc -n ${NAMESPACE}
    
    echo ""
    log_info "Deployments:"
    kubectl get deployments -n ${NAMESPACE}
    
    echo ""
    log_info "PVC:"
    kubectl get pvc -n ${NAMESPACE}
    
    echo ""
    log_info "HPA:"
    kubectl get hpa -n ${NAMESPACE}
}

# Afficher les logs
show_logs() {
    local component=$1
    
    case $component in
        "bot"|"trading-bot")
            log_info "Logs du Trading Bot:"
            kubectl logs -f deployment/trading-bot -n ${NAMESPACE}
            ;;
        "web"|"web-interface")
            log_info "Logs de l'Interface Web:"
            kubectl logs -f deployment/trading-web-interface -n ${NAMESPACE}
            ;;
        *)
            echo "Usage: $0 logs [bot|web]"
            echo "Composants disponibles:"
            echo "  bot, trading-bot     - Logs du bot de trading"
            echo "  web, web-interface   - Logs de l'interface web"
            ;;
    esac
}

# Se connecter à un pod
connect_pod() {
    local component=$1
    
    case $component in
        "bot"|"trading-bot")
            log_info "Connexion au pod Trading Bot:"
            kubectl exec -it deployment/trading-bot -n ${NAMESPACE} -- /bin/bash
            ;;
        "web"|"web-interface")
            log_info "Connexion au pod Interface Web:"
            kubectl exec -it deployment/trading-web-interface -n ${NAMESPACE} -- /bin/bash
            ;;
        *)
            echo "Usage: $0 shell [bot|web]"
            ;;
    esac
}

# Port forwarding pour accès local
port_forward() {
    local service=$1
    local local_port=${2:-8080}
    
    case $service in
        "web"|"web-interface")
            log_info "Port forwarding pour l'interface web sur le port ${local_port}..."
            kubectl port-forward svc/trading-web-service ${local_port}:5000 -n ${NAMESPACE}
            ;;
        *)
            echo "Usage: $0 port-forward [web] [port]"
            ;;
    esac
}

# Redémarrer un déploiement
restart_deployment() {
    local component=$1
    
    case $component in
        "bot"|"trading-bot")
            log_info "Redémarrage du Trading Bot..."
            kubectl rollout restart deployment/trading-bot -n ${NAMESPACE}
            kubectl rollout status deployment/trading-bot -n ${NAMESPACE}
            ;;
        "web"|"web-interface")
            log_info "Redémarrage de l'Interface Web..."
            kubectl rollout restart deployment/trading-web-interface -n ${NAMESPACE}
            kubectl rollout status deployment/trading-web-interface -n ${NAMESPACE}
            ;;
        "all")
            log_info "Redémarrage de tous les composants..."
            kubectl rollout restart deployment/trading-bot -n ${NAMESPACE}
            kubectl rollout restart deployment/trading-web-interface -n ${NAMESPACE}
            kubectl rollout status deployment/trading-bot -n ${NAMESPACE}
            kubectl rollout status deployment/trading-web-interface -n ${NAMESPACE}
            ;;
        *)
            echo "Usage: $0 restart [bot|web|all]"
            ;;
    esac
}

# Scaler un déploiement
scale_deployment() {
    local component=$1
    local replicas=$2
    
    if [ -z "$replicas" ]; then
        echo "Usage: $0 scale [bot|web] [replicas]"
        return 1
    fi
    
    case $component in
        "bot"|"trading-bot")
            log_info "Scaling Trading Bot à ${replicas} répliques..."
            kubectl scale deployment/trading-bot --replicas=${replicas} -n ${NAMESPACE}
            ;;
        "web"|"web-interface")
            log_info "Scaling Interface Web à ${replicas} répliques..."
            kubectl scale deployment/trading-web-interface --replicas=${replicas} -n ${NAMESPACE}
            ;;
        *)
            echo "Usage: $0 scale [bot|web] [replicas]"
            ;;
    esac
}

# Afficher les métriques
show_metrics() {
    log_info "Métriques des pods:"
    kubectl top pods -n ${NAMESPACE}
    
    echo ""
    log_info "Métriques des nodes:"
    kubectl top nodes
}

# Menu d'aide
show_help() {
    echo "🤖 AlphaBeta808 Trading Bot - Gestion Kubernetes"
    echo "================================================"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commandes disponibles:"
    echo "  status                    - Afficher le statut général"
    echo "  logs [bot|web]           - Afficher les logs d'un composant"
    echo "  shell [bot|web]          - Se connecter à un pod"
    echo "  port-forward [web] [port] - Port forwarding pour accès local"
    echo "  restart [bot|web|all]    - Redémarrer un déploiement"
    echo "  scale [bot|web] [n]      - Scaler un déploiement"
    echo "  metrics                  - Afficher les métriques"
    echo "  help                     - Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  $0 status"
    echo "  $0 logs bot"
    echo "  $0 port-forward web 8080"
    echo "  $0 restart all"
    echo "  $0 scale web 2"
}

# Fonction principale
main() {
    case $1 in
        "status"|"")
            show_status
            ;;
        "logs")
            show_logs $2
            ;;
        "shell"|"exec")
            connect_pod $2
            ;;
        "port-forward"|"pf")
            port_forward $2 $3
            ;;
        "restart")
            restart_deployment $2
            ;;
        "scale")
            scale_deployment $2 $3
            ;;
        "metrics"|"top")
            show_metrics
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            echo "Commande inconnue: $1"
            echo "Utilisez '$0 help' pour voir les commandes disponibles"
            exit 1
            ;;
    esac
}

# Vérifier que kubectl est disponible
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl n'est pas installé ou pas dans le PATH"
    exit 1
fi

# Vérifier que le namespace existe
if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
    log_error "Namespace '${NAMESPACE}' n'existe pas. Avez-vous déployé l'application?"
    exit 1
fi

main "$@"

#!/bin/bash

# Script de validation de la configuration CI/CD
# Usage: ./validate-setup.sh

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Validation de la Configuration CI/CD AlphaBeta808Trading ===${NC}"
echo ""

# Compteurs
CHECKS_PASSED=0
CHECKS_FAILED=0
TOTAL_CHECKS=0

# Fonction pour afficher le résultat d'un test
check_result() {
    local description=$1
    local success=$2
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$success" = true ]; then
        echo -e "${GREEN}✓${NC} $description"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $description"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
}

# Fonction pour vérifier une commande
check_command() {
    local cmd=$1
    local description=$2
    
    if command -v "$cmd" &> /dev/null; then
        check_result "$description" true
    else
        check_result "$description" false
    fi
}

# Fonction pour vérifier un fichier
check_file() {
    local file_path=$1
    local description=$2
    
    if [ -f "$file_path" ]; then
        check_result "$description" true
    else
        check_result "$description" false
    fi
}

# Fonction pour vérifier les secrets GitHub
check_github_secrets() {
    echo -e "${YELLOW}=== Vérification des outils requis ===${NC}"
    
    check_command "gh" "GitHub CLI installé"
    check_command "kubectl" "kubectl installé"
    check_command "docker" "Docker installé"
    check_command "openssl" "OpenSSL installé"
    
    # Vérifier l'authentification GitHub
    if gh auth status &> /dev/null; then
        check_result "Authentification GitHub CLI" true
    else
        check_result "Authentification GitHub CLI" false
        echo -e "${YELLOW}  Connectez-vous avec: gh auth login${NC}"
    fi
    
    echo ""
}

# Fonction pour vérifier les fichiers de configuration
check_config_files() {
    echo -e "${YELLOW}=== Vérification des fichiers de configuration ===${NC}"
    
    # Fichiers GitHub Actions
    check_file ".github/workflows/ci-cd.yml" "Pipeline CI/CD principal"
    check_file ".github/workflows/staging.yml" "Pipeline staging"
    check_file ".github/workflows/release.yml" "Pipeline release"
    check_file ".github/setup-secrets.sh" "Script configuration secrets"
    
    # Fichiers Kubernetes
    check_file "k8s/bot-deployment.yaml" "Déploiement bot"
    check_file "k8s/web-deployment.yaml" "Déploiement web"
    check_file "k8s/configmap.yaml" "ConfigMap"
    check_file "k8s/secrets.yaml" "Template secrets"
    check_file "k8s/services.yaml" "Services"
    check_file "k8s/deploy-with-secrets.sh" "Script déploiement local"
    
    # Fichiers application
    check_file "Dockerfile" "Dockerfile"
    check_file "requirements.txt" "Requirements Python"
    check_file "trader_config.json" "Configuration trader"
    check_file "monitoring_config.json" "Configuration monitoring"
    
    echo ""
}

# Fonction pour vérifier la configuration Kubernetes
check_kubernetes() {
    echo -e "${YELLOW}=== Vérification de la configuration Kubernetes ===${NC}"
    
    # Vérifier la connexion au cluster
    if kubectl cluster-info &> /dev/null; then
        check_result "Connexion au cluster Kubernetes" true
        
        # Vérifier les namespaces
        if kubectl get namespace alphabeta808-trading &> /dev/null; then
            check_result "Namespace production existe" true
        else
            check_result "Namespace production existe" false
            echo -e "${YELLOW}  Créez avec: kubectl create namespace alphabeta808-trading${NC}"
        fi
        
        if kubectl get namespace alphabeta808-trading-staging &> /dev/null; then
            check_result "Namespace staging existe" true
        else
            check_result "Namespace staging existe" false
            echo -e "${YELLOW}  Créez avec: kubectl create namespace alphabeta808-trading-staging${NC}"
        fi
        
    else
        check_result "Connexion au cluster Kubernetes" false
        echo -e "${YELLOW}  Configurez kubectl avec votre cluster${NC}"
    fi
    
    echo ""
}

# Fonction pour vérifier la configuration Docker
check_docker() {
    echo -e "${YELLOW}=== Vérification de la configuration Docker ===${NC}"
    
    # Vérifier que Docker fonctionne
    if docker info &> /dev/null; then
        check_result "Docker daemon actif" true
    else
        check_result "Docker daemon actif" false
    fi
    
    # Vérifier si l'image existe localement
    if docker images | grep -q "alphabeta808-trading-bot"; then
        check_result "Image Docker locale disponible" true
    else
        check_result "Image Docker locale disponible" false
        echo -e "${YELLOW}  Buildez avec: docker build -t alphabeta808-trading-bot .${NC}"
    fi
    
    echo ""
}

# Fonction pour vérifier les secrets GitHub (si possible)
check_github_secrets_exist() {
    echo -e "${YELLOW}=== Vérification des secrets GitHub ===${NC}"
    
    if gh auth status &> /dev/null; then
        # Liste des secrets requis
        required_secrets=(
            "BINANCE_API_KEY_STAGING"
            "BINANCE_API_SECRET_STAGING"
            "BINANCE_API_KEY_PRODUCTION"
            "BINANCE_API_SECRET_PRODUCTION"
            "SCW_SECRET_KEY"
            "KUBECONFIG_STAGING"
            "KUBECONFIG_PRODUCTION"
            "WEBHOOK_SECRET"
            "WEB_ADMIN_USER"
            "WEB_ADMIN_PASSWORD"
        )
        
        # Obtenir la liste des secrets configurés
        if secrets_list=$(gh secret list 2>/dev/null); then
            for secret in "${required_secrets[@]}"; do
                if echo "$secrets_list" | grep -q "$secret"; then
                    check_result "Secret $secret configuré" true
                else
                    check_result "Secret $secret configuré" false
                fi
            done
        else
            check_result "Accès aux secrets GitHub" false
            echo -e "${YELLOW}  Vérifiez les permissions de votre token GitHub${NC}"
        fi
    else
        echo -e "${YELLOW}Authentification GitHub requise pour vérifier les secrets${NC}"
    fi
    
    echo ""
}

# Fonction pour vérifier la syntaxe YAML
check_yaml_syntax() {
    echo -e "${YELLOW}=== Vérification de la syntaxe YAML ===${NC}"
    
    yaml_files=(
        ".github/workflows/ci-cd.yml"
        ".github/workflows/staging.yml"
        ".github/workflows/release.yml"
        "k8s/bot-deployment.yaml"
        "k8s/web-deployment.yaml"
        "k8s/configmap.yaml"
        "k8s/secrets.yaml"
        "k8s/services.yaml"
        "docker-compose.yml"
    )
    
    for yaml_file in "${yaml_files[@]}"; do
        if [ -f "$yaml_file" ]; then
            if python3 -c "import yaml; yaml.safe_load(open('$yaml_file'))" 2>/dev/null; then
                check_result "Syntaxe YAML valide: $(basename "$yaml_file")" true
            else
                check_result "Syntaxe YAML valide: $(basename "$yaml_file")" false
            fi
        fi
    done
    
    echo ""
}

# Fonction pour donner des recommandations
give_recommendations() {
    echo -e "${BLUE}=== Recommandations ===${NC}"
    echo ""
    
    if [ $CHECKS_FAILED -gt 0 ]; then
        echo -e "${YELLOW}Actions recommandées pour corriger les problèmes:${NC}"
        echo ""
        
        # Recommandations spécifiques
        if ! command -v gh &> /dev/null; then
            echo "📥 Installez GitHub CLI: https://cli.github.com/"
        fi
        
        if ! gh auth status &> /dev/null; then
            echo "🔐 Authentifiez-vous avec GitHub: gh auth login"
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            echo "☸️ Configurez kubectl avec votre cluster Kubernetes"
        fi
        
        if ! docker info &> /dev/null; then
            echo "🐳 Démarrez Docker daemon"
        fi
        
        echo ""
        echo -e "${YELLOW}Scripts disponibles pour vous aider:${NC}"
        echo "• Configuration des secrets: .github/setup-secrets.sh"
        echo "• Déploiement local: k8s/deploy-with-secrets.sh"
        echo "• Guide complet: .github/CI-CD-SETUP-GUIDE.md"
        echo ""
    fi
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 Excellent ! Votre configuration semble complète.${NC}"
        echo ""
        echo -e "${BLUE}Prochaines étapes:${NC}"
        echo "1. Testez en staging: git push origin develop"
        echo "2. Surveillez le déploiement dans GitHub Actions"
        echo "3. Vérifiez les logs: kubectl logs -f deployment/trading-bot -n alphabeta808-trading-staging"
        echo "4. Accédez à l'interface: kubectl port-forward svc/trading-web-service 8080:80 -n alphabeta808-trading-staging"
        echo ""
    else
        echo -e "${YELLOW}⚠️ Quelques éléments nécessitent votre attention.${NC}"
        echo "Corrigez les problèmes identifiés avant de déployer en production."
        echo ""
    fi
}

# Script principal
main() {
    check_github_secrets
    check_config_files
    check_kubernetes
    check_docker
    check_github_secrets_exist
    check_yaml_syntax
    
    echo -e "${BLUE}=== Résumé de la validation ===${NC}"
    echo ""
    echo -e "Tests réussis: ${GREEN}$CHECKS_PASSED${NC}"
    echo -e "Tests échoués: ${RED}$CHECKS_FAILED${NC}"
    echo -e "Total: $TOTAL_CHECKS"
    echo ""
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ Configuration validée avec succès !${NC}"
    else
        echo -e "${YELLOW}⚠️ Configuration incomplète ($CHECKS_FAILED problème(s) détecté(s))${NC}"
    fi
    
    echo ""
    give_recommendations
}

# Exécuter la validation
main

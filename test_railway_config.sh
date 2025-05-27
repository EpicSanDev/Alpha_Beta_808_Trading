#!/bin/bash

# AlphaBeta808 Trading Bot - Railway Pre-deployment Test
# Ce script teste la configuration Railway localement avant le déploiement

set -e

echo "🚀 AlphaBeta808 Trading Bot - Railway Pre-deployment Test"
echo "========================================================"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les résultats
function print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Test 1: Vérifier la présence des fichiers Railway
echo -e "\n${BLUE}📋 Test 1: Vérification des fichiers de configuration Railway${NC}"

if [ -f "railway.toml" ]; then
    print_status 0 "railway.toml existe"
else
    print_status 1 "railway.toml manquant"
    exit 1
fi

if [ -f "Procfile" ]; then
    print_status 0 "Procfile existe"
else
    print_status 1 "Procfile manquant"
    exit 1
fi

if [ -f "Dockerfile.railway" ]; then
    print_status 0 "Dockerfile.railway existe"
else
    print_status 1 "Dockerfile.railway manquant"
    exit 1
fi

# Test 2: Vérifier la configuration railway.toml
echo -e "\n${BLUE}📋 Test 2: Validation de railway.toml${NC}"

if grep -q "dockerfilePath = \"Dockerfile.railway\"" railway.toml; then
    print_status 0 "Dockerfile.railway correctement référencé"
else
    print_status 1 "Dockerfile.railway mal référencé dans railway.toml"
fi

if grep -q "healthcheckPath = \"/health\"" railway.toml; then
    print_status 0 "Health check path configuré"
else
    print_status 1 "Health check path mal configuré"
fi

# Test 3: Vérifier l'application Flask
echo -e "\n${BLUE}📋 Test 3: Test de l'application Flask${NC}"

if [ -f "web_interface/app_enhanced.py" ]; then
    print_status 0 "Application Flask existe"
    
    # Vérifier l'endpoint de santé
    if grep -q "@app.route('/health'" web_interface/app_enhanced.py; then
        print_status 0 "Endpoint /health trouvé"
    else
        print_status 1 "Endpoint /health manquant"
    fi
else
    print_status 1 "Application Flask manquante"
    exit 1
fi

# Test 4: Vérifier les dépendances
echo -e "\n${BLUE}📋 Test 4: Vérification des dépendances${NC}"

if [ -f "requirements.txt" ]; then
    print_status 0 "requirements.txt existe"
    
    # Vérifier quelques dépendances clés
    if grep -q "flask" requirements.txt; then
        print_status 0 "Flask dans requirements.txt"
    else
        print_status 1 "Flask manquant dans requirements.txt"
    fi
    
    if grep -q "flask-socketio" requirements.txt; then
        print_status 0 "Flask-SocketIO dans requirements.txt"
    else
        print_status 1 "Flask-SocketIO manquant dans requirements.txt"
    fi
else
    print_status 1 "requirements.txt manquant"
fi

# Test 5: Variables d'environnement
echo -e "\n${BLUE}📋 Test 5: Vérification des variables d'environnement${NC}"

if [ -f ".env.railway" ]; then
    print_status 0 "Fichier .env.railway exemple existe"
else
    print_warning "Fichier .env.railway manquant (optionnel)"
fi

# Test 6: Test de build Docker (optionnel)
echo -e "\n${BLUE}📋 Test 6: Test de build Docker (optionnel)${NC}"

print_info "Voulez-vous tester le build Docker ? (y/n)"
read -p "Build Docker: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Construction de l'image Docker Railway..."
    
    if docker build -f Dockerfile.railway -t alphabeta808-railway-test .; then
        print_status 0 "Build Docker réussi"
        
        # Nettoyer l'image de test
        docker rmi alphabeta808-railway-test > /dev/null 2>&1 || true
    else
        print_status 1 "Build Docker échoué"
    fi
else
    print_info "Test Docker ignoré"
fi

# Test 7: Validation des ports
echo -e "\n${BLUE}📋 Test 7: Validation de la configuration des ports${NC}"

if grep -q "PORT = \$PORT" railway.toml; then
    print_status 0 "Variable PORT configurée pour Railway"
else
    print_status 1 "Variable PORT mal configurée"
fi

# Test 8: Sécurité de base
echo -e "\n${BLUE}📋 Test 8: Vérification de sécurité de base${NC}"

# Vérifier qu'il n'y a pas de secrets hardcodés
if grep -r "sk-" . --include="*.py" --include="*.toml" --include="*.txt" > /dev/null 2>&1; then
    print_status 1 "Secrets potentiels trouvés dans le code"
else
    print_status 0 "Pas de secrets hardcodés détectés"
fi

if grep -r "AKIA" . --include="*.py" --include="*.toml" --include="*.txt" > /dev/null 2>&1; then
    print_status 1 "Clés AWS potentielles trouvées dans le code"
else
    print_status 0 "Pas de clés AWS hardcodées détectées"
fi

# Test 9: Structure des répertoires
echo -e "\n${BLUE}📋 Test 9: Vérification de la structure des répertoires${NC}"

required_dirs=("web_interface" "src" "config")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_status 0 "Répertoire $dir existe"
    else
        print_status 1 "Répertoire $dir manquant"
    fi
done

# Résumé final
echo -e "\n${BLUE}🎯 Résumé du test de pré-déploiement${NC}"
echo "=================================================="

print_info "Configuration Railway : ✅ Prête"
print_info "Application Flask : ✅ Configurée"
print_info "Dockerfile optimisé : ✅ Présent"
print_info "Health check : ✅ Configuré"

echo -e "\n${GREEN}🚀 Prochaines étapes pour le déploiement Railway :${NC}"
echo "1. Créer un compte Railway (https://railway.app)"
echo "2. Connecter votre repository GitHub"
echo "3. Configurer les variables d'environnement (voir .env.railway)"
echo "4. Générer des clés sécurisées :"
echo "   openssl rand -base64 32  # Pour SECRET_KEY"
echo "   openssl rand -base64 32  # Pour WEBHOOK_SECRET"
echo "5. Ajouter vos clés API Binance"
echo "6. Déployer et tester l'endpoint de santé"

echo -e "\n${YELLOW}⚠️  Important :${NC}"
echo "- Utilisez le mode sandbox/testnet initialement"
echo "- Changez le mot de passe admin par défaut"
echo "- Surveillez les logs lors du premier déploiement"

echo -e "\n${GREEN}✅ Test de pré-déploiement terminé avec succès !${NC}"

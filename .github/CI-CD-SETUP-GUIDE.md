# Guide Configuration CI/CD pour AlphaBeta808Trading

Ce guide vous explique comment configurer le déploiement automatique de votre bot de trading avec GitHub Actions et Kubernetes.

## 🎯 Vue d'ensemble

Notre pipeline CI/CD comprend :
- **Tests automatiques** sur chaque push/PR
- **Build et push Docker** vers Scaleway Container Registry
- **Déploiement automatique** vers staging (branche `develop`) et production (branche `main`)
- **Gestion sécurisée des secrets** via GitHub Secrets

## 📋 Prérequis

### 1. Comptes et accès requis
- [ ] Compte GitHub avec droits d'administration sur le repository
- [ ] Compte Binance avec API activée
- [ ] Cluster Kubernetes configuré
- [ ] Scaleway Container Registry configuré

### 2. Outils locaux
- [ ] `gh` CLI installé et configuré
- [ ] `kubectl` installé et configuré
- [ ] `docker` installé

## 🔑 Configuration des Secrets GitHub

### Étape 1: Configuration automatique
Utilisez notre script automatisé pour configurer tous les secrets :

```bash
# Depuis la racine du projet
cd .github/
./setup-secrets.sh
```

### Étape 2: Configuration manuelle (alternative)

Si vous préférez configurer manuellement, allez dans **Settings > Secrets and variables > Actions** de votre repository GitHub et ajoutez :

#### Secrets Binance (REQUIS)
| Nom du Secret | Description | Exemple |
|---------------|-------------|---------|
| `BINANCE_API_KEY_STAGING` | Clé API Binance testnet | `abc123...` |
| `BINANCE_API_SECRET_STAGING` | Secret API Binance testnet | `def456...` |
| `BINANCE_API_KEY_PRODUCTION` | Clé API Binance mainnet | `ghi789...` |
| `BINANCE_API_SECRET_PRODUCTION` | Secret API Binance mainnet | `jkl012...` |

#### Secrets Infrastructure
| Nom du Secret | Description | Comment obtenir |
|---------------|-------------|-----------------|
| `SCW_SECRET_KEY` | Clé Scaleway Container Registry | Console Scaleway > IAM |
| `KUBECONFIG_STAGING` | Config Kubernetes staging (base64) | `cat ~/.kube/config \| base64 -w 0` |
| `KUBECONFIG_PRODUCTION` | Config Kubernetes production (base64) | `cat ~/.kube/config-prod \| base64 -w 0` |

#### Secrets Application
| Nom du Secret | Description | Génération |
|---------------|-------------|------------|
| `WEBHOOK_SECRET` | Secret pour webhooks | `openssl rand -hex 32` |
| `WEB_ADMIN_USER` | Utilisateur admin web | `admin` |
| `WEB_ADMIN_PASSWORD` | Mot de passe admin | Mot de passe fort |
| `SMTP_PASSWORD` | Mot de passe email SMTP | Depuis votre provider email |
| `DATABASE_URL_STAGING` | URL base de données staging | `postgresql://...` (optionnel) |
| `DATABASE_URL_PRODUCTION` | URL base de données production | `postgresql://...` (optionnel) |

## 🚀 Processus de Déploiement

### Déploiement Staging
1. **Push vers `develop`** déclenche automatiquement :
   - Tests automatiques
   - Build Docker
   - Déploiement vers staging

```bash
git checkout develop
git push origin develop
```

### Déploiement Production
1. **Push vers `main`** déclenche automatiquement :
   - Tests automatiques
   - Build Docker
   - Déploiement vers production

```bash
git checkout main
git merge develop
git push origin main
```

### Déploiement Manuel
Vous pouvez aussi déclencher manuellement via GitHub Actions :
1. Allez dans **Actions** sur GitHub
2. Sélectionnez **CI/CD Pipeline**
3. Cliquez **Run workflow**

## 🏗️ Architecture des Environnements

### Staging (`develop` branch)
- **Namespace**: `alphabeta808-trading-staging`
- **Binance**: Testnet (argent virtuel)
- **Surveillance**: Limitée
- **URL**: Port-forward local

### Production (`main` branch)
- **Namespace**: `alphabeta808-trading`
- **Binance**: Mainnet (argent réel!)
- **Surveillance**: Complète (HPA, PDB, monitoring)
- **URL**: Ingress configuré

## 🛠️ Déploiement Local

Pour déployer localement sans passer par GitHub Actions :

```bash
# Staging
cd k8s/
./deploy-with-secrets.sh staging

# Production (ATTENTION!)
./deploy-with-secrets.sh production
```

## 📊 Monitoring et Logs

### Voir les pods
```bash
# Staging
kubectl get pods -n alphabeta808-trading-staging

# Production
kubectl get pods -n alphabeta808-trading
```

### Voir les logs
```bash
# Bot de trading
kubectl logs -f deployment/trading-bot -n alphabeta808-trading

# Interface web
kubectl logs -f deployment/trading-web-interface -n alphabeta808-trading
```

### Accéder à l'interface web
```bash
# Port-forward local
kubectl port-forward svc/trading-web-service 8080:80 -n alphabeta808-trading

# Puis ouvrir: http://localhost:8080
```

## 🔒 Sécurité

### Bonnes Pratiques
- ✅ Utilisez **toujours le testnet** pour staging
- ✅ **Vérifiez deux fois** avant de déployer en production
- ✅ **Limitez les permissions** des clés API Binance
- ✅ **Surveillez les logs** régulièrement
- ✅ **Configurez des alertes** pour les problèmes critiques

### Configuration Clés API Binance
1. **Pour Staging (Testnet)** :
   - Créez des clés sur testnet.binance.vision
   - Permissions : Trading, Reading
   - Restriction IP recommandée

2. **Pour Production (Mainnet)** :
   - Créez des clés sur binance.com
   - Permissions : Trading, Reading uniquement
   - **OBLIGATOIRE** : Restriction IP
   - Activez la 2FA sur votre compte

## 🚨 Dépannage

### Problèmes courants

#### 1. Échec de déploiement Kubernetes
```bash
# Vérifier les événements
kubectl get events -n alphabeta808-trading --sort-by='.lastTimestamp'

# Vérifier les logs
kubectl describe pod <nom-du-pod> -n alphabeta808-trading
```

#### 2. Secrets manquants
```bash
# Vérifier les secrets
kubectl get secrets -n alphabeta808-trading

# Voir le contenu (masqué)
kubectl describe secret trading-secrets -n alphabeta808-trading
```

#### 3. Images Docker non trouvées
- Vérifiez que `SCW_SECRET_KEY` est correct
- Vérifiez que l'image existe dans le registry

#### 4. Erreurs de connexion Binance
- Vérifiez les clés API (testnet vs mainnet)
- Vérifiez les restrictions IP
- Vérifiez les permissions des clés

### Logs de debug
```bash
# Voir tous les logs récents
kubectl logs --previous -l app=trading-bot -n alphabeta808-trading

# Suivre les logs en temps réel
kubectl logs -f -l app=trading-bot -n alphabeta808-trading --tail=100
```

## 📁 Structure des Fichiers

```
.github/
├── workflows/
│   ├── ci-cd.yml           # Pipeline principal
│   ├── staging.yml         # Pipeline staging
│   └── release.yml         # Pipeline release
├── setup-secrets.sh        # Script configuration secrets
└── README.md

k8s/
├── deploy-with-secrets.sh  # Déploiement local automatisé
├── secrets.yaml           # Template secrets
├── configmap.yaml         # Configuration
├── bot-deployment.yaml    # Déploiement bot
├── web-deployment.yaml    # Déploiement web
├── services.yaml          # Services
├── hpa.yaml               # Auto-scaling
├── pdb.yaml               # Disruption budget
├── monitoring.yaml        # Monitoring
└── ingress.yaml           # Exposition externe
```

## 🎯 Prochaines Étapes

1. **Configurez vos secrets** avec `./setup-secrets.sh`
2. **Testez en staging** en pushant vers `develop`
3. **Surveillez les logs** et les métriques
4. **Déployez en production** en pushant vers `main`
5. **Configurez la surveillance** et les alertes

## 🆘 Support

En cas de problème :
1. Consultez les logs Kubernetes et GitHub Actions
2. Vérifiez la documentation Binance API
3. Consultez le troubleshooting dans ce guide
4. Ouvrez une issue sur le repository

---
**⚠️ RAPPEL IMPORTANT** : Le trading automatisé implique des risques financiers. Testez toujours en staging avant la production !

# 🔧 Guide de Configuration du Pipeline CI/CD

Ce guide vous accompagne dans la configuration complète du pipeline CI/CD pour le projet AlphaBeta808 Trading Bot.

## 📋 Prérequis

- [ ] Accès administrateur au repository GitHub
- [ ] Cluster Kubernetes configuré (dev, staging, prod)
- [ ] Compte Scaleway avec Container Registry activé
- [ ] Canal Slack pour notifications (optionnel)

## 🔐 Configuration des Secrets GitHub

### 1. Accéder aux Settings

1. Allez sur votre repository GitHub
2. Cliquez sur **Settings** → **Secrets and variables** → **Actions**
3. Cliquez sur **New repository secret**

### 2. Secrets Obligatoires

#### Container Registry (Scaleway)

```bash
Name: SCW_SECRET_KEY
Value: scw_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Comment obtenir** :
1. Connectez-vous à la console Scaleway
2. Allez dans **Identity and Access Management** → **API Keys**
3. Créez une nouvelle clé API avec les permissions Container Registry
4. Copiez la Secret Key

#### Kubernetes Production

```bash
Name: KUBECONFIG
Value: <contenu_base64_du_kubeconfig>
```

**Comment obtenir** :
```bash
# Encodez votre kubeconfig en base64
cat ~/.kube/config | base64 | pbcopy
```

#### Kubernetes Staging

```bash
Name: KUBECONFIG_STAGING  
Value: <contenu_base64_du_kubeconfig_staging>
```

#### Kubernetes Production (spécifique)

```bash
Name: KUBECONFIG_PROD
Value: <contenu_base64_du_kubeconfig_production>
```

### 3. Secrets Optionnels

#### Notifications Slack

```bash
Name: SLACK_WEBHOOK_URL
Value: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
```

**Comment obtenir** :
1. Allez sur https://api.slack.com/apps
2. Créez une nouvelle app Slack
3. Activez **Incoming Webhooks**
4. Créez un webhook pour votre canal
5. Copiez l'URL du webhook

## 🏗️ Configuration des Environnements Kubernetes

### 1. Namespaces Kubernetes

Créez les namespaces nécessaires sur vos clusters :

```bash
# Development
kubectl create namespace alphabeta808-development

# Staging  
kubectl create namespace alphabeta808-staging

# Production
kubectl create namespace alphabeta808-trading
```

### 2. Secrets Kubernetes

Dans chaque namespace, créez les secrets pour les API :

```bash
# Remplacez les valeurs par vos vraies clés API
kubectl create secret generic trading-secrets \
  --namespace=alphabeta808-trading \
  --from-literal=binance-api-key="your_binance_api_key" \
  --from-literal=binance-api-secret="your_binance_api_secret" \
  --from-literal=webhook-secret="your_webhook_secret" \
  --from-literal=web-admin-user="admin" \
  --from-literal=web-admin-password="secure_password_123"
```

Répétez pour les autres namespaces (`alphabeta808-staging`, `alphabeta808-development`).

### 3. RBAC et ServiceAccount

Appliquez les configurations RBAC :

```bash
kubectl apply -f k8s/rbac.yaml -n alphabeta808-trading
kubectl apply -f k8s/rbac.yaml -n alphabeta808-staging
kubectl apply -f k8s/rbac.yaml -n alphabeta808-development
```

## 🌐 Configuration du Container Registry

### 1. Registry Scaleway

Assurez-vous que votre registry est accessible :

```bash
# Test de connexion au registry
docker login rg.fr-par.scw.cloud/namespace-ecstatic-einstein -u nologin -p <SCW_SECRET_KEY>
```

### 2. Mise à jour des Manifests Kubernetes

Les manifests ont été mis à jour pour utiliser le bon registry. Vérifiez dans :
- `k8s/bot-deployment.yaml`
- `k8s/web-deployment.yaml`

L'image utilisée est maintenant :
```
rg.fr-par.scw.cloud/namespace-ecstatic-einstein/alphabeta808-trading-bot:latest
```

## 🧪 Test de la Configuration

### 1. Test des Secrets

Vérifiez que tous les secrets sont configurés :

```bash
# Via GitHub CLI
gh secret list

# Doit afficher :
# SCW_SECRET_KEY
# KUBECONFIG
# KUBECONFIG_STAGING  
# KUBECONFIG_PROD
# SLACK_WEBHOOK_URL (optionnel)
```

### 2. Test de Build Local

Testez le build Docker localement :

```bash
# Build de l'image
docker build -t test-alphabeta808 .

# Test de l'interface web
docker run -p 5000:5000 test-alphabeta808 python web_interface/app_enhanced.py

# Test du bot (mode dry-run)
docker run test-alphabeta808 python live_trading_bot.py --dry-run
```

### 3. Test du Pipeline

Déclenchez manuellement un workflow pour tester :

1. Allez dans **Actions** → **Staging Deployment**
2. Cliquez sur **Run workflow**
3. Sélectionnez `develop` et `testing`
4. Surveillez l'exécution

## 🚀 Premier Déploiement

### 1. Déploiement Staging

```bash
# Push sur develop pour déclencher staging
git checkout develop
git add .
git commit -m "feat: configure CI/CD pipeline"
git push origin develop
```

### 2. Déploiement Production

```bash
# Merge vers main pour production
git checkout main
git merge develop
git push origin main
```

Ou utilisez le workflow de release :

1. **Actions** → **Release and Production**
2. **Run workflow**
3. Release type: `minor`
4. Environment: `production`

## 📊 Monitoring du Pipeline

### 1. GitHub Actions

Surveillez les workflows dans l'onglet **Actions** :
- ✅ Tests et linting
- ✅ Build Docker
- ✅ Déploiement Kubernetes
- ✅ Health checks

### 2. Kubernetes

Vérifiez les déploiements :

```bash
# Statut des pods
kubectl get pods -n alphabeta808-trading

# Logs en temps réel
kubectl logs -f deployment/trading-bot -n alphabeta808-trading
kubectl logs -f deployment/trading-web-interface -n alphabeta808-trading

# Accès à l'interface web
kubectl port-forward svc/trading-web-service 8080:5000 -n alphabeta808-trading
```

### 3. Slack (si configuré)

Surveillez les notifications dans vos canaux :
- `#deployments` : Statut des déploiements
- `#releases` : Nouvelles releases

## 🛠️ Configuration Avancée

### 1. Environnements GitHub

Configurez les environnements pour protection :

1. **Settings** → **Environments**
2. Créez les environnements :
   - `development`
   - `staging` 
   - `production`
3. Ajoutez des règles de protection pour `production`

### 2. Branch Protection Rules

Protégez les branches importantes :

1. **Settings** → **Branches**
2. Ajoutez des règles pour `main` et `develop` :
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require pull request reviews before merging

### 3. Webhooks Personnalisés

Pour intégrations avancées :

```bash
# Exemple webhook pour monitoring externe
Name: MONITORING_WEBHOOK_URL
Value: https://your-monitoring-system.com/webhooks/deployments
```

## 🔧 Maintenance

### 1. Rotation des Secrets

Planifiez la rotation régulière :
- **API Keys** : Tous les 90 jours
- **Kubeconfig** : Selon votre politique de sécurité
- **Webhooks** : Si compromis

### 2. Mise à jour des Workflows

Les workflows sont versionnés. Pour les mettre à jour :

```bash
# Créez une branche de feature
git checkout -b feature/update-cicd

# Modifiez les workflows
vim .github/workflows/ci-cd.yml

# Testez sur staging
git push origin feature/update-cicd
```

### 3. Surveillance des Quotas

Surveillez :
- **GitHub Actions** : Minutes utilisées
- **Container Registry** : Espace de stockage  
- **Kubernetes** : Ressources utilisées

## 🚨 Dépannage

### Erreur de Secret

```bash
Error: secret "SCW_SECRET_KEY" not found
```

**Solution** : Vérifiez que tous les secrets sont configurés dans GitHub.

### Erreur Kubeconfig

```bash
Error: error loading config file "~/.kube/config": invalid configuration
```

**Solution** : Vérifiez l'encodage base64 du kubeconfig.

### Erreur Registry

```bash
Error: failed to push image: unauthorized
```

**Solution** : Vérifiez les permissions de la clé API Scaleway.

### Échec de Déploiement

```bash
Error: deployment "trading-bot" exceeded its progress deadline
```

**Solution** : Vérifiez les ressources Kubernetes et les health checks.

## 📞 Support

- **Documentation** : [README du pipeline](.github/README.md)
- **Issues** : Créez une issue GitHub avec le label `ci-cd`
- **Logs** : Consultez les logs dans GitHub Actions
- **Monitoring** : Utilisez les dashboards Kubernetes

## 🎯 Prochaines Étapes

Après configuration :

1. [ ] Testez le pipeline complet
2. [ ] Configurez le monitoring avancé
3. [ ] Implémentez les tests d'intégration
4. [ ] Documentez les procédures d'urgence
5. [ ] Formez l'équipe aux outils CI/CD

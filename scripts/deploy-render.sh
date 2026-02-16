#!/bin/bash
# Render Deployment Script
# Deploymant erfolgt automatisch bei Git-Push zu main, aber dieses Skript prüft Pre-Deployment

set -e  # Exit on error

echo "🚀 Enterprise AI Gateway - Render Deployment"
echo "============================================="

# 1. Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml nicht gefunden!"
    exit 1
fi
echo "✅ render.yaml gefunden"

# 2. Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile nicht gefunden!"
    exit 1
fi
echo "✅ Dockerfile gefunden"

# 3. Check if requirements.txt has asyncpg (PostgreSQL support)
if ! grep -q "asyncpg" requirements.txt; then
    echo "❌ asyncpg nicht in requirements.txt — PostgreSQL-Support fehlt!"
    exit 1
fi
echo "✅ PostgreSQL-Support (asyncpg) vorhanden"

# 4. Run tests before deployment
echo ""
echo "🧪 Tests ausführen..."
if command -v pytest &> /dev/null; then
    if ! pytest tests/ -v; then
        echo "❌ Tests fehlgeschlagen! Deployment abgebrochen."
        exit 1
    fi
    echo "✅ Alle Tests bestanden"
else
    echo "⚠️  pytest nicht installiert — Tests übersprungen"
fi

# 5. Check git status
echo ""
echo "📦 Git-Status prüfen..."
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Uncommitted changes gefunden:"
    git status --short
    echo ""
    read -p "Trotzdem fortfahren? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment abgebrochen."
        exit 1
    fi
fi

# 6. Commit and push to trigger Render auto-deploy
echo ""
echo "📤 Push to main branch (löst Render Auto-Deploy aus)..."
git push origin main

echo ""
echo "✅ Deployment-Prozess gestartet!"
echo ""
echo "📋 Nächste Schritte:"
echo "1. Gehe zu https://dashboard.render.com"
echo "2. Setze Secrets in Render Dashboard:"
echo "   - OPENAI_API_KEY"
echo "   - ANTHROPIC_API_KEY"
echo "   - GEMINI_API_KEY"
echo "3. Warte auf Deployment (5-10 Minuten)"
echo "4. Teste Health-Endpoint: https://enterprise-ai-gateway.onrender.com/health"
echo ""
echo "🔗 Render Dashboard: https://dashboard.render.com"

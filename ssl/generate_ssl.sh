#!/bin/bash
# SSL Certificate Generation Script for AlphaBeta808 Trading Bot
# This script generates self-signed certificates for local/development use

set -e

# Create SSL directory if it doesn't exist
mkdir -p /Users/bastienjavaux/Desktop/AlphaBeta808Trading/ssl

# Navigate to SSL directory
cd /Users/bastienjavaux/Desktop/AlphaBeta808Trading/ssl

echo "🔐 Generating SSL certificates for AlphaBeta808 Trading Bot..."

# Generate private key
openssl genrsa -out server.key 2048

# Generate certificate signing request
openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=AlphaBeta808/OU=Trading/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Set proper permissions
chmod 600 server.key
chmod 644 server.crt

echo "✅ SSL certificates generated successfully!"
echo "📁 Files created:"
echo "   - server.key (private key)"
echo "   - server.crt (certificate)"
echo "   - server.csr (certificate signing request)"
echo ""
echo "🔒 For production, replace with certificates from a trusted CA"
echo "🌐 These certificates are valid for 'localhost' domain"

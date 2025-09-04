#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Pull latest changes from git
git pull origin main

echo "Deployment completed successfully!"
#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Pull latest changes from git
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt

# Setup or update cronjob
echo "Setting up cronjob..."
./setup_cron.sh

# Setup web server if not already running
echo "Checking web server..."
if ! systemctl is-active --quiet benchmark-web; then
    echo "Setting up web server for benchmark plots..."
    sudo tee /etc/systemd/system/benchmark-web.service > /dev/null << SERVICE
[Unit]
Description=Benchmark Plot Web Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/data/chap_benchmarking
ExecStart=/usr/bin/python3 -m http.server 8080
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE
    
    sudo systemctl daemon-reload
    sudo systemctl enable benchmark-web
    sudo systemctl start benchmark-web
    echo "✓ Web server running on port 8080"
    echo "Access benchmark plot at: http://$(hostname -I | awk '{print $1}'):8080/benchmark_plot.html"
else
    echo "✓ Web server already running"
fi

echo "Deployment completed successfully!"

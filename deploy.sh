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

# Setup secure web server if not already running
echo "Checking web server..."
if ! systemctl is-active --quiet benchmark-web; then
    echo "Setting up secure web server for benchmark plots (HTML files only)..."
    sudo tee /etc/systemd/system/benchmark-web.service > /dev/null << SERVICE
[Unit]
Description=Secure Benchmark Plot Web Server (HTML only)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/data/chap_benchmarking
ExecStart=/data/chap_benchmarking/.venv/bin/python /data/chap_benchmarking/serve_html_only.py 8080 /data/chap_benchmarking
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE
    
    sudo systemctl daemon-reload
    sudo systemctl enable benchmark-web
    sudo systemctl start benchmark-web
    echo "✓ Secure web server running on port 8080 (serving HTML files only)"
    echo "Access benchmark plot at: http://$(hostname -I | awk '{print $1}'):8080/benchmark_plot.html"
else
    echo "✓ Web server already running"
    echo "Note: If switching from old to secure server, run: sudo systemctl restart benchmark-web"
fi

echo "Deployment completed successfully!"

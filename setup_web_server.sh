#!/bin/bash

echo "=== Setting up web server for benchmark_plot.html ==="
echo ""
echo "Choose your setup method:"
echo ""

# Option 1: Python's built-in HTTP server (simplest)
echo "OPTION 1: Python HTTP Server (simplest, port 8080)"
echo "To start immediately:"
echo "  cd /data/chap_benchmarking && python3 -m http.server 8080"
echo ""
echo "To run as a background service, create systemd service:"
cat << 'EOF'
sudo tee /etc/systemd/system/benchmark-web.service > /dev/null << 'SERVICE'
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
EOF
echo ""
echo "Then access at: http://YOUR_SERVER_IP:8080/benchmark_plot.html"
echo ""
echo "---"
echo ""

# Option 2: Nginx (production-ready)
echo "OPTION 2: Nginx (production-ready, port 80)"
echo "Install nginx if not already installed:"
echo "  sudo apt update && sudo apt install -y nginx"
echo ""
echo "Create nginx config:"
cat << 'EOF'
sudo tee /etc/nginx/sites-available/benchmark > /dev/null << 'NGINX'
server {
    listen 80;
    server_name _;
    
    location / {
        root /data/chap_benchmarking;
        index benchmark_plot.html;
        try_files $uri $uri/ =404;
    }
    
    # Optional: Auto-refresh the page every 5 minutes
    location /benchmark_plot.html {
        root /data/chap_benchmarking;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/benchmark /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
EOF
echo ""
echo "Then access at: http://YOUR_SERVER_IP/benchmark_plot.html"
echo ""
echo "---"
echo ""

# Option 3: Apache (alternative to nginx)
echo "OPTION 3: Apache (alternative to nginx, port 80)"
echo "Install apache if not already installed:"
echo "  sudo apt update && sudo apt install -y apache2"
echo ""
echo "Create apache config:"
cat << 'EOF'
sudo tee /etc/apache2/sites-available/benchmark.conf > /dev/null << 'APACHE'
<VirtualHost *:80>
    DocumentRoot /data/chap_benchmarking
    
    <Directory /data/chap_benchmarking>
        Options Indexes FollowSymLinks
        AllowOverride None
        Require all granted
    </Directory>
</VirtualHost>
APACHE

sudo a2dissite 000-default.conf
sudo a2ensite benchmark.conf
sudo systemctl restart apache2
EOF
echo ""
echo "Then access at: http://YOUR_SERVER_IP/benchmark_plot.html"
echo ""
echo "---"
echo ""

# Quick setup function for Python server
echo "QUICK SETUP - Python server with systemd:"
echo "Run this command to set up Option 1 automatically:"
echo ""
cat << 'SCRIPT'
setup_python_server() {
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
    echo "Access at: http://$(hostname -I | awk '{print $1}'):8080/benchmark_plot.html"
}

# Run the function
setup_python_server
SCRIPT
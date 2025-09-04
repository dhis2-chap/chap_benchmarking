#!/usr/bin/env python3
"""
Secure HTTP server that only serves HTML files from the current directory.
Blocks access to Python files, hidden files, and subdirectories.
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

class HTMLOnlyHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler that only serves HTML files."""
    
    def do_GET(self):
        """Handle GET requests - only serve HTML files."""
        # Parse the requested path
        requested_path = self.path.lstrip('/')
        
        # Default to benchmark_plot.html for root
        if requested_path == '' or requested_path == '/':
            requested_path = 'benchmark_plot.html'
        
        # Security checks
        # 1. Block directory traversal attempts
        if '..' in requested_path:
            self.send_error(403, "Access denied")
            return
        
        # 2. Block subdirectory access (no slashes allowed)
        if '/' in requested_path:
            self.send_error(403, "Subdirectory access denied")
            return
        
        # 3. Block hidden files (starting with .)
        if requested_path.startswith('.'):
            self.send_error(403, "Hidden file access denied")
            return
        
        # 4. Only allow .html files
        if not requested_path.endswith('.html'):
            self.send_error(403, f"Only HTML files are accessible")
            return
        
        # 5. Check if file exists and is in the current directory
        file_path = Path(self.directory) / requested_path
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        
        if not file_path.is_file():
            self.send_error(403, "Not a file")
            return
        
        # Verify the file is actually in the working directory (extra safety)
        try:
            file_path.resolve().relative_to(Path(self.directory).resolve())
        except ValueError:
            self.send_error(403, "Access denied")
            return
        
        # Serve the HTML file
        self.path = '/' + requested_path
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def list_directory(self, path):
        """Disable directory listing."""
        self.send_error(403, "Directory listing disabled")
        return None


def run_server(port=8080, directory=None):
    """Run the secure HTML-only HTTP server."""
    if directory:
        os.chdir(directory)
    
    server_address = ('', port)
    handler = HTMLOnlyHTTPRequestHandler
    handler.directory = os.getcwd()
    
    httpd = HTTPServer(server_address, handler)
    print(f"Serving HTML files only from {os.getcwd()} on port {port}")
    print(f"Access at: http://localhost:{port}/benchmark_plot.html")
    print("Security: Only *.html files in the root directory are accessible")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        sys.exit(0)


if __name__ == '__main__':
    # Use port from command line argument if provided
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    # Use directory from command line argument if provided
    directory = sys.argv[2] if len(sys.argv) > 2 else '/data/chap_benchmarking'
    
    run_server(port, directory)
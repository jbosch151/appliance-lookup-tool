#!/bin/bash
# Stop the Appliance Lookup server

cd "$(dirname "$0")"

if [ -f .server.pid ]; then
    PID=$(cat .server.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✓ Server stopped (PID: $PID)"
    else
        echo "Server is not running (PID: $PID not found)"
    fi
    rm .server.pid
else
    # Fallback: kill by process name
    pkill -f "python3 app_launcher.py"
    echo "✓ Server stopped"
fi

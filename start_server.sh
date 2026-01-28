#!/bin/bash
# Start the Appliance Lookup server in the background

cd "$(dirname "$0")"

# Kill any existing instances
pkill -f "python3 app_launcher.py" 2>/dev/null

# Start server in background, suppressing warnings
nohup python3 app_launcher.py > server.log 2>&1 &

# Get the process ID
SERVER_PID=$!
echo $SERVER_PID > .server.pid

echo "✓ Server started in background (PID: $SERVER_PID)"
echo "✓ Local:   http://127.0.0.1:5001"
echo "✓ Network: http://192.168.1.32:5001"
echo ""
echo "To view logs: tail -f server.log"
echo "To stop: ./stop_server.sh or kill $SERVER_PID"

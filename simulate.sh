# Start the central server
echo "Starting the central server..."
python central_server.py &

# Start the participant servers
echo "Starting the participant servers..."
python client_server.py 1 &
python client_server.py 2 &
python client_server.py 3 &
python client_server.py 4 &
python client_server.py 5 &
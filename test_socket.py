"""
Simple HTTP Server using Flask's built-in test client wrapper
Testing if basic socket binding works on this Windows machine
"""
import socket
import sys

def test_socket():
    """Test if we can bind to a socket at all"""
    print("Testing basic socket binding...")
    
    for port in [5000, 8000, 3000, 9000]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', port))
            sock.listen(1)
            print(f"✓ Successfully bound to port {port}")
            sock.close()
            return port
        except OSError as e:
            print(f"✗ Port {port}: {e}")
            continue
    
    return None

if __name__ == '__main__':
    port = test_socket()
    if port:
        print(f"\n✓ Socket binding works! Using port {port}")
        
        # Now try with Flask
        print("\nStarting Flask app on port", port)
        from app import app
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    else:
        print("\n✗ ERROR: Cannot bind to any port!")
        print("This suggests a system-level issue with network sockets")
        sys.exit(1)

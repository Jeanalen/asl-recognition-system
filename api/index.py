from http.server import BaseHTTPRequestHandler
import json
# Import your existing class
from asl_recognizer import ASLRecognizer 

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # This is a simple test response to show the API is alive
        response = {"status": "success", "message": "ASL Recognizer API is running"}
        self.wfile.write(json.dumps(response).encode('utf-8'))

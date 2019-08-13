from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging


logging.basicConfig(level=logging.INFO)

class HttpServerEngine():
  def __init__(self, config, ml_function):
    HttpServerEngine.config = config
    HttpServerEngine.ml_function = ml_function
    server = HTTPServer(('',HttpServerEngine.config['port']), InferenceServerHandler)
    logging.info("listening on port {}".format(HttpServerEngine.config['port']))
    server.serve_forever()



class InferenceServerHandler(BaseHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()

  def do_POST(self):
    if(self.path == '/inference'):
      self.do_inference()


  def do_inference(self):

    self._set_headers()

    self.data_string = self.rfile.read(int(self.headers['Content-Length'])).decode("utf-8")

    self.data = json.loads(self.data_string)

    #now apply the ML function
    predictions = HttpServerEngine.ml_function(self.data)

    self.wfile.write(json.dumps(predictions).encode('utf-8'))
   

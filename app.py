from flask import Flask,request, make_response,jsonify
from flask_cors import CORS

from ai_service.main import DocumentAI
import os
from dotenv import load_dotenv
import time
import logging
import uuid

logging.basicConfig(level=logging.INFO)


load_dotenv()
chroma_host = os.getenv("CHROMA_HOST", "localhost")
domain = os.getenv("DOMAIN", "http://localhost:3000")

app = Flask("OpenAI-compatible API")
CORS(app, resources={r"/*": {"origins": domain}})


docAI = DocumentAI(host=chroma_host)


#TODO: Enable stream responses
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json(force=True)
    logging.info(f"Received chat completion request: {data}")
    messages = data["messages"]
    if len(messages) > 0:
        data = messages[-1]['content']
    else:
        return make_response(jsonify({"error": "No messages provided"}), 400)
    ai_response = docAI.query(data)
    response = {
        "id": str(uuid.uuid4()),
        "created": int(time.time()),
        "model": "rag-model",
        "choices": [{ "index":0, "message":{'role':'assistant','content': ai_response}, 'finish_reason': 'stop'}],
        "object": "chat.completion",
    }
    return jsonify(response)

@app.get('/engines')
@app.get('/models')
def list_models():
    return make_response(jsonify({
            'data': [{
                'object': 'engine',
                'id': str(uuid.uuid4()),
                'ready': True,
                'owner': 'huggingface',
                'permissions': None,
                'created': None
            }]
        }))

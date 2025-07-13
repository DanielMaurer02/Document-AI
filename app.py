from flask import Flask,request, make_response,jsonify
from flask_cors import CORS

from ai_service.main import DocumentAI
import os
from dotenv import load_dotenv
import uuid
import time
import uvicorn

load_dotenv()
host = os.getenv("CHROMA_HOST", "localhost")

app = Flask("OpenAI-compatible API")
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


docAI = DocumentAI(host=host)


#TODO: Enable stream responses
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json(force=True)
    print(data)
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
                'id': id,
                'ready': True,
                'owner': 'huggingface',
                'permissions': None,
                'created': None
            }]
        }))


if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=4200)
    app.run(host="0.0.0.0", port=4200, debug=True)


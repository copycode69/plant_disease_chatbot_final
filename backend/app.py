from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_logic import get_bot_reply

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    reply = get_bot_reply(query)
    return jsonify({"response": reply})

if __name__ == "__main__":
    # To allow access from other devices on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)

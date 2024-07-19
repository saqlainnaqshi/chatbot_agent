from flask import Flask, request, jsonify
from chatbot import get_response
from vector_store import set_vector_store
import requests
import json

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
async def chat():
  try:
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
      return jsonify({"error": "No message provided"}), 400

    response = await get_response(user_message)
    return jsonify({"response": response})
  except Exception as e:
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/quick-replies', methods=['GET'])
def get_quick_replies():
    quick_replies = {
        "type": "radio",
        "keepIt": True,
        "values": [
            {
                "title": "Where is gulmarg",
                "value": "Where is gulmarg"
            },
            {
                "title": "Where is Kashmir",
                "value": "Where is Kashmir"
            },
            {
                "title": "Which places should i visit in kashmir",
                "value": "Which places should i visit in kashmir"
            }
        ]
    }
    return jsonify(quick_replies)

@app.route('/create-vector-store', methods=['POST'])
def create_vector_store():
    try:
        response = requests.get('https://kashmirbookings.in/api/tour/search')
        response.raise_for_status()  

        data = response.json()

        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        set_vector_store()

        return jsonify({"message": "Vector store created successfully"}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7001)

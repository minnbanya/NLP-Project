from flask import Flask, render_template, request, jsonify
from models_utils import load_model
from utils import chat_answer
import json

app = Flask(__name__)

chat_history = []
model_name   = 't5' #'gpt2_no_cpu' #'gpt2' # 't5'
temp         = 0.5
rep          = 1.2

llm = load_model(model_name, temp=temp, rep=rep)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Get bot response using the chatbot model
    
    bot_response = chat_answer(user_message, llm)['answer']
    src_doc      = chat_answer(user_message, llm)['source_documents']

    # Store user message and bot response in chat history
    chat_history.append({'user': user_message, 'bot': bot_response})

    product_list = []

    for doc in src_doc:
        product_info = json.loads(doc.page_content)
        product_list.append(product_info)

    return jsonify({'bot_response': bot_response, 'product_list':product_list})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

if __name__ == '__main__':
    app.run(debug=True)
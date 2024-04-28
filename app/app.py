from flask import Flask, render_template, request, jsonify
from models_utils import load_model
from utils import chat_answer
import json

app = Flask(__name__)

chat_history = []
model_name   = 'flan_t5' #'t5' #'gpt2_no_cpu' #'gpt2' # 't5'
temp         = 0
rep          = 1.5

llm = load_model(model_name, temp=temp, rep=rep)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Get bot response using the chatbot model
    
    bot_response = chat_answer(user_message, llm)
    answer       = bot_response['answer']
    src_doc      = bot_response['source_documents']

    print(bot_response)

    # Store user message and bot response in chat history
    chat_history.append({'user': user_message, 'bot': answer})

    product_list = []

    for doc in src_doc:
        product_info = json.loads(doc.page_content)
        product_list.append(product_info)

    return jsonify({'bot_response': answer, 'product_list':product_list})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

if __name__ == '__main__':
    app.run(debug=True)
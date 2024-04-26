from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from torchtext.data.utils import get_tokenizer
import dill
import re
import torch
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

#locate vectorstore
vector_path = './vector_stores'
if not os.path.exists(vector_path):
    os.makedirs(vector_path)
    print('create path done')

def predict(text_str):
    text_str = text_str.lower()
    device = 'cpu'
    regex_s = re.sub("\\(.+?\\)|[\r\n|\n\r]|!", "", text_str)
    text = " ".join(regex_s.split())
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    loaded_model = torch.jit.load('../question_classification/CNN.pt')
    with open('../question_classification/vocab.pkl', 'rb') as f:
        loaded_vocab = dill.load(f)
    text = torch.tensor(loaded_vocab(tokenizer(text))).to(device)
    text = text.reshape(1, -1)
    with torch.no_grad():
        output = loaded_model(text).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return predicted.item()
    
categories = [
    'Toys_and_Games', 'Health_and_Personal_Care', 'Cell_Phones_and_Accessories', 
    'Home_and_Kitchen', 'Musical_Instruments', 'Baby_Products', 'Sports_and_Outdoors', 
    'Patio_Lawn_and_Garden', 'Video_Games', 'Pet_Supplies', 'Tools_and_Home_Improvement', 
    'Beauty_and_Personal_Care', 'Electronics', 'Automotive', 'Office_Products', 
    'Amazon_Fashion'
]

def choose_vector_store(text, size):

    print(predict(text))
    category = categories[predict(text)]
    #calling vector from local
    vector_path = './vector_stores'

    db_file_name = f"{size}/{category}"

    vectordb = FAISS.load_local(
        folder_path = os.path.join(vector_path, db_file_name),
        embeddings = embedding_model,
        index_name = f'{category}' #default index
    )
    retriever = vectordb.as_retriever()

    return retriever

def create_chain(llm, retriever):
    question_generator = LLMChain(
        llm = llm,
        prompt = CONDENSE_QUESTION_PROMPT,
        verbose = True
    )

    prompt_template = """
        Test prompt for NLP Amazon sales chatbot.
        {context}
        Question: {question}
        Answer:
        """.strip()

    PROMPT = PromptTemplate.from_template(
        template = prompt_template
    )

    PROMPT
    #using str.format 
    #The placeholder is defined using curly brackets: {} {}
    doc_chain = load_qa_chain(
        llm = llm,
        chain_type = 'stuff',
        prompt = PROMPT,
        verbose = True
    )

    memory = ConversationBufferWindowMemory(
        k=1, 
        memory_key = "chat_history",
        return_messages = True,
        output_key = 'answer'
    )

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory=memory,
        verbose=True,
        get_chat_history=lambda h : h
    )
    
    return chain

def chat_answer(prompt_question, llm):
    torch.cuda.empty_cache()
    retriever = choose_vector_store(prompt_question, 100)
    chain = create_chain(llm, retriever)
    answer = chain({"question":prompt_question})

    return answer





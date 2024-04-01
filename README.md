# NLP-Project

# Paper Summaries: 

## 1. Active Retrieval Augmented Generation

**Authors:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu,
Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig

**Link:** https://aclanthology.org/2023.emnlp-main.495/

**Year:** 2023

**Citations:** 104

**Read by:** Minn Banya

FLARE (Forward-Looking Active Retrieval augmented generation) introduces an innovative method for text generation by utilizing forward prediction of upcoming sentences to retrieve relevant documents for sentence regeneration. It compares direct retrieval, based on confidence levels, with instructed retrieval, favoring the former for its superior performance. Optimal confidence values for retrieval range between 40% and 80%. Additionally, masking low-probability tokens during direct retrieval using a threshold value, beta, proves more effective than retrieving entire sentences. FLARE shows promise in enhancing text generation tasks through the integration of active retrieval mechanisms.

## 2. Lingke: A Fine-grained Multi-turn Chatbot for Customer Service

**Authors:** Pengfei Zhu, Zhuosheng Zhang, Jiangtong Li, Yafang Huang, Hai Zhao

**Link:** https://aclanthology.org/C18-2024/

**Year:** 2018

**Citations:** 51

**Read by:** Minn Banya

Linke aims to improve chatbot performance by following the 6 steps below:

1. Coreference Resolution and Document Separation: Implements `Stanford CoreNLP` for coreference resolution and document segmentation, resulting in a list of sentences (A).

2. Target Sentences Retrieval: Selects k relevant sentences (2 in this study) from sentence collection (A) using `Apache Lucene`.

3. Candidate Responses Generation: Extracts subject-verb-object (SVO) triples from selected sentences using the `ReVerb` framework, generating simple sentences (T). Candidate responses are formed by the union between relevant sentences (E) and simple sentences (T).

Dialogue Manager:

4. Self-matching Attention: Employs GRUs to convert conversation utterances and candidate responses into word embeddings. Self-matching attention strategy filters redundant information.

5. Response Selection: Matches each candidate response with conversation utterances in chronological order, selecting the final response based on accumulated matching score.

6. Chit-chat Response Generation: Utilizes an attention-based seq2seq model to generate conversational responses when user questions are irrelevant to the current database (matching score of lower than 0.3 for all candidate responses).

## 3. Improving the Domain Adaptation of Retrieval Augmented Generation
(RAG) Models for Open Domain Question Answering

**Authors:** Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen,
Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara

**Link:** https://aclanthology.org/2023.tacl-1.1/

**Year:** 2023

**Citations:** 38

**Read by:** Thamakorn Kiattikaikul

The paper introduces Retrieval Augment Generation (RAG) which is the process enhanced in Open-Domain Question Answering (ODQA). However, RAG is not optimized for other specific domains since it was trained on Wikipedia-based knowledge. In this paper, authors propose RAG-end2end, an extension to RAG, for domain adaptation in ODQA. It updates all components of the external knowledge base during training and introduces an additional training signal to inject domain-specific knowledge. Unlike RAG, RAG-end2end jointly trains the retriever and generator for the end QA task and domain adaptation. This approach is evaluated on COVID-19, News, and Conversations datasets, showing significant performance improvements over the original RAG model. The proposed RAG-end2end model aims to finetune all components, including the DPR retriever, and dynamically update the external knowledge base during training to facilitate domain-specific question answering. Experiments on these datasets demonstrate significant improvements in performance metrics like exact match, F1, and retrieval accuracy with RAG-end2end compared to RAG-original-QA. Adding the statement-reconstruction task can enhance domain adaptation. RAG-end2end also outperforms standalone Dense Passage Retrieval (DPR) fine tuning with domain-specific data. Initializing RAG with domain-adapted DPR prior to fine tuning yields improvements, but RAG-end2end training still performs better.

## 4. SuperAgent: A Customer Service Chatbot for E-commerce Websites

**Authors:** Lei Cui, Shaohan Huang, Furu Wei, Chuanqi Tan, Chaoqun Duan, and Ming Zhou

**Link:** https://aclanthology.org/P17-4017/

**Year:** 2017

**Citations:** 424

**Read by:** Thamakorn Kiattikaikul

The authors demonstrated SuperAgent, an add-on web browser extension customer service chatbot. The chatbot consists of 3 sub-engines: 1) a fact question answering engine for Product Information; 2) an FAQ search engine for QA; 3) an opinion mining & text question engine in Customer Reviews. The benefits is they don't need to deploy the website, data update easily, and different engines can process parallel (between these 3 engines). The result shows that the chatbot is able to answer specific questions, conduct FAQ search from QA pairs in customer review section, and reply to chit chat conversations.

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

## 5. Question Classification using Head Words and their Hypernyms

**Authors:** Zhiheng Huang, Marcus Thint, and Zengchang Qin

**Link:** https://aclanthology.org/D08-1097/

**Year:** 2008

**Citations:** 269

**Read by:** Noppawee Teeraratchanon
This paper is about Question Classification to classify the questions and narrow down the search space to identify the correct answer. They used 2 models to compare which are `LIBSVM` and `Stanford Maximum Entropy`. Moreover, they used 5 different features to deal with questions including:
1.	Question wh-word: to separate the question into what, which, when, where, who, how, why, and rest type.
2.	Heas words: to focus on identifying the single word that carries the most meaning or significance Instead of considering the entire question.
3.	WordNet Semantic:
    &nbsp; 3.1 Direct hypernym: use WordNet to assign the broader categories or concepts of words. For example, "animal" is a hypernym of "dog" and "cat."
    &nbsp; 3.2 Indirect hypernym: use WordNet to find the similarity between the head word on such question and the description word in question categories. The question category that has the highest similarity is marked for head word as a feature.
4.	N-gram: to separate the question into chunks. In this paper, they focus on unigram, bigram, and trigram.
5.	Word shape: to classify the tokens into all upper case, all lower case, mixed case, all digits, and other categories.
To evaluate, First, the paper tests SVM and ME models with individual features. As a result, the wh-word and head word is the best among individual features. Direct hypernym is better than indirect hypernym. Unigram is also better than bigram and trigram. Next, they evaluate the model with a combined feature from baseline (wh-word and head word), and they can conclude that the linear SVM and ME with wh-word, head word, direct hypernym, unigram, and word shape are the best with 89.2% and 89.0% respectively.

## 6. AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine

**Authors:** Minghui Qiu, Feng-Lin Li, Siyu Wang, Xing Gao, Yan Chen, Weipeng Zhao, Haiqing Chen, Jun Huang, Wei Chu

**Link:** https://aclanthology.org/P17-2079/ 

**Year:** 2017

**Citations:** 232

**Read by:** Noppawee Teeraratchanon
The paper is about a chatbot called AliMe Chat that's really good at answering questions and having conversations. It introduced a new method for improving chatbots by combining 3 models including:
1.	Information Retrieval (IR) model is used for retrieving the number of sets of candidate QA pairs from the database that has the top similarity score between the input question and the retrieved question calculated by BM25.
2.	Rerank model (Seq2Seq) is used for calculating the confidence score between the input question and all candidate answers and selecting the answer that has the highest score. If the score is higher than the threshold, the chatbot takes the answer.
3.	Generation based model (Seq2Seq with GRU) is used for generating the answer from scratch based on context if the score from the Rerank model is lower than the threshold.
By combining 3 models, it handles the limitations of the IR model (struggle with less common questions and difficulty to handle with long-tail questions) and the Seq2Seq based generation model (sometimes gives inconsistent or nonsensical answers). Moreover, the paper compared IR, Generation, IR + Rerank, and IR + Rerank + Generation. They concluded that IR + Rerank + Generation got the highest accuracy.

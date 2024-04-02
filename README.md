# NLP-Project : Sales Chatbot for Amazon

## Introduction to our team
Allow us to introduce the exceptional individuals who form the backbone of our project team:
- Minn Banya                (st124145)
- Wut Yee Aung              (st124377)
- Kyi Thin Nu               (st124087)
- Thamakorn Kiattikaikul    (st124481)
- Noppawee Teeraratchanon   (st124482)

## Project Background: The Rise of Sales Chatbots for Amazon

Sales chatbots, specifically tailored for Amazon, are AI-driven virtual assistants designed to engage customers in conversational interactions, guide them through the Amazon sales process, and facilitate purchases.

Key Factors Driving Adoption:

- 24/7 Availability: Provides instant assistance to Amazon customers round-the-clock.
- Scalability: Handles multiple conversations on Amazon simultaneously without compromising quality.
- Enhanced Customer Experience: Delivers personalized recommendations based on Amazon customer preferences and browsing history.
- Cost-Efficiency: Reduces operational expenses for Amazon sellers by automating routine sales tasks.
- Data Insights: Generates valuable data for Amazon sellers to analyze customer behavior and refine sales strategies.<br>
In summary, sales chatbots tailored for Amazon offer sellers a modern and effective solution for enhancing customer engagement, driving sales, and gaining valuable insights into Amazon customer preferences and behavior.

## Problems and Motivation
#### Problem:
In the realm of online sales, two critical challenges emerge. 
    - Firstly, customers frequently abandon the sales process due to a lack of immediate assistance or the inability to obtain answers swiftly. 
    - Secondly, customers often seek accurate information and may pose follow-up questions that conventional pre-programmed chatbots cannot sufficiently address. This deficiency leads to customer frustration and potential loss of sales as they turn elsewhere for assistance or abandon their purchases altogether. Traditional chatbots, constrained by their inability to adapt to nuanced inquiries, struggle to provide satisfactory responses in these scenarios.

#### Motivation:
Addressing these challenges is imperative to bridge the gap in customer service and ensure a seamless and satisfying shopping experience. By tackling this issue head-on, businesses can enhance customer trust, foster loyalty, and ultimately drive sales growth on platforms such as Amazon. The motivation stems from the desire to increase sales, elevate customer satisfaction levels, and alleviate the workload on human sales representatives. A sales chatbot equipped to handle nuanced inquiries and provide accurate information in real-time presents a promising solution to these pressing issues.

## Solution Requirements
- Natural Language Processing (NLP): Ability to understand and respond to customer inquiries in natural language.
- Faster and more accurate information retrieval
- Preprocessing the database by separating into labeled bins/ vector stores
- Classifier to determine which vector store the LLM should retrieve information from based on user input
- Designing a suitable prompt template
- Instruction tuning the LLM on Amazon QA data
- Experiments carried out to determine the best model for both classifier and LLM
- Adding additional components to the models, such as Active RAG
- Ablation study to determine which components have the most impact
- Web application that can serve as a chat window for the customer

## Architecture of the Solution
- Data Source and Preprocessing
- NLP
    - Classification Part 
        - MLflow will be used to save the models and see the performance of the models (select the best performance model as our best model)
    - LLM Part
        - Database will be split into vector stores
        - MLflow will be used for experiment logging and model integration
        - Instruction tuning tested and included if there is a positive impact on the performance
- Design test cases and evaluations for the system’s performance
- Web Design
    - Frontend: User interface for interacting with the chatbot, possibly through a website or messaging platform.
    - will update according to Ma Wut Yee

## Dataset Information

We will use the dataset from:
- Amazon product dataset (small) - https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020
- Amazon product dataset (large) - https://www.kaggle.com/datasets/piyushjain16/amazon-product-data/data
- Amazon QA dataset - https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/ 

We acknowledge and appreciate the creators and contributors of the dataset for making it available for public use.

## Task Distribution
- Data Collection  - Puay and Kyi
- Modeling	
    - Classification - Nopawee
    - Language Model - Minn
- Web Application - Wut Yee
- Experiment Testing - Puay and Kyi
- Reporting	and Others - Puay and Kyi

# Paper Summaries: 

## 1. Active Retrieval Augmented Generation

**Authors:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu,
Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig

**Link:** https://aclanthology.org/2023.emnlp-main.495/

**Year:** 2023

**Citations:** 104

**Read by:** Minn Banya

FLARE (Forward-Looking Active Retrieval augmented generation) presents a novel approach to text generation by leveraging forward prediction to anticipate upcoming sentences and retrieve pertinent documents for sentence generation. It conducts a comparative analysis between direct retrieval, based on confidence levels, and instructed retrieval, with direct retrieval showing superior performance. The study identifies optimal confidence values for retrieval falling within the range of 40% to 80%. Furthermore, the technique of masking low-probability tokens during direct retrieval using a threshold value, beta, is demonstrated to be more effective than retrieving entire sentences. Overall, FLARE demonstrates considerable potential in enhancing text generation tasks through the incorporation of active retrieval mechanisms, offering a nuanced understanding of the interplay between forward prediction and document retrieval in the generation process

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
<br>
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
<br>
By combining 3 models, it handles the limitations of the IR model (struggle with less common questions and difficulty to handle with long-tail questions) and the Seq2Seq based generation model (sometimes gives inconsistent or nonsensical answers). Moreover, the paper compared IR, Generation, IR + Rerank, and IR + Rerank + Generation. They concluded that IR + Rerank + Generation got the highest accuracy.

## 7. Bot2Vec: Learning Representations of Chatbots

**Authors:** Jonathan Herzig, Tommy Sandbank, Michal Shmueli-Scheuer
David Konopnicki1 John Richards

**Link:** https://aclanthology.org/S19-1009.pdf

**Year:** 2019

**Citations:** 1

**Read by:** Kyi Thin Nu

BOT2VEC is a framework for learning vector representations of conversational bots using neural networks. The framework offers two approaches: content-based (BOT2VEC-C) and structure-based (BOT2VEC-S) representations. Conversations are represented either by textual content or by analyzing the structure of bot graphs. The framework is evaluated on two classification tasks: detecting production bots and identifying egregious conversations. Experimental results demonstrate the effectiveness of BOT2VEC representations in improving classification performance compared to baseline models. Additionally, the analysis reveals semantic similarities between bots within the same domain based on their structure representations.

Keywords: Conversational Bots, Representation Learning, BOT2VEC, Neural Networks, Classification Tasks, GitHub Repository.

## 8. Conversational Recommender System and Large Language Model Are Made for Each Other in E-commerce Pre-sales Dialogue

**Authors:** Yuanxing Liu, Weinan Zhang, Yifan Chen, Yuchi Zhang, Haopeng Bai, Fan Feng, Hengbin Cui, Yongbin Li, Wanxiang Che

**Link:** https://aclanthology.org/2023.findings-emnlp.643/

**Year:** 2023

**Citations:** 106

**Read by:** Kyi Thin Nu

This paper explores the synergistic integration of Conversational Recommender Systems (CRS) and Large Language Models (LLMs) in e-commerce pre-sales dialogues. By combining CRS, which suggest products based on user preferences, with LLMs, capable of generating human-like responses, the system aims to enhance the shopping experience. The study investigates how the joint utilization of these technologies improves recommendation accuracy and user satisfaction. Experimental results demonstrate the effectiveness of the integrated approach in facilitating meaningful and personalized interactions between users and e-commerce platforms, leading to increased engagement and conversion rates.

Keywords: Conversational Recommender System, Large Language Model, E-commerce, Pre-sales Dialogue, Personalization, User Engagement.

## Acknowledgments

We extend our heartfelt gratitude to our dedicated team for their unwavering support and guidance throughout the project. Their invaluable contributions have been instrumental in the development and testing of this project. We would also like to express our special thanks to Minn, Wut Yee, Kyi, Puay and Nopawee for their exceptional efforts and commitment to ensuring the success of this endeavor.
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

## Background and Related Work
1. Chatbots in E-commerce:
    In recent years, chatbots have become integral to e-commerce, serving as 24/7 virtual sales assistants. With AI and NLP, they offer personalized support, recommend products, and streamline the sales process. By automating tasks, they boost efficiency and scalability, while gathering valuable customer insights. Chatbots not only enhance the shopping experience but also drive sales growth, making them indispensable tools for modern e-commerce businesses.

    Successful e-commerce chatbots, like Sephora's Virtual Artist and eBay's ShopBot, personalize shopping experiences by offering product recommendations and style advice. They streamline the purchasing process, increasing sales conversion rates by providing instant assistance and addressing customer queries promptly. These chatbots enhance customer engagement, foster loyalty, and contribute to revenue growth, showcasing the transformative impact of AI-driven virtual assistants in the e-commerce landscape.

2. Natural Language Processing (NLP):
    NLP techniques enable chatbots to comprehend and generate natural language responses. These include intent recognition, entity recognition, and sentiment analysis. By parsing user input, chatbots identify user intentions and extract relevant information to formulate appropriate responses. Advanced techniques like machine translation and summarization enhance language understanding and generation capabilities. NLP empowers conversational chatbots to engage users in meaningful interactions, providing personalized assistance and recommendations. This ensures a seamless and natural dialogue between users and chatbots, driving improved user experience and satisfaction.

    NLP is pivotal for chatbots in deciphering customer queries, extracting pertinent details, and delivering precise responses. Techniques like intent recognition classify user requests into actionable categories, while entity recognition identifies key information such as product names or dates. Sentiment analysis gauges user emotions, enabling chatbots to tailor responses accordingly. These NLP capabilities empower chatbots to understand complex language nuances, adapt to varying user intents, and provide contextually relevant answers. By leveraging NLP, chatbots optimize user interactions, streamline information retrieval, and enhance overall user satisfaction by delivering accurate and timely assistance.

3. Classification Models:
    Classification models categorize data into predefined classes or categories based on input features. In the context of customer queries in e-commerce, these models analyze text inputs to classify them into relevant product categories or topics. For example, a classification model trained on past customer queries can identify whether a query pertains to electronics, clothing, or home goods. By learning patterns in the text data, the model assigns probabilities to each category, enabling it to classify new queries accurately. This application of classification models streamlines customer support processes, improves query routing, and enhances the overall user experience in e-commerce platforms.
    - Logistic Regression:
        - Strengths: Simple and computationally efficient, making it easy to implement and interpret. It provides probabilistic outputs, which are useful for understanding the certainty of predictions. Works well with linearly separable data and handles binary classification tasks effectively.
        - Limitations: Assumes a linear relationship between input features and the log-odds of the response variable. May struggle with non-linear relationships and complex datasets. Not suitable for tasks with multiple class labels without modification.
    - Decision Trees:
        - Strengths: Easy to understand and visualize, making it ideal for explaining model decisions. Can handle both numerical and categorical data. Automatically selects the most informative features and can capture non-linear relationships.
        - Limitations: Prone to overfitting, especially with deep trees and noisy data. Sensitive to small variations in the training data, which can lead to different tree structures. Decision trees are relatively unstable and may require ensemble methods like random forests to improve performance.
    - Support Vector Machines (SVM):
        - Strengths: Effective in high-dimensional spaces and with datasets that have clear margins of separation. Works well with both linear and non-linear data through the use of different kernel functions. Robust against overfitting, especially in high-dimensional spaces.
        - Limitations: Can be computationally expensive, especially with large datasets. Requires careful selection of kernel functions and tuning of hyperparameters. SVMs perform poorly with datasets that have a large number of noise points or overlapping classes.

4. Language Generation Models (LLMs):
    Language generation models like GPT-3 utilize transformer architectures to produce human-like text based on input prompts. Trained on vast datasets, GPT-3 excels in generating coherent and contextually relevant responses across diverse topics. With 175 billion parameters, it captures intricate language patterns, demonstrating fluency and creativity. Its versatility enables applications in content generation, chatbots, translation, and code synthesis. GPT-3's remarkable ability to mimic human writing style and produce high-quality text marks a significant advancement in natural language processing.

    Advancements in Language Generation Models (LLMs) like GPT-3 enhance chatbots' conversational capabilities significantly. LLMs can generate detailed product descriptions by analyzing various sources, personalize recommendations based on user preferences, and provide accurate responses to customer inquiries. By understanding context and synthesizing information, LLM-powered chatbots offer more engaging and tailored interactions, driving improved customer satisfaction and sales conversion rates in e-commerce.

5. Existing Research and Implementations:
    E-commerce platforms like eBay and Sephora have deployed chatbots to enhance customer experiences. eBay's ShopBot assists users in finding products through natural language queries, receiving positive feedback for its ease of use and helpfulness. Sephora's Virtual Artist recommends makeup products based on user preferences, receiving acclaim for its personalized recommendations and interactive features. Both chatbots have contributed to increased user engagement, higher conversion rates, and improved customer satisfaction, showcasing the effectiveness of AI-driven virtual assistants in e-commerce.

6. Challenges and Opportunities:
    Common challenges in developing and deploying e-commerce chatbots include:
    - Data Privacy Concerns: Ensuring compliance with data protection regulations and safeguarding sensitive customer information.
    - Language Understanding Limitations: Addressing the complexity of natural language queries, slang, and ambiguous intents to improve chatbot comprehension.
    - User Acceptance: Overcoming skepticism and building trust in chatbot capabilities through effective communication and user education.
    - Integration with Existing Systems: Seamlessly integrating chatbots with e-commerce platforms, CRM systems, and backend databases to access real-time data and provide accurate responses.
    - Scalability and Maintenance: Scaling chatbot capabilities to handle increasing user volumes and maintaining performance over time through regular updates and improvements.
    - User Experience Design: Designing intuitive and user-friendly interfaces that guide users through interactions and provide value-added experiences.
    - Training Data Quality: Ensuring high-quality training data to improve chatbot accuracy and minimize errors in understanding user intents and preferences.
    - Multilingual Support: Addressing language diversity to cater to global audiences and provide consistent support in multiple languages.

    Opportunities for innovation in leveraging chatbots for e-commerce include:
    - Personalization: Tailoring product recommendations and offers based on user preferences and browsing history.
    - Conversational Commerce: Enabling seamless transactions and order management through natural language interactions.
    - Visual Search: Allowing users to search for products using images, enhancing the shopping experience.
    - Proactive Assistance: Anticipating user needs and providing proactive support and recommendations.
    - Integration with Social Media: Leveraging chatbots on social platforms for shopping and customer support, tapping into wider audience reach.
    - Gamification: Introducing gamified elements to engage users and incentivize purchases, fostering customer loyalty and retention.

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
    - Classification Part
        - MLflow will be used to save the models and see the performance of the models (select the best performance model as our best model)
    - LLM Part
        - Database will be split into vector stores
        - MLflow will be used for experiment logging and model integration
        - Instruction tuning tested and included if there is a positive impact on the performance
- Design test cases and evaluations for the system’s performance
- Web Design
    - Flask as backend framework to handle interaction between user requests and model responses
    - HTML as frontend framework for the user interface.
    - MLflow for managing machine learning models including experiment tracking, model versioning, and model serving.
    - GitHub will manage codebase changes while also tracking modifications in application code changes.
    - Flask as backend framework to handle interaction between user requests and model responses.
HTML as frontend framework for the user interface.
MLflow for managing machine learning models including experiment tracking, and model versioning.
GitHub will manage codebase changes while also tracking modifications in application code changes.

## Dataset Information

We will use the dataset from:
- Amazon product dataset (small) - https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020
- Amazon product dataset (large) - https://www.kaggle.com/datasets/piyushjain16/amazon-product-data/data
- Amazon QA dataset - https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/ 

We acknowledge and appreciate the creators and contributors of the dataset for making it available for public use.

## Task Distribution
- Data Collection  - Puay and Kyi
- Modeling	
    - Classification - Noppawee
    - Language Model - Minn
- Web Application - Wut Yee
- Experiment Testing - Puay and Kyi
- Reporting	and Others - Puay and Kyi

## Progress Update
### Progress on classification model
- Categorize into 17 classes
  - Toys_and_Games
  - Health_and_Personal_Care
  - Cell_Phones_and_Accessories
  - Home_and_Kitchen
  - Musical_Instruments
  - Baby, Sports_and_Outdoors
  - Patio_Lawn_and_Garden
  - Video_Games
  - Pet_Supplies
  - Tools_and_Home_Improvement
  - Beauty
  - Electronics
  - Grocery_and_Gourmet_Food
  - Automotive
  - Office_Products
  - Clothing_Shoes_and_Jewelry
- We have trained 2 models (biLSTM and CNN) on different parameter with small (500 questions per product category) and large dataset (1000 questions per product category). Moreover, we saved the model when the validation loss is improve. Therefore, we select the best model with specific parameter based on minimum validation loss

Github link for our experiment: https://github.com/Noppawee-Teeraratchanon/NLP_Project_Question_Classification_By_ProductType.git

#### biLSTM with small dataset

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/biLSTM500_1.png)

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/biLSTM500_2.png)



#### biLSTM with large dataset

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/biLSTM1000_1.png)

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/biLSTM1000_2.png)


#### CNN with small dataset

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/CNN500.png)


#### CNN with large dataset

![Alt Text](https://github.com/minnbanya/NLP-Project/blob/main/question_classification/question_classification_mlflow_image/CNN1000.png)




From the experiment, we can conclude that both models are better when the dataset is larger. For biLSTM, when we increase the epoch, hidden dimension, and number of layers, the model is worse. For CNN, we can conclude that when the number of filters and epoch increase, the model is better
<br>

Based on minimum validation loss, CNN with 30 epochs, 150 number of filters is our best model and we will use it to train with a larger dataset (2000 questions per category)

- We have tested model with test set and the model got around 44% accuracy

### Progress on web application

- Chat Interface:
    - Functionality:
        - Allows users to type messages (Type your message...), send them, and receive responses from the chatbot.
        - Messages are displayed in a visually distinct format, with user messages on the right and bot responses on the left.

- Additional Sections:
    - References:
        - Contains detailed information about the product.
    - Product Review:
        - Displays product review content for user reference.
    - Product Link:
        - Provides a link for users to access more detailed product information.

![Alt Text](figures/progress_webapp.png)

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

## 9. CHAI: A CHatbot AI for Task-Oriented Dialogue with Offline Reinforcement Learning

**Authors:** Siddharth Verma, Justin Fu, Mengjiao Yang, Sergey Levine

**Link:** https://aclanthology.org/2022.naacl-main.332.pdf

**Year:** 2022

**Citations:** 30

**Read by:** Wut Yee Aung

This paper study how offline reinforcement learning can instead be used to train dialogue agents entirely using static datasets collected from human speakers. Their experiments show that recently developed offline RL methods can be combined with language models to yield realistic dialogue agents that better accomplish task goals.

**Methodology:**

- Reinforcement Learning Setup:
    - This study formulate the task-oriented dialogue problem as an RL problem, where the agent serves the role of seller and the environment serves the role as the buyer. It utilizes offline RL, which leverages pre-collected datasets of human conversations.
    - The problem is defined as a Markov decision process, with states, actions, transition distribution, and reward function.
        - States (**S**) include information like action type, utterance content, normalized price related to list price, and context like ad description.
        - Actions (**A**) involve sending messages or offers with proposed prices.
        - Transition Function (**T**) determines how likely different responses are after each action taken by our seller bot.
        - Reward Function (**R**) : Chatbot receives rewards based on how well it negotiates:
            - If it sells an item, it gets points proportional to the selling price (normalized against list price and multiplied by ten).
            - If its offer gets rejected, it loses points (a penalty of -20) to push it toward making successful deals.
    - The RL objective is to find a policy that maximizes the expected returns over time. It can be done either through online RL (interacting with the environment) or offline RL (using pre-collected interactions).
- Offline Reinforcement Learning with Language Models:
    - Proposed approach combines offline RL with fine-tuned language models.
    - The approached begins with training a language model, such as GPT-2 and fine-tuning it on task-specific dialogue corpus.
    - Then train a critic or Q-function which is responsible for scoring good and bad responses and is used to select responses from a pool of candidates generated from the language model. Implement 3 Q-learning framework for training and compare.
        - Proposal Sampling (CHAI-prop)
        - Conservative Q-Learning (CHAI-CQL)
        - Behavior-Regularized Q-Learning (CHAI-BRAC)
- Dialogue Generation from trained Q-function:
    - Finetuned language model is used to generate candidate responses conditioned on the conversation history, along with sampled prices.
    - Each action is scored using a critic.
    - Final response is returned by sampling the Q-value scores.

**Dataset and Evaluation:**
- CraigslistBargain dataset which consists of real dialogues between buyers and sellers on Craigslist.
- CHAI is evaluated on a negotiation scenario using this dataset.
- The seller role is played by CHAI, while the buyer is simulated by the environment.

**Results and Analysis**

- CHAI combines the strengths of two techniques:
    - Offline Reinforcement Learning (RL): allows CHAI to learn from pre-collected datasets of human conversations, making it efficient and avoiding the need for real-time interaction.
    - Language Models (LMs): This enables CHAI to generate natural and grammatically correct language during conversations.
- CHAI gives both descriptive responses to questions and reasonable bargaining behavior, whereas the retrieval-based agent only shows good bargaining behavior, and the language model agent only gives descriptive responses.

## 10. SalesBot: Transitioning from Chit-Chat to Task-Oriented Dialogues

**Authors:** Ssu Chiu,  Maolin Li, Yen-Ting Lin, Yun-Nung Chen

**Link:** https://aclanthology.org/2022.acl-long.425.pdf

**Year:** 2022

**Citations:** 25

**Read by:** Wut Yee Aung

This paper focuses on investigating the conversations starting from open-domain social chatting and then gradually transitioning to task oriented purposes, and releases a large-scale dataset with detailed annotations for encouraging this research direction. To achieve this goal, this paper proposes a framework to automatically generate many dialogues without human involvement, in which any powerful opendomain dialogue generation model can be easily leveraged.

**Background:** Existing dialogue systems are usually categorized into two types: 
- Open-domain : focus on free chatting with users and are not able to complete tasks as human assistants.
- Task-oriented : is designed to complete the users’ goals.
- These two types have been studied separately till this study has released.

**Proposed Approach:** proposed approach introduces SalesBot as a new framework that generates data transitioning from open-domain chit-chats to task-oriented dialogues. This framework consists of 
- Open-Domain Dialogue Generation
    - use BlenderBot as pre-trained generation model open-domain conversations
    - use ParlAI2 software to create two BlenderBots with different personas to generate conversations which cover a wide range of topics.
- Chit-Chat to Task-Oriented Transition : focus on capturing the suitable timing and deciding how to smoothly transition to the target task respectively
    - Task-oriented intent detector to find out the good timing when someone might be interested in the product based on their conversation.
    - Transition turn generator focuses on smoothly switching from casual conversation to talking about the product.
- Task-Oriented Dialogue Generation : generating the conversation after the salesperson has transitioned from chit-chat to talking about the product.
    - Merge SGD: merge appropriate task-oriented dialogue from TOD with a chit-chat dialogue.
    - Task-Oriented Simulation: creates a new conversation from scratch. They train two different AI models, one for the customer and one for the salesperson. These models can have an open-ended conversation until they reach a stopping point.

**Results and Analysis:** The researchers evaluated their generated dialogues using human judges. The results showed the system could produce natural conversations relevant to the initial chat and transition smoothly to sales talk without being overly aggressive. Two methods for generating the sales conversation (Merge SGD and TOD Generation) showed similar performance while one underperformed due to the complexity of detecting implicit intents in chit-chat conversations.


## Acknowledgments

We extend our heartfelt gratitude to our dedicated team for their unwavering support and guidance throughout the project. Their invaluable contributions have been instrumental in the development and testing of this project. We would also like to express our special thanks to Minn, Wut Yee, Kyi, Puay and Noppawee for their exceptional efforts and commitment to ensuring the success of this endeavor.

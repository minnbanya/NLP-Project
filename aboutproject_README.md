# NLP Project - Sales Chatbot for Amazon

## Introduction to our team
Allow us to introduce the exceptional individuals who form the backbone of our project team:
- Minn Banya                (st124145)
- Wut Yee Aung              (st124377)
- Kyi Thin Nu               (st124087)
- Thamakorn Kiattikaikul    (st124481)
- Noppawee Teeraratchanon   (st124482)

## Our Paper Presentation
Please refer to this link [https://github.com/minnbanya/NLP-Project/blob/main/README.md]

## Project Background: The Rise of Sales Chatbots for Amazon

Sales chatbots, specifically tailored for Amazon, are AI-driven virtual assistants designed to engage customers in conversational interactions, guide them through the Amazon sales process, and facilitate purchases.

Key Factors Driving Adoption:

- 24/7 Availability: Provides instant assistance to Amazon customers round-the-clock.
- Scalability: Handles multiple conversations on Amazon simultaneously without compromising quality.
- Enhanced Customer Experience: Delivers personalized recommendations based on Amazon customer preferences and browsing history.
- Cost-Efficiency: Reduces operational expenses for Amazon sellers by automating routine sales tasks.
- Data Insights: Generates valuable data for Amazon sellers to analyze customer behavior and refine sales strategies.
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
        - will update according to Nopawee
    - LLM Part
        - Database will be split into vector stores
        - MLflow will be used for experiment logging and model integration
        - Instruction tuning tested and included if there is a positive impact on the performance
- Design test cases and evaluations for the system’s performance
- Web Design
    - Frontend: User interface for interacting with the chatbot, possibly through a website or messaging platform.
    - will update according to Ma Wut Yee

## Dataset Information

We will use the dataset from ---- and
write proper credit here

## Acknowledgments

We extend our heartfelt gratitude to our dedicated team for their unwavering support and guidance throughout the project. Their invaluable contributions have been instrumental in the development and testing of this project. We would also like to express our special thanks to Min, Wut Yee, Kyi, Puay and Nopawee for their exceptional efforts and commitment to ensuring the success of this endeavor.




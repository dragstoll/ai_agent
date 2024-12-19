# Advanced-AI-Project
Advanced-AI-Group-Project-2024


# Conversational Agent Project

## Project Overview

In this project, we are tasked with building a conversational agent capable of answering five types of questions:

1. **Factual Questions**
2. **Embedding Questions**
3. **Recommendation Questions**
4. **Multimedia Questions**
5. **Crowdsourcing Questions**

Throughout the project, we will undergo several evaluations, each testing the agent's ability to handle specific tasks. Below is the breakdown of tasks and requirements for each evaluation stage.

---

## 1st Intermediate Evaluation (14.10.2024)

### Task: Answer simple SPARQL queries using the provided knowledge graph.

### Bot Requirements:
1. **Connect to Speakeasy Infrastructure:**
   - The bot should log into Speakeasy, check assigned chatrooms, and interact via the API.
   
2. **Understand and Execute SPARQL Queries:**
   - The bot must read plain SPARQL queries sent in the conversation and execute them using the provided knowledge graph.
   
3. **Return Answers in Conversations:**
   - The bot should retrieve and present correct responses from the knowledge graph.

### Bot Capabilities for this Stage:
- Handle factual questions expressed in SPARQL.
- Basic interaction with the user (reading queries and returning results).
- Handle one chatroom at a time.

---

## 2nd Intermediate Evaluation (04.11.2024)

### Task: Answer factual and embedding questions by interpreting natural language queries and transforming them into SPARQL queries.

### Bot Requirements:
1. **Natural Language Understanding (Factual Questions):**
   - The bot should implement NLP to convert natural language questions into SPARQL queries.
   - Handle simple factual questions (e.g., "Who directed X?" or "When was Y released?").
   
2. **Answer Embedding Questions:**
   - Use pre-trained embeddings to answer similarity-based questions.
   - Clearly label embedding answers (e.g., "(Embedding Answer)").

3. **Improved Conversation Handling:**
   - Manage multiple chatrooms simultaneously.
   - Handle user corrections and provide additional context where needed.

### Bot Capabilities for this Stage:
- Answer factual and embedding questions from both the knowledge graph and embeddings.
- Properly label and explain answers.
- Handle multiple chatrooms effectively.

---

## 3rd Intermediate Evaluation (27.11.2024)

### Task: Answer recommendation questions using the knowledge graph.

### Bot Requirements:
1. **Recommendation System:**
   - Recommend movies based on user preferences or similar movies.
   - Use the knowledge graph to find relevant recommendations (e.g., similar genres, directors, or release years).
   
2. **Interpretation of User Preferences:**
   - Understand and process user input (e.g., "I like X, Y, Z movies. Can you recommend more?").

3. **Conversation Management:**
   - The bot should continue handling multiple chatrooms, responding with timely recommendations.

### Bot Capabilities for this Stage:
- Provide relevant movie recommendations using the knowledge graph.
- Manage multiple conversations while maintaining coherent dialogues.

---

## Final Evaluation (11.12.2024)

### Task: Answer all five types of questions—factual, embedding, recommendation, multimedia, and crowdsourcing—using the provided datasets.

### Bot Requirements:
1. **Answer Factual Questions:**
   - Handle natural language factual questions using SPARQL queries and the knowledge graph.
   
2. **Answer Embedding Questions:**
   - Provide embedding-based answers (similar to the 2nd evaluation).
   
3. **Recommendation System:**
   - Recommend movies based on user preferences (as in the 3rd evaluation).
   
4. **Answer Multimedia Questions:**
   - Use the multimedia dataset to show media-rich answers (e.g., return images or links to actors, directors, or movies).
   
5. **Answer Crowdsourcing Questions:**
   - Use the crowdsourcing dataset to answer questions (e.g., box office results, ratings).
   - Provide context on the crowd agreement (e.g., "This answer has a crowd agreement score of 0.5").
   
6. **Handle Multiple Conversations:**
   - The bot must be able to manage multiple conversations concurrently, providing accurate, relevant, and human-like responses.

### Bot Capabilities for this Stage:
- Fully functional across all five question types: factual, embedding, recommendation, multimedia, and crowdsourcing.
- Manage multiple chatrooms, providing human-like interactions.
- Ensure accurate, timely, and relevant responses.
- Provide crowdsourced answers with agreement context.

---

## Summary of Key Milestones:

- **1st Evaluation:** Focus on executing SPARQL queries and basic conversation handling.
- **2nd Evaluation:** Expand to natural language processing and embedding-based answers.
- **3rd Evaluation:** Add a recommendation system and improve chatroom handling.
- **Final Evaluation:** Handle all five question types with natural conversation flow and crowdsourcing data integration.

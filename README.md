# Overview

I have developed a chatbot using the Retrieval-Augmented Generation (RAG) framework to handle custom data, such as PDFs. The system utilizes Pinecone as the database for efficient storage and retrieval. The chatbot offers two distinct search options: **semantic search** and **hybrid search**. 

Semantic search leverages embeddings and is well-suited for querying individual documents. In contrast, hybrid search combines semantic search with syntactic search (using TF-IDF) and employs a re-ranking approach. This hybrid method is particularly effective for multi-document queries, where the balance between semantic and syntactic search can be fine-tuned using an adjustable alpha parameter. For instance, I have conducted two demos using custom scenarios with PDFs to showcase the chatbot's functionality.


# Overview

I have developed a chatbot using the Retrieval-Augmented Generation (RAG) framework to handle custom data, such as PDFs. The system utilizes Pinecone as the database for efficient storage and retrieval. The chatbot offers two distinct search options: **semantic search** and **hybrid search**. 

Semantic search leverages embeddings and is well-suited for querying individual documents. In contrast, hybrid search combines semantic search with syntactic search (using TF-IDF) and employs a re-ranking approach. This hybrid method is particularly effective for multi-document queries, where the balance between semantic and syntactic search can be fine-tuned using an adjustable alpha parameter. For instance, I have conducted two demos using custom scenarios with PDFs to showcase the chatbot's functionality.

# Chat-PDFs

## Demo 1

**Objective**: Imagine you are tasked with implementing a denoising solution for TEM/SEM microscopy videos. 

**Challenges**: The challenge is that there are only a limited number of videos available, and no ground truth exists for training. Since supervised learning is not feasible without ground truth data, and your knowledge of denoising algorithms is limited in this new domain, you must choose the best approach within a limited time frame. Algorithms like Noise2Noise, Noise2Void, and others are available, but selecting the most suitable one for this specific problem can be overwhelming. This is where a custom chatbot can be particularly useful, helping you navigate through the available methods and guide your decision-making process.
 
 <table>
  <tr>
      <td><img src="https://github.com/user-attachments/assets/3033a73c-c96d-4419-8212-cbb886738b52" alt="Image 1" width="2000"/></td>
      <td><img src="https://github.com/user-attachments/assets/e10b842c-5d7f-420a-9490-fa4147413c0f" alt="Image 2" width="2000"/></td>
  </tr>
   </table>


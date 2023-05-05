# VecLucene
VecLucene is an open-source vector search engine library built on top of Lucene and popular ANN (approximate nearest neighbor) search libraries. Its purpose is to simplify the process of vector search for users. VecLucene introduces the following enhancements to Lucene:

## Open Models
VecLucene currently supports [OpenAI's text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and Sentence_Transformer models, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by default, for embeddings. It has a flexible framework to support additional models and can be further extended to accommodate custom models. The larger models generally result in higher latency. The application can select the most suitable model based on the workload.

In addition, VecLucene can be expanded to support question-answering (QA) functionality. VecLucene can store the document text. For a query, VecLucene will find the matched text chunks and send them as context to models like ChatGPT to generate answers.

## Open ANN libraries
The default choice for VecLucene is [Hnswlib](https://github.com/nmslib/hnswlib). There are plans to support [Faiss](https://github.com/facebookresearch/faiss) and other libraries if necessary. This flexibility allows the application to choose the best library that aligns with its workload and requirements.

Lucene's KNN feature currently supports one embedding per document. However, for document text, a single embedding is often insufficient. One possible solution is to store text chunks within a document as multiple Lucene documents. While, this makes the inverted index more complex for the document.

## Self-Managed Document Store
With VecLucene, the application simply uploads the document, and VecLucene handles the rest. It automatically extracts text from the file (currently only plain text documents are supported), splits the text into chunks, calls the model to generate embeddings for each chunk, and persists the embeddings in the ANN library.

## Hybrid Search
VecLucene retains all of Lucene's existing abilities. The application can define multiple fields to store additional information for each document and use traditional Lucene queries to access these fields. For instance, the application can send a natural language query string with filters on other fields. VecLucene will generate an embedding for the query string, find similar documents in the ANN library, and filter out documents that don't meet the specified filtering conditions.

Furthermore, the text is indexed in Lucene using the traditional inverted index format. The application can choose to use either type of search or even perform a hybrid search by combining the results of both inverted index search and ANN search. Inverted index search is generally faster, while ANN search provides a better understanding of semantic relationships.

## Install

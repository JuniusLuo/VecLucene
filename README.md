# VecLucene
VecLucene is an open-source vector search engine library built on top of Lucene and popular ANN (approximate nearest neighbor) search libraries. Its purpose is to simplify the process of vector search for users. VecLucene introduces the following enhancements to Lucene:

## Open Models
VecLucene currently supports [OpenAI's text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and Sentence_Transformer models, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by default, for embeddings. It has a flexible framework to support additional models and can be further extended to accommodate custom models. The larger models generally result in higher latency. The application can select the most suitable model based on the workload. [HuggingFace MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb) measures the speed and performance of text embedding models. HuggingFace MTEB was published on October 19, 2022. So it does not include OpenAI embedding model.

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
VecLucene is built on top of PyLucene-9.4.1. Please follow the instructions in the [PyLucene Install guide](https://lucene.apache.org/pylucene/install.html) to install PyLucene. Note that after building jcc, you will need to edit Makefile to set "PYTHON", "JCC" and "NUM_FILES" for your platform. Please make sure you have installed JDK and GCC before building PyLucene. [JCC Install](https://lucene.apache.org/pylucene/jcc/install.html) suggests to install Temurin Java.

Other Python packages for VecLucene are managed by Poetry. You can use the "poetry export" command to create a requirements file and then install the packages with pip.

The Dockerfile is a good reference for how to build VecLucene.

## Usage
Once you have installed VecLucene, you can start it as an HTTP server by running `python main.py`. You can use the `example/cli.py` file to upload files, commit them, and query the server.

If you prefer, you can skip the installation process and use the pre-built Docker container:
1. Pull the docker image using `docker pull junius/veclucene-arm64`. For the amd64 platform, pull `junius/veclucene-amd64`. Please note that the size of the amd64 container image is much larger compared to the arm64 platform. This is because the packages required to run SentenceTransformer on the amd64 platform are significantly larger. You can pull `junius/veclucene-gpt-amd64`, which only works with the OpenAI Embedding model and is much smaller in size.
2. Run the container using `docker run -d --name vltest -p 127.0.0.1:8080:8080 junius/veclucene-arm64`, which uses the SetenceTransformer `all-MiniLM-L6-v2` model. To use ChatGPT, run `docker run -d --env ENV_EMBEDDING_MODEL_PROVIDER=openai_embedding --env OPENAI_API_KEY=xxx --name vltest -p 127.0.0.1:8080:8080 junius/veclucene-arm64`
3. Use `python3 example/cli.py --op upload --file path/to/localfile` and `python3 example/cli.py --op query --query_string "xxx" --query_type "vector"` to upload files and query the server. To use the traditional inverted search, use `--query_type "lucene"`. Don't forget to run `python3 example/cli.py --op commit` before stopping the container to ensure that the index is committed and can be queried again later.

Please note that VecLucene is still at an early stage and has limited abilities:
1. It only supports plain text files.
2. It is limited to 5000 embeddings.
3. The vector search does not parse the query string yet, e.g. simply generate the embedding from the entire query string. For the inverted search, the query string is parsed using Lucene parser.

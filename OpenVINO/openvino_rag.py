"""
OpenVINO RAG (Retrieval Augmented Generation) Module
---------------------------------------------------
This module implements a retrieval-based system for enhancing Large Language Model responses
with relevant information from a document corpus. It provides functionality to:

1. Index documents (PDF, TXT, DOCX, DOC, MD) using embeddings generated with OpenVINO-optimized models
2. Store and retrieve these embeddings using a FAISS vector database
3. Query the database for relevant content based on semantic similarity
4. Use the retrieved context to improve LLM responses

The module consists of two main classes:
- EmbeddingWrapper: A wrapper for the embedding model
- EmbeddingDatabase: A manager for the FAISS vector store and document processing

The implementation uses LangChain components for document loading, text splitting, and vector store operations.
"""

import gc
import json
import os
import time
from typing import Any

from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores.faiss import FAISS, Document

#### CONFIGURATIONS ------------------------------------------------------------------------------------------------------------------------
# Path to store the FAISS index and related metadata
INDEX_DATABASE_PATH = "./db/"  # Faiss database folder
# Text chunking parameters for document processing
CHUNK_SIZE = 1600  # Chunk size for text splitter
CHUNK_OVERLAP = 400  # Chunk overlap for text splitter
# Retrieval parameters
INDEX_NUM = 2  # Number of content pieces to retrieve
# Generation parameters
MAX_NEW_TOKENS = 320  # Max length of LLM output


# Embedding model class - create a wrapper for embedding model
class EmbeddingWrapper:
    """
    A wrapper class for the LlamaCppEmbeddings model.

    This class provides an interface for embedding documents and queries
    using the LlamaCppEmbeddings model, with performance timing.

    Attributes:
        model: An instance of LlamaCppEmbeddings used for generating embeddings
    """

    def __init__(self, model_path: str):
        """
        Initialize the embedding model with the specified model path.

        Args:
            model_path: Path to the embedding model file
        """
        start = time.time()
        print(f"******* loading {model_path} start ")
        self.model = LlamaCppEmbeddings(model_path=model_path)
        print(f"******* loading {model_path} finish. cost {time.time() - start:3f}s")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        t0 = time.time()
        embeddings = self.model.embed_documents(texts)
        t1 = time.time()
        print("-----------LlamaCpp--embedding cost time(s): ", t1 - t0)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text: The query text to embed

        Returns:
            Embedding vector as a list of floats
        """
        return self.model.embed_query(text)


# Faiss database - manage embeddings and file indexing
class EmbeddingDatabase:
    """
    Manages a FAISS vector database for document embeddings and retrieval.

    This class handles document loading, chunking, embedding, indexing,
    and similarity search operations. It supports various document formats
    and maintains metadata about indexed files.

    Attributes:
        db: FAISS vector store instance
        embeddings: EmbeddingWrapper for generating embeddings
        text_splitter: For splitting documents into chunks
        index_list: List of indexed files with metadata
    """

    db: FAISS
    embeddings: EmbeddingWrapper
    text_splitter: RecursiveCharacterTextSplitter
    index_list: list[dict[str, Any]]

    def __init__(self, embeddings: EmbeddingWrapper):
        """
        Initialize the embedding database with the provided embedding model.

        Args:
            embeddings: An EmbeddingWrapper instance for generating embeddings
        """
        self.embeddings = embeddings
        index_cache = os.path.join(INDEX_DATABASE_PATH, "index.faiss")
        self.db = FAISS.load_local(INDEX_DATABASE_PATH, self.embeddings) if os.path.exists(index_cache) else None
        index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
        self.index_list = self.__load_exists_index(index_json) if os.path.exists(index_json) else []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def __load_exists_index(self, index_json: str):
        """
        Load existing index metadata from a JSON file.

        Args:
            index_json: Path to the index metadata JSON file

        Returns:
            List of indexed file metadata or empty list on error
        """
        try:
            with open(index_json) as f:
                return json.load(f)
        except Exception as e:
            print(f"load index.json error: {e}")
            return []

    def __save_index(self, file_base_name: str, md5: str, doc_ids: str):
        """
        Save index metadata to the index list and JSON file.

        Args:
            file_base_name: Base name of the indexed file
            md5: MD5 hash of the file content
            doc_ids: List of document IDs in the vector store
        """
        self.index_list.append({"name": file_base_name, "md5": md5, "doc_ids": doc_ids})
        if not os.path.exists(INDEX_DATABASE_PATH):
            os.makedirs(INDEX_DATABASE_PATH)
        index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
        with open(index_json, "w") as f:
            json.dump(self.index_list, f)
        self.db.save_local(INDEX_DATABASE_PATH)

    def __add_documents(self, file_base_name: str, docs: list[Document], md5: str):
        """
        Add documents to the FAISS vector store.

        Args:
            file_base_name: Base name of the source file
            docs: List of Document objects to add
            md5: MD5 hash of the source file
        """
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
        else:
            self.db.add_documents(docs)
        print(docs[0].metadata)
        self.__save_index(file_base_name, md5, [doc.metadata["doc_id"] for doc in docs])

    def __analyze_file_to_db(self, file: str, md5: str):
        """
        Process a file, split it into chunks, and add to the database.

        This method handles different file types with appropriate loaders,
        splits the content into chunks, and adds them to the vector store.

        Args:
            file: Path to the file to process
            md5: MD5 hash of the file content

        Raises:
            Exception: If file type is unsupported or analysis fails
        """
        file_base_name = os.path.basename(file)
        file_ext = os.path.splitext(file_base_name)[1].lower()

        if file_ext == ".txt":
            raw_documents = TextLoader(file, encoding="utf-8").load()
        elif file_ext == ".pdf":
            raw_documents = PyPDFLoader(file).load()
        elif file_ext == ".doc":
            raw_documents = UnstructuredWordDocumentLoader(file).load()
        elif file_ext == ".docx":
            raw_documents = Docx2txtLoader(file).load()
        elif file_ext == ".md":
            raw_documents = UnstructuredMarkdownLoader(file).load()
        else:
            raise Exception(f"Unsupported file extension {file_ext}")

        docs = self.text_splitter.split_documents(raw_documents)
        if docs:
            print(f"Analyze {file_base_name} got {len(docs)} index files.")
            self.__add_documents(file_base_name, docs, md5)
        else:
            raise Exception(f"Cannot analyze {file_base_name}")

    def add_index_file(self, file: str):
        """
        Add a file to the index if it hasn't been indexed already.

        Args:
            file: Path to the file to index

        Returns:
            Tuple of (status_code, md5_hash)
            status_code: 0 for newly indexed, 1 for already indexed
        """
        md5 = self.__calculate_md5(file)
        for item in self.index_list:
            if item["md5"] == md5:
                print(f"{os.path.basename(file)} already indexed.")
                return 1, md5

        self.__analyze_file_to_db(file, md5)
        return 0, md5

    def query_database(self, query: str):
        """
        Query the database for documents similar to the query text.

        Args:
            query: The query text to find relevant documents for

        Returns:
            Tuple of (success, context, sources)
            success: Boolean indicating if relevant documents were found
            context: Combined text of relevant documents
            sources: List of source file names

        Raises:
            Exception: If query is empty or None
        """
        if not query:
            raise Exception("Query cannot be None or empty")

        print("******* Querying database...")
        if self.db is None:
            return False, None, None

        docs = self.db.similarity_search_with_relevance_scores(query, k=INDEX_NUM, score_threshold=0.4)
        if not docs:
            return False, None, None

        doc_contents = [doc.page_content for doc, _ in docs]
        source_files = {doc.metadata["source"] for doc, _ in docs}
        return True, "\n\n".join(doc_contents), "\n".join(source_files)

    def __calculate_md5(self, file_path: str) -> str:
        """
        Calculate MD5 hash for a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as a hexadecimal string
        """
        import hashlib

        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


# Global module variables for the embedding wrapper and database
embedding_wrapper = None
embedding_database = None


def init(model_path: str):
    """
    Initialize the RAG system with the specified embedding model.

    Args:
        model_path: Path to the embedding model file
    """
    global embedding_database, embedding_wrapper
    embedding_wrapper = EmbeddingWrapper(model_path=model_path)
    embedding_database = EmbeddingDatabase(embedding_wrapper)


def add_index_file(file: str):
    """
    Add a file to the index.

    Args:
        file: Path to the file to index

    Returns:
        Result from EmbeddingDatabase.add_index_file()
    """
    return embedding_database.add_index_file(file)


def query(query: str):
    """
    Query the database for relevant content.

    Args:
        query: The query text to find relevant documents for

    Returns:
        Result from EmbeddingDatabase.query_database()
    """
    return embedding_database.query_database(query)


def dispose():
    """
    Clean up resources by releasing references to global objects.
    Triggers garbage collection to free memory.
    """
    global embedding_database, embedding_wrapper
    embedding_database = None
    embedding_wrapper = None
    gc.collect()


if __name__ == "__main__":
    # Example Usage
    init(model_path="/Users/daniel/silicon/AI-Playground/LlamaCPP/models/llm/gguf/bge-large-en-v1.5-q8_0.gguf")
    add_index_file("/Users/daniel/silicon/AI-Playground/hello.txt")
    success, context, source = query("What is the content about?")
    print("Query success:", success)
    print("Context:", context)
    print("Source Files:", source)
    dispose()

"""
Retrieval-Augmented Generation (RAG) Module
------------------------------------------
This module implements RAG functionality to enhance AI responses with context from documents.

The RAG system allows:
- Loading and embedding various document types (PDF, DOCX, TXT, MD)
- Creating a searchable vector database using FAISS
- Querying the database for relevant content based on user input
- Managing document indexes with add/delete operations

The implementation uses the LangChain framework and SentenceTransformer for embeddings,
with FAISS for efficient similarity search and Intel XPU optimizations for performance.
"""

import gc
import json
import os
import re
import time
from typing import Any, List, Dict

# from sentence_transformers import SentenceTransformer
import intel_extension_for_pytorch as ipex  # noqa: F401
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores.faiss import FAISS, Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

import aipg_utils as utils
import service_config

#### CONFIGURATIONS ------------------------------------------------------------------------------------------------------------------------
INDEX_DATABASE_PATH = "./db/"  # Faiss database folder
CHUNK_SIZE = 1600  # Chunk size for text spliter
CHUNK_OVERLAP = 400  # Chunk overlap for text spliter
INDEX_NUM = 2  # How many pieces of content to index from db
MAX_NEW_TOKENS = 320  # Max length of LLM output


class EmbeddingWrapper(Embeddings):
    """
    Wrapper class for SentenceTransformer embeddings to integrate with LangChain.
    
    This class implements the LangChain Embeddings interface, loading embedding models
    from local storage and providing methods to create embeddings for documents and queries.
    """
    def __init__(self, repo_id: str):
        """
        Initialize the embedding model.
        
        Args:
            repo_id: Repository ID for the embedding model to load
        """
        model_embd_path = os.path.join(
            service_config.service_model_paths.get("embedding"), repo_id.replace("/", "---")
        )
        start = time.time()
        print(f"******* loading {model_embd_path} start ")
        self.model = SentenceTransformer(
            model_embd_path, trust_remote_code=True, device=service_config.device
        )

        print(
            "******* loading {} finish. cost{:3f}".format(
                model_embd_path, time.time() - start
            )
        )

    def to(self, device: str):
        """
        Move the model to the specified device.
        
        Args:
            device: Target device for the model
        """
        self.model.to(device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        torch.xpu.synchronize()
        t0 = time.time()
        embeddings = [
            self.model.encode(text, normalize_embeddings=True) for text in texts
        ]
        # Convert embeddings from NumPy arrays to lists for serialization
        embeddings_as_lists = [embedding.tolist() for embedding in embeddings]
        torch.xpu.synchronize()
        t1 = time.time()
        print("-----------SentenceTransformer--embedding cost time(s): ", t1 - t0)
        return embeddings_as_lists

    def embed_query(self, text: str) -> List[float]:
        """
        Create embedding for a query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embed_documents([text])[0]


class EmbeddingDatabase:
    """
    FAISS vector database for document embeddings.
    
    Manages document embeddings, persistence, and retrieval operations.
    Supports adding, querying, and deleting document indexes.
    """
    db: FAISS
    embeddings: EmbeddingWrapper
    text_splitter: RecursiveCharacterTextSplitter
    index_list: List[Dict[str, Any]]

    def __init__(self, embeddings: EmbeddingWrapper):
        """
        Initialize the embedding database.
        
        Args:
            embeddings: EmbeddingWrapper instance for creating embeddings
        """
        self.embeddings = embeddings
        index_cache = os.path.join(INDEX_DATABASE_PATH, "index.faiss")
        self.db = (
            FAISS.load_local(
                INDEX_DATABASE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            if os.path.exists(index_cache)
            else None
        )
        index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
        self.index_list = (
            self.__load_exists_index(index_json)
            if os.path.exists(index_json)
            else list()
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def to(self, device: str):
        """
        Move the embedding model to the specified device.
        
        Args:
            device: Target device for the embedding model
        """
        self.embeddings.to(device)

    def __load_exists_index(self, index_json: str):
        """
        Load existing index information from JSON file.
        
        Args:
            index_json: Path to the index JSON file
            
        Returns:
            List of index entries or empty list if loading fails
        """
        try:
            with open(index_json, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"load index.json error: {e}")
            return list()

    def __save_index(self, file_base_name: str, md5: str, doc_ids: str):
        """
        Save index information to disk.
        
        Args:
            file_base_name: Base name of the indexed file
            md5: MD5 hash of the file for identification
            doc_ids: Document IDs in the FAISS database
        """
        self.index_list.append({"name": file_base_name, "md5": md5, "doc_ids": doc_ids})
        if not os.path.exists(INDEX_DATABASE_PATH):
            os.makedirs(INDEX_DATABASE_PATH)
        index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
        with open(index_json, "w") as f:
            json.dump(self.index_list, f)
        self.db.save_local(INDEX_DATABASE_PATH)

    def __add_documents(self, file_base_name: str, docs: List[Document], md5: str):
        """
        Add documents to the FAISS database.
        
        Args:
            file_base_name: Base name of the file being added
            docs: List of Document objects to add
            md5: MD5 hash of the file for identification
        """
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
            docs_ids = list()
            for key in self.db.index_to_docstore_id:
                docs_ids.append(self.db.index_to_docstore_id[key])
        else:
            docs_ids = self.db.add_documents(docs)
        self.__save_index(file_base_name, md5, docs_ids)

    def __analyze_file_to_db(self, file: str, md5: str):
        """
        Process a file and add its contents to the database.
        
        Loads the file using the appropriate loader based on file type,
        splits the content into chunks, and adds them to the database.
        
        Args:
            file: Path to the file to analyze
            md5: MD5 hash of the file for identification
        """
        if not os.path.exists(INDEX_DATABASE_PATH):
            os.makedirs(INDEX_DATABASE_PATH)

        file_base_name = os.path.basename(file)
        file_ext = os.path.splitext(file_base_name)[1].lower()
        if file_ext == ".txt":
            # Load TXT and split into embeded pieces
            raw_documents = TextLoader(file, encoding="utf-8").load()
        elif file_ext == ".pdf":
            # Load PDF and split into embeded pieces
            raw_documents = PyPDFLoader(file).load()
        elif file_ext == ".doc":
            # Load WORD doc and split into embeded pieces
            raw_documents = UnstructuredWordDocumentLoader(file).load()
        elif file_ext == ".docx":
            # Load WORD doxc and split into embeded pieces
            raw_documents = Docx2txtLoader(file).load()
        elif file_ext == ".md":
            # Load markdown and split into embeded pieces
            raw_documents = UnstructuredMarkdownLoader(
                file, mode="elements", strategy="fast"
            ).load()  # UnstructuredFileLoader
        else:
            raise Exception(f"unsupported file ext {file_ext}")

        docs = self.text_splitter.split_documents(raw_documents)

        # Embedding the splitted pieces of text
        if docs is not None:
            print("anayze {} got index file {}".format(file_base_name, docs.__len__()))
            self.__add_documents(file_base_name, docs, md5)
        else:
            raise Exception("can't not anayze {} ".format(file_base_name))

    def add_index_file(self, file: str):
        """
        Add a file to the index.
        
        Args:
            file: Path to the file to index
            
        Returns:
            Tuple of (status_code, md5) where status_code is 1 if file already exists
            and 0 if file was newly added
        """
        md5 = utils.calculate_md5(file)
        for item in self.index_list:
            if item["md5"] == md5:
                base_name = os.path.basename(file)
                print(f"{base_name} index {md5} eixsts")
                return 1, md5

        self.__analyze_file_to_db(file, md5)
        return 0, md5

    def query_database(self, query: str):
        """
        Query the database for relevant document content.
        
        Args:
            query: The query string to search for
            
        Returns:
            Tuple of (success, context, source_file) where:
            - success: Boolean indicating if relevant content was found
            - context: Combined text of the relevant document chunks
            - source_file: Names of the source files that provided the content
        """
        if query is None or query == "":
            raise Exception("query can't be None or Empty")

        print("******* query from database ++ ")
        if self.db is None:
            return False, None, None
        docs = self.db.similarity_search_with_relevance_scores(
            query, k=2, score_threshold=0.4
        )
        if docs.__len__() == 0:
            return False, None, None
        # print("------------docs: ", docs[:2])
        doc_contents = list()
        source_set = set()
        for doc, _ in docs[:INDEX_NUM]:
            print("{}  --- {}", _, doc.page_content)
            doc_contents.append(doc.page_content)
            source_set.add(os.path.basename(doc.metadata["source"]))
        context = "\n\n".join(doc_contents)
        source_file = "\n\n".join(source_set)
        return True, context, source_file  # , page_num

    def delete_index(self, md5: str):
        """
        Delete a document from the index by its MD5 hash.
        
        Args:
            md5: MD5 hash of the document to delete
        """
        del_index = None
        for index in self.index_list:
            if index.get("md5") == md5:
                del_index = index
                break
        if del_index is not None:
            self.index_list.remove(del_index)
            if self.index_list.__len__() > 0:
                if not os.path.exists(INDEX_DATABASE_PATH):
                    os.makedirs(INDEX_DATABASE_PATH)
                index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
                with open(index_json, "w") as f:
                    json.dump(self.index_list, f)
                self.db.delete(del_index["doc_ids"])
                self.db.save_local(INDEX_DATABASE_PATH)
            else:
                index_json = os.path.join(INDEX_DATABASE_PATH, "index.json")
                if os.path.exists(index_json):
                    os.remove(index_json)
                index_faiss = os.path.join(INDEX_DATABASE_PATH, "index.faiss")
                if os.path.exists(index_faiss):
                    os.remove(index_faiss)
                index_pkl = os.path.join(INDEX_DATABASE_PATH, "index.pkl")
                if os.path.exists(index_pkl):
                    os.remove(index_pkl)


def add_index_file(file: str):
    """
    Add a file to the RAG index.
    
    Args:
        file: Path to the file to index
        
    Returns:
        Tuple of (status_code, md5) where status_code is 1 if file already exists
        and 0 if file was newly added
        
    Raises:
        Exception: If file type is not supported
    """
    global embedding_database
    if re.search(".(txt|docx?|pptx?|md|pdf)$", file, re.IGNORECASE) is not None:
        torch.xpu.synchronize()
        start = time.time()
        result = embedding_database.add_index_file(file)
        torch.xpu.synchronize()
        end = time.time()
        print(f"add index file cost {end-start}s")
    else:
        raise Exception("not suppported file type")
    return result


def to(device: str):
    """
    Move the embedding model to the specified device.
    
    Args:
        device: Target device for the embedding model
    """
    global embedding_database
    embedding_database.to(device)


def query(query: str):
    """
    Query the RAG database for relevant content.
    
    Args:
        query: The query string to search for
        
    Returns:
        Tuple of (success, context, source_file) with relevant document information
    """
    global embedding_database
    torch.xpu.synchronize()
    start = time.time()
    success, context, source_file = embedding_database.query_database(query)
    end = time.time()
    print(f'query by keyword "{query}" cost {end-start}s')
    torch.xpu.synchronize()
    return success, context, source_file


def delete_index(md5: str):
    """
    Delete a document from the RAG index by its MD5 hash.
    
    Args:
        md5: MD5 hash of the document to delete
    """
    global embedding_database
    embedding_database.delete_index(md5)


def get_index_list():
    """
    Get a list of all indexed documents.
    
    Returns:
        List of document index entries with name, MD5, and document IDs
    """
    global embedding_database
    return embedding_database.index_list


# Global variables for the RAG system
embedding_wrapper: EmbeddingWrapper = None
embedding_database: EmbeddingDatabase = None
Is_Inited = False


def init(repo_id: str, device: int):
    """
    Initialize the RAG system.
    
    Sets up the embedding model and database, and configures the device.
    
    Args:
        repo_id: Repository ID for the embedding model
        device: Device ID for XPU acceleration
    """
    global embedding_database, embedding_wrapper, Is_Inited
    torch.xpu.set_device(device)
    service_config.device = f"xpu:{device}"
    embedding_wrapper = EmbeddingWrapper(repo_id)
    embedding_database = EmbeddingDatabase(embedding_wrapper)
    Is_Inited = True


def dispose():
    """
    Clean up RAG system resources.
    
    Releases memory used by embedding models and database, and clears GPU cache.
    """
    global embedding_database, embedding_wrapper, Is_Inited
    if Is_Inited:
        if embedding_wrapper is not None:
            del embedding_wrapper
            embedding_wrapper = None
        if embedding_database is not None:
            del embedding_database
            embedding_database = None
        Is_Inited = False
    gc.collect()
    torch.xpu.empty_cache()

class FAISS:
    def __init__(self, *args, **kwargs):
        pass

    def add_documents(self, documents):
        pass

    def search(self, query, k):
        return []


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"

import chromadb 
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorInit():
    def __init__(self, db_path):
        """
        Initializes the VectorInit class.

        Args:
            db_path (str): The path to the vector store database.

        Returns:
            None
        """
        self.db_path = db_path

    def vectorstore_client(self):
        """
        Creates and returns a persistent client for the vector store.

        Returns:
            chromadb.PersistentClient: The persistent client for the vector store.
        """
        return chromadb.PersistentClient(
            path = self.db_path
        )

    def create_embeddings(self):
        """
        Creates and returns an embedding function for sentence transformation.

        Returns:
            embedding_functions.SentenceTransformerEmbeddingFunction: The embedding function for sentence transformation.
        """
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class InsertVectorData(VectorInit):
    def __init__(self, collection_name, text_context, db_path):
        """
        Initializes the InsertVectorData class.

        Args:
            collection_name (str): The name of the collection.
            text_context (str): The text context to be inserted.
            db_path (str): The path to the vector store database.

        Returns:
            None
        """
        super().__init__(db_path)
        self.collection_name = collection_name
        self.text_context = text_context

    def delete_collection(self):
        """
        Deletes the collection from the vector store.

        Returns:
            None
        """
        client = super().vectorstore_client()
        client.delete_collection(name=self.collection_name)

    def make_pull_collection(self):
        """
        Creates a collection in the vector store and adds the embedding function.

        Returns:
            None
        """
        client = super().vectorstore_client()
        collection = client.get_or_create_collection(
            name = self.collection_name,
            metadata = {
                "hnsw:space": "cosine"
            },

            embedding_function = super().create_embeddings()
        )
        print(client.list_collections())
    
    def context_chunking(self):
        """
        Splits the text context into smaller chunks.

        Returns:
            list: The list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_text(str(self.text_context))
        return texts

    def insert_data(self, chunked_context,file_name):
        """
        Inserts the chunked context into the collection in the vector store.

        Args:
            chunked_context (list): The list of text chunks.

        Returns:
            None
        """
        chunks_index = list(range(len(chunked_context)))
        chunks_index = [str(x) for x in chunks_index]

        #print(chunked_context)
        #print(chunks_index)

        metadata_dict = [{"document_name":file_name} for _ in range(len(chunks_index))]

        client = super().vectorstore_client()
        collection = client.get_collection(name=self.collection_name)
        collection.add(
            documents=chunked_context,
            ids=chunks_index,
            metadatas = metadata_dict
        )

class VectorSimilaritySearch(VectorInit):
    def __init__(self, text_prompt, collection_name, db_path):
        """
        Initializes the VectorSimilaritySearch class.

        Args:
            text_prompt (str): The text prompt for similarity search.
            collection_name (str): The name of the collection.
            db_path (str): The path to the vector store database.

        Returns:
            None
        """
        super().__init__(db_path)
        self.text_prompt = text_prompt
        self.collection_name = collection_name
        
    def embedding_prompt(self):
        """
        Generates the embedding for the text prompt.

        Returns:
            numpy.ndarray: The embedding for the text prompt.
        """
        embedding_func = super().create_embeddings()
        text_embedding = embedding_func([self.text_prompt])
        return text_embedding

    def query_vectorstore(self, text_embedding):
        """
        Queries the vector store for similar documents.

        Args:
            text_embedding (numpy.ndarray): The embedding for the text prompt.

        Returns:
            list: The list of similar documents.
        """
        client = super().vectorstore_client()
        collection = client.get_collection(name=str(self.collection_name))
        data = collection.get()
        sim_search_result = collection.query(
            query_embeddings=text_embedding,
            n_results=3
        )

        supporting_docs_list = []
        for i in range(len((sim_search_result['metadatas'])[0])):
            supporting_docs_list.append((sim_search_result['metadatas'])[0][i])

        groundings = sim_search_result['documents']

        return groundings, supporting_docs_list
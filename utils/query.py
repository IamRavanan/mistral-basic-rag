from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import pickle
from utils.make_db import BasicMistralRag as mrag


class QueryMistral(mrag):
    def __init__(self, question) -> None:
        super().__init__()
        self.question = question

    def load_vector_db_from_file(self):
        """
        Loads the vector database from a file using pickle.

        This method opens the specified file in read-binary mode, deserializes the data
        using pickle, and returns the loaded vector database.

        Returns:
            faiss.IndexFlatL2: The FAISS index loaded from the file.
        """
        with open(self.vector_db_file_name, 'rb') as f:
            index = pickle.load(f)
        print(f" - Vector DB loaded from {self.vector_db_file_name}")
        return index

    def create_embedding_for_question(self, question: str, mistral_client: MistralClient) -> np.ndarray:
        """
        Creates an embedding for a given question using the Mistral client.

        This method retrieves the embedding for the provided question using the Mistral client
        and returns it as a NumPy array.

        Args:
            question (str): The question text to be embedded.
            mistral_client (MistralClient): The Mistral client instance used to get the embedding.

        Returns:
            np.ndarray: An array containing the embedding of the question.
        """
        question_embeddings = np.array(
            [self.get_text_embedding(question, mistral_client)])
        return question_embeddings

    def retrieve_chunks_from_vector_db(self, index, question_embeddings: np.ndarray, chunks: list) -> list:
        """
        Retrieves the most relevant chunks from the vector database based on the question embeddings.

        This method searches the vector database using the provided question embeddings to find
        the most relevant chunks. It returns the top-k retrieved chunks.

        Args:
            index (faiss.IndexFlatL2): The FAISS index containing the text embeddings.
            question_embeddings (np.ndarray): The embeddings of the question to search for.
            chunks (list): The list of text chunks corresponding to the embeddings in the index.

        Returns:
            list: A list of the most relevant text chunks.
        """
        D, I = index.search(question_embeddings, k=3)  # distance, index
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        return retrieved_chunks

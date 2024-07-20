from mistralai.client import MistralClient
import numpy as np
import faiss
import os
import pickle

KB_FILE_NAME = "kb/essay.txt"
VECTOR_DB_FILE_NAME = "vector_db/vector_db.pkl"
CHUNKS_FILE_NAME = "vector_db/chunks.pkl"


class BasicMistralRag:
    def __init__(self, kb_file_name='', vector_db_file_name='', chunks_file_name='') -> None:
        self.apikey = os.environ["MISTRAL_API_KEY"]
        self.model = "mistral-large-latest"
        self.vector_db_file_name = VECTOR_DB_FILE_NAME
        self.chunks_file_name = CHUNKS_FILE_NAME
        self.kb_file_name = KB_FILE_NAME

    def load_doc(self, file_name: str) -> str:
        """
        Loads the content of a file.

        This method opens the specified file in read mode, reads its content, and returns the data.

        Args:
            file_name (str): The name of the file to be read.

        Returns:
            str: The content of the file.
        """
        with open(file_name, "r") as f:
            data = f.read()
        f.close()
        return data

    def split_doc(self, data: str) -> list:
        """
        Splits the given data into chunks of a specified size.

        This method divides the input data into smaller chunks, each of a specified size,
        and returns a list of these chunks.

        Args:
            data (str): The data to be split into chunks.

        Returns:
            list: A list of data chunks.
        """
        chunk_size = 2048
        chunks = [data[i:i + chunk_size]
                  for i in range(0, len(data), chunk_size)]
        return chunks

    def get_text_embedding(self, chunk: str, mistral_client: MistralClient) -> list:
        """
        Retrieves the text embedding for a given chunk using the Mistral client.

        This method sends the chunk to the Mistral client to obtain its embedding
        using the specified model, and returns the embedding.

        Args:
            chunk (str): The text chunk to be embedded.
            mistral_client (MistralClient): The Mistral client instance used to get the embedding.

        Returns:
            list: The embedding of the text chunk.
        """
        embeddings_batch_response = mistral_client.embeddings(
            model="mistral-embed",
            input=chunk
        )
        return embeddings_batch_response.data[0].embedding

    def get_all_text_embeddings(self, chunks: list, mistral_client: MistralClient) -> np.ndarray:
        """
        Retrieves text embeddings for all chunks using the Mistral client.

        This method iterates over each chunk, retrieves its embedding using the Mistral client,
        and returns an array of all the embeddings.

        Args:
            chunks (list): A list of text chunks to be embedded.
            mistral_client (MistralClient): The Mistral client instance used to get the embeddings.

        Returns:
            np.ndarray: An array of embeddings for the text chunks.
        """
        text_embeddings = np.array([self.get_text_embedding(
            chunk, mistral_client) for chunk in chunks])
        return text_embeddings

    def load_vector_db(self, text_embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Loads the vector database with text embeddings.

        This method creates a FAISS index, adds the provided text embeddings to it,
        and returns the index.

        Args:
            text_embeddings (np.ndarray): An array of text embeddings to be added to the index.

        Returns:
            faiss.IndexFlatL2: The FAISS index containing the text embeddings.
        """
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
        return index

    def write_to_file(self, data: str, file_name: str) -> None:
        """
        Writes the given data to a file using pickle.

        This method opens the specified file in write-binary mode, serializes the provided data
        using pickle, and writes it to the file.

        Args:
            data (any): The data to be written to the file.
            file_name (str): The name of the file to write the data to.
        """
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
        f.close()

    def create_vector_db(self):
        print(" - Creating Mistral Client ...")
        mc = MistralClient(api_key=self.apikey)

        print(" - Loading your document ...")
        data = self.load_doc(self.kb_file_name)

        print(" - Splitting your document to chunks ...")
        chunks = self.split_doc(data)

        print(" - Storing local copy of chunks ...")
        self.write_to_file(chunks, self.chunks_file_name)

        print(" - Converting chunks to vector embeddings ...")
        text_embeddings = self.get_all_text_embeddings(chunks, mc)

        print(" - Loading to vector DB ...")
        index = self.load_vector_db(text_embeddings)

        print(" - Creating local copy of vector DB ...")
        self.write_to_file(index, self.vector_db_file_name)

        print(f" - Vector DB saved to {self.vector_db_file_name}")

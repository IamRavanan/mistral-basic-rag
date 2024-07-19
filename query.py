from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import pickle
from make_db import BasicMistralRag as mrag


class QueryMistral(mrag):
    def __init__(self, question) -> None:
        super().__init__()
        self.question = question

    def load_vector_db_from_file(self):
        with open(self.vector_db_file_name, 'rb') as f:
            index = pickle.load(f)
        print(f" - Vector DB loaded from {self.vector_db_file_name}")
        print(index)
        return index

    def create_embedding_for_question(self, question, mistral_client):
        question_embeddings = np.array(
            [self.get_text_embedding(question, mistral_client)])
        return question_embeddings

    def retrieve_chunks_from_vector_db(self, index, question_embeddings, chunks):
        D, I = index.search(question_embeddings, k=3)  # distance, index
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        return retrieved_chunks

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import pickle
from make_db import BasicMistralRag as mrag
import argparse

class QueryMistral(mrag):
    def __init__(self, question) -> None:
        super().__init__()
        self.question = question               

    def load_vector_db_from_file(self):
        # Load the vector DB from a file
        with open(self.vector_db_file_name, 'rb') as f:
            index = pickle.load(f)
        print(f" - Vector DB loaded from {self.vector_db_file_name}")
        print(index)
        return index

    def create_embedding_for_question(self, question, mistral_client):
        question_embeddings = np.array([self.get_text_embedding(question, mistral_client)])
        return question_embeddings
    
    def retrieve_chunks_from_vector_db(self, index, question_embeddings, chunks):
        D, I = index.search(question_embeddings, k=3) # distance, index      
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        return retrieved_chunks        
    
    def run(self, model="mistral-medium-latest"):
        print(" - Creating Mistral Client ...")
        mc =  MistralClient(api_key=self.apikey)
        
        print(" - Loading index from vector DB")
        index = self.load_vector_db_from_file()

        print(" - Creating vector embedding for question")
        question_embedding = self.create_embedding_for_question(self.question, mc)

        print("Vector embedding for a question:")
        print(question_embedding)

        print(" - Loading all chunks ...")
        chunks = self.load_doc(self.chunks_file_name)

        print(" - Retrieve chunks from vector DB ....")
        retrieved_chunk = self.retrieve_chunks_from_vector_db(index, question_embedding, chunks)

        print("Retrieved Chunks : ")
        print(retrieved_chunk)

        prompt = f"""
                    Context information is below.
                    ---------------------
                    {retrieved_chunk}
                    ---------------------
                    Given the context information and not prior knowledge, answer the query.
                    Query: {self.question}
                    Answer:
                    """
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        # chat_response = mc.chat(
        #     model=model,
        #     messages=messages
        # )
        # return (chat_response.choices[0].message.content)    

if __name__ == "__main__":    
    question = "What were the two main things the author worked on before college?"
    parser = argparse.ArgumentParser(description="Request user input")
    parser.add_argument('question', type=str, help='Post your question')    
    args = parser.parse_args()
    
    qm = QueryMistral(args.question)
    response = qm.run()
    print(response)

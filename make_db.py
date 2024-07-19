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

    def load_doc(self, file_name):
        with open (file_name, "r") as f:
            data = f.read()
        f.close()
        return data
    
    def split_doc(self, data):
        chunk_size = 2048
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks  


    def get_text_embedding(self, chunk, mistral_client):
        embeddings_batch_response = mistral_client.embeddings(
            model="mistral-embed",
            input=chunk
        )
        return embeddings_batch_response.data[0].embedding
    
    def get_all_text_embeddings(self, chunks, mistral_client):
        text_embeddings = np.array([self.get_text_embedding(chunk, mistral_client) for chunk in chunks])
        return text_embeddings
    
    def load_vector_db(self, text_embeddings):
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
        return index
    
    def write_to_file(self, data, file_name):
        with open (file_name, "wb") as f:
            pickle.dump(data,f)

    def create_vector_db(self):
        print(" - Creating Mistral Client ...")
        mc =  MistralClient(api_key=self.apikey)
        
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


if __name__ == "__main__":
    mrag = BasicMistralRag(KB_FILE_NAME, VECTOR_DB_FILE_NAME)
    mrag.create_vector_db()
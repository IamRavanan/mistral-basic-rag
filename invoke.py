from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from utils.query import QueryMistral as qm
import argparse
import os


class Invoke(qm):

    def __init__(self, question) -> None:
        self.apikey = os.environ["MISTRAL_API_KEY"]
        self.question = question
        super().__init__(question)

    def run(self):

        print(" - Creating Mistral Client ...")
        mc = MistralClient(api_key=self.apikey)

        print(" - Loading data ...")
        data = self.load_doc(self.kb_file_name)

        print(" - Transforming data into chunks ...")
        chunks = self.split_doc(data)

        print(" - Converting chunks to vector embeddings ...")
        text_embeddings = self.get_all_text_embeddings(chunks, mc)

        print(" - Loading Vector DB ...")
        index = self.load_vector_db(text_embeddings)

        print(" - Converting user prompt to embedding ...")
        question_embedding = self.create_embedding_for_question(
            self.question, mc)

        print(" - Retrieving matching chunks from vector DB ...")
        retrieved_chunks = self.retrieve_chunks_from_vector_db(
            index, question_embedding, chunks)

        print(" - Formulating prompt ...")
        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunks}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {self.question}
        Answer:
        """
        print(" - Augumenting response with Mistral's mistral-medium-latest model...")
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        chat_response = mc.chat(
            model="mistral-medium-latest",
            messages=messages
        )
        return (chat_response.choices[0].message.content)


if __name__ == "__main__":

    question = "What were the two main things the author worked on before college?"

    parser = argparse.ArgumentParser(description="Request user input")
    parser.add_argument('question', type=str, help='Post your question')
    args = parser.parse_args()

    invoke = Invoke(args.question)
    response = invoke.run()
    print(" - Printing response ...")
    print(f"\n Response : {response}")

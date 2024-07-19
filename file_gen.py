import requests

DOC_LINK = 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt'

class FileGenrator:

    def __init__(self, url, file_name) -> None:
        self.url = url
        self.file_name = file_name

    def retrieve_doc(self):
        """
        Retrieves the document from the specified URL.

        This method sends a GET request to the URL stored in the instance variable `self.url`
        and returns the response text.

        Returns:
            str: The text content of the response from the URL.
        """
        response = requests.get(self.url)
        return response.text

    def write_to_file(self, data):
        """
        Writes the given data to a file.

        This method opens a file with the name stored in the instance variable `self.file_name`
        in write mode, writes the provided data to the file, and then closes the file.

        Args:
            data (str): The data to be written to the file.
        """
        with open(self.file_name, "w") as f:
            f.write(data)
        f.close()

    def generate_file(self):
        """
        Generates a file by retrieving data from a URL and writing it to a file.

        This method first retrieves the document from the URL using the `retrieve_doc` method,
        then writes the retrieved data to a file using the `write_to_file` method.
        """
        data = self.retrieve_doc()
        self.write_to_file(data)



if __name__ == "__main__":
    file_gen = FileGenrator(DOC_LINK, 'essay.txt')
    file_gen.generate_file()

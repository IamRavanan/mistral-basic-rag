# MISTRAL : RAG from scratch

This project guides you through building a basic Retrieval-Augmented Generation (RAG) system. It aims to help you understand RAG’s internal workings and provide the essential skills to create a RAG with minimal dependencies.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Clone the repository
```bash
git clone https://github.com/IamRavanan/mistral-basic-rag.git
```

### Navigate to the project directory
```bash
cd mistral-basic-rag
```
### Create virtual environment
```bash
python -m venv .venv
```

### Activate your virtual environment

#### Windows
```bash
.venv\Scripts\activate.bat\
```
#### MacOs
```bash
source .venv/bin/activate
```
### Install dependencies
```bash
python -m pip install --upgrade pip && pip install -r requirements.txt
```

## Usage

#### Invoke the script
```bash
python .\invoke.py "What were the two main things the author worked on before college?"
```

#### Example Output
```bash
PS D:\GitHub\mistral-rag> python .\invoke.py "What were the two main things the author worked on before college?"
 - Creating Mistral Client ...
 - Loading data ...
 - Transforming data into chunks ...
 - Converting chunks to vector embeddings ...
 - Loading Vector DB ...
 - Converting user prompt to embedding ...
 - Loading Vector DB ...
 - Converting user prompt to embedding ...
 - Converting user prompt to embedding ...
 - Retrieving matching chunks from vector DB ...
 - Formulating prompt ...
 - Augumenting response with Mistral's mistral-medium-latest model...
 - Printing response ...

 Response : The two main things the author worked on before college were writing and programming. They wrote short stories, which they described as having hardly any plot and mostly focusing on characters with strong feelings. In terms of programming, they tried writing programs on an IBM 1401 in 9th grade using an early version of Fortran. They typed programs on punch cards, which were then loaded into memory and run on the machine. However, they couldn't remember any specific programs they wrote as they didn't have any data stored on punched cards and didn't know enough math to do anything interesting without input. With the advent of microcomputers, they found that programming became much more accessible and interesting.
```

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License
```
MIT License
```

## Contact
- [LinkedIn](https://www.linkedin.com/in/ndeenadayalan)
- [GitHub](https://github.com/IamRavanan)
- [Blog](https://configmistakes.wordpress.com/)



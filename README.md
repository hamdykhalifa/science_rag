## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Implementing a vector store](#implementing-vectorstore)
- [Structure](#structure)



## Getting Started

The project has three main components: the frontend directory, the backend directory and a testing jupyter notebook (lc_rag.ipynb).

The lc_rag.ipynb file is there for personal testing purposes pre-implementation into the application.
Furthermore it contains code that is not implemented in the application such as the extraction of text from all test data, the implementation of a chroma data base and the processing
of data to fill the data base respectively.

It is recommended to download the test data from into the root folder of the project into a /data subdirectory, so that e.g. "data/luri_higher_topos.pdf".

The frontend folder contains a light-weight streamlit application that will send requests to the api in the backend upon user action.

The backend folder contains the fast api which at the moment only contains on api endpoint for the creation of a summary.


### Prerequisites

First of all, create a virtual environment so installed packages do not conflict with globally installed packages on your machine.

With conda:

```
conda create --name <name_of_your_env> python=3.11

conda activate <name_of_your_env>
```

With venv:

Navigate into the project root folder. Then:

```
python -m venv <name_of_your_env>

<name_of_your_env>\Scripts\activate
```


Another prerequisite is the creation of a .env file in the root folder of the directory. There place your api key to OpenAI like this:
``` OPENAI_API_KEY="sk-Tbp..." ```

### Installation

After having activated the virtual environment install the project specific requirements.txt with
``` pip install -r requirements.txt```

### Implementing a vector store

In order to have a lightweight vector store implemented, to give the models a knowledge base, it is useful to create a local in-memory database with Chroma.
To do so, please run the lc_rag.ipynb notebook until the section "Retrieval" (Run Import cells > Load Document > Split text > Create vectorstore).

## Structure

As mentioned in the [Getting Started] section the current setup consists of a frontend and a backend directory.

In order to start them both, you need to run individual commands into two different terminals.

To start the backend:

- cd into the backend directory
- ```uvicorn central_api:app --reload```

You should wait until the debug info is saying "INFO: Application startup complete."


To start the frontend:
- cd into the frontend directory
- ```streamlit run app.py```

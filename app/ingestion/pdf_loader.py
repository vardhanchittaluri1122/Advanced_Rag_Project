from langchain_community.document_loaders import PyPDFLoader # type: ignore


def load_pdf(file_path):

    loader = PyPDFLoader(
        file_path,
        mode="page"
    )

    documents = loader.load()

    return documents
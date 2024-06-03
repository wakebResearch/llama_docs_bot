from pathlib import Path

# For LLM and embedding model
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# For Reading Documents
from llama_index.core import SimpleDirectoryReader
from llama_docs_bot.markdown_docs_reader import MarkdownDocsReader

# Make our printing look nice
from llama_index.core.schema import MetadataMode, Document

# for Logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_markdown_docs(filepath) -> list[Document]:
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        exclude=["*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg",
                 "*.jpeg", "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True
    )

    return loader.load_data()


if __name__ == '__main__':

    """Loading Documents"""  # Load our documents from each folder.
    # we keep them seperate for now, in order to create seperate indexes later
    base_path = Path('docs')
    getting_started_docs = load_markdown_docs(base_path / "getting_started")
    community_docs = load_markdown_docs(base_path / "community")
    data_docs = load_markdown_docs(base_path / "core_modules/data_modules")
    agent_docs = load_markdown_docs(base_path / "core_modules/agent_modules")
    model_docs = load_markdown_docs(base_path / "core_modules/model_modules")
    query_docs = load_markdown_docs(base_path / "core_modules/query_modules")
    supporting_docs = load_markdown_docs(
        base_path / "core_modules/supporting_modules")
    tutorials_docs = load_markdown_docs(base_path / "end_to_end_tutorials")
    contributing_docs = load_markdown_docs(base_path / "development")

    # Printing metedata
    print(agent_docs[5].get_content(metadata_mode=MetadataMode.ALL))

    """For LLM and Embedding Model"""
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5")
    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    print('Success !')

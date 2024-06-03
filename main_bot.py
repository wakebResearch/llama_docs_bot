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

# For Index Database
from llama_index.core import (
    VectorStoreIndex, StorageContext, load_index_from_storage)

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

    """Create Indicies"""
    # create a vector store index for each folder
    index_base_path = Path('indices')
    if not index_base_path.is_dir():
        index_base_path.mkdir()
    try:
        getting_started_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./getting_started_index"))
        community_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./community_index"))
        data_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./data_index"))
        agent_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./agent_index"))
        model_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./model_index"))
        query_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./query_index"))
        supporting_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./supporting_index"))
        tutorials_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./tutorials_index"))
        contributing_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./contributing_index"))
    except:
        getting_started_index = VectorStoreIndex.from_documents(
            getting_started_docs)
        getting_started_index.storage_context.persist(
            persist_dir="./getting_started_index")

        community_index = VectorStoreIndex.from_documents(community_docs)
        community_index.storage_context.persist(
            persist_dir="./community_index")

        data_index = VectorStoreIndex.from_documents(data_docs)
        data_index.storage_context.persist(persist_dir="./data_index")

        agent_index = VectorStoreIndex.from_documents(agent_docs)
        agent_index.storage_context.persist(persist_dir="./agent_index")

        model_index = VectorStoreIndex.from_documents(model_docs)
        model_index.storage_context.persist(persist_dir="./model_index")

        query_index = VectorStoreIndex.from_documents(query_docs)
        query_index.storage_context.persist(persist_dir="./query_index")

        supporting_index = VectorStoreIndex.from_documents(supporting_docs)
        supporting_index.storage_context.persist(
            persist_dir="./supporting_index")

        tutorials_index = VectorStoreIndex.from_documents(tutorials_docs)
        tutorials_index.storage_context.persist(
            persist_dir="./tutorials_index")

        contributing_index = VectorStoreIndex.from_documents(contributing_docs)
        contributing_index.storage_context.persist(
            persist_dir="./contributing_index")

    print('Success !')

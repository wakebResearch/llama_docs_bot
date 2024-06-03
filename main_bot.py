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

# For Query Engine Tool
from llama_index.core.tools import QueryEngineTool

# For Unified Query Engine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import SubQuestionQueryEngine


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
            StorageContext.from_defaults(persist_dir=index_base_path / "getting_started_index"))
        community_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "community_index"))
        data_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "data_index"))
        agent_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "agent_index"))
        model_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "model_index"))
        query_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "query_index"))
        supporting_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "supporting_index"))
        tutorials_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "tutorials_index"))
        contributing_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_base_path / "contributing_index"))
    except:
        getting_started_index = VectorStoreIndex.from_documents(
            getting_started_docs)
        getting_started_index.storage_context.persist(
            persist_dir=index_base_path / "getting_started_index")

        community_index = VectorStoreIndex.from_documents(community_docs)
        community_index.storage_context.persist(
            persist_dir=index_base_path / "community_index")

        data_index = VectorStoreIndex.from_documents(data_docs)
        data_index.storage_context.persist(
            persist_dir=index_base_path / "data_index")

        agent_index = VectorStoreIndex.from_documents(agent_docs)
        agent_index.storage_context.persist(
            persist_dir=index_base_path / "agent_index")

        model_index = VectorStoreIndex.from_documents(model_docs)
        model_index.storage_context.persist(
            persist_dir=index_base_path / "model_index")

        query_index = VectorStoreIndex.from_documents(query_docs)
        query_index.storage_context.persist(
            persist_dir=index_base_path / "query_index")

        supporting_index = VectorStoreIndex.from_documents(supporting_docs)
        supporting_index.storage_context.persist(
            persist_dir=index_base_path / "supporting_index")

        tutorials_index = VectorStoreIndex.from_documents(tutorials_docs)
        tutorials_index.storage_context.persist(
            persist_dir=index_base_path / "tutorials_index")

        contributing_index = VectorStoreIndex.from_documents(contributing_docs)
        contributing_index.storage_context.persist(
            persist_dir=index_base_path / "contributing_index")

    """Create Qurey Engine Tool"""
    # create a query engine tool for each folder
    getting_started_tool = QueryEngineTool.from_defaults(
        query_engine=getting_started_index.as_query_engine(),
        name="Getting Started",
        description="Useful for answering questions about installing and running llama index, as well as basic explanations of how llama index works."
    )

    community_tool = QueryEngineTool.from_defaults(
        query_engine=community_index.as_query_engine(),
        name="Community",
        description="Useful for answering questions about integrations and other apps built by the community."
    )

    data_tool = QueryEngineTool.from_defaults(
        query_engine=data_index.as_query_engine(),
        name="Data Modules",
        description="Useful for answering questions about data loaders, documents, nodes, and index structures."
    )

    agent_tool = QueryEngineTool.from_defaults(
        query_engine=agent_index.as_query_engine(),
        name="Agent Modules",
        description="Useful for answering questions about data agents, agent configurations, and tools."
    )

    model_tool = QueryEngineTool.from_defaults(
        query_engine=model_index.as_query_engine(),
        name="Model Modules",
        description="Useful for answering questions about using and configuring LLMs, embedding modles, and prompts."
    )

    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_index.as_query_engine(),
        name="Query Modules",
        description="Useful for answering questions about query engines, query configurations, and using various parts of the query engine pipeline."
    )

    supporting_tool = QueryEngineTool.from_defaults(
        query_engine=supporting_index.as_query_engine(),
        name="Supporting Modules",
        description="Useful for answering questions about supporting modules, such as callbacks, service context, and avaluation."
    )

    tutorials_tool = QueryEngineTool.from_defaults(
        query_engine=tutorials_index.as_query_engine(),
        name="Tutorials",
        description="Useful for answering questions about end-to-end tutorials and giving examples of specific use-cases."
    )

    contributing_tool = QueryEngineTool.from_defaults(
        query_engine=contributing_index.as_query_engine(),
        name="Contributing",
        description="Useful for answering questions about contributing to llama index, including how to contribute to the codebase and how to build documentation."
    )

    """Create Unified Query Egnine"""
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            getting_started_tool,
            community_tool,
            data_tool,
            agent_tool,
            model_tool,
            query_tool,
            supporting_tool,
            tutorials_tool,
            contributing_tool
        ],
        # enable this for streaming
        # response_synthesizer=get_response_synthesizer(streaming=True),
        verbose=True
    )

print('Success !')

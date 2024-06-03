from pathlib import Path


from llama_docs_bot.markdown_docs_reader import MarkdownDocsReader
from llama_index import SimpleDirectoryReader

# Make our printing look nice
from llama_index.schema import MetadataMode


def load_markdown_docs(filepath):
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

    # Load our documents from each folder.
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

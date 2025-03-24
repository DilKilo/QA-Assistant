import functions_framework
import config
from backing_up.backing_up import BackupClient
from updating.chroma_updating import ChromaClient
from fetching.html_processor import HtmlProcessor
from fetching.confluence_fetcher import ConfluenceFetcher
from embedding.embedder import VertexAITokenizer, VertexAIChromaEmbedder
from atlassian import Confluence

# TODO:
# - add looging
# - add error handling
# - think about links chunking
# - think about links embedding


@functions_framework.cloud_event
def event_handler(cloud_event):

    backup_client = BackupClient(
        source_storage_name=config.CHROMA_STORAGE_NAME,
        backup_storage_name=config.BACKUP_STORAGE_NAME
    )

    print("Starting backup...")

    backup_result = backup_client.backup(backups_number=config.BACKUPS_NUMBER)

    print(backup_result['message'])

    confluence_client = Confluence(
        url=config.CONFLUENCE_URL,
        username=config.CONFLUENCE_USERNAME,
        password=config.CONFLUENCE_PASSWORD
    )

    confluence_client = ConfluenceFetcher(
        confluence_client=confluence_client,
        max_workers=config.MAX_WORKERS
    )

    print("Fetching pages...")

    page_ids = confluence_client.get_all_space_pages(
        space=config.CONFLUENCE_SPACE,
        exclude_roots=config.EXCLUDE_PAGES_IDS
    )

    pages = confluence_client.get_pages_content(page_ids)

    print(f'Fetcting completed\nNumber of pages: {len(pages)}')

    print("Processing pages...")

    tokenizer = VertexAITokenizer(
        model_name=config.VERTEXAI_MODEL_NAME,
        task_type=config.VERTEXAI_TASK_TYPE
    )
    embedder = VertexAIChromaEmbedder(
        model_name=config.VERTEXAI_MODEL_NAME,
        task_type=config.VERTEXAI_TASK_TYPE
    )

    html_processor = HtmlProcessor(
        confluence_client=confluence_client,
        tokenizer=tokenizer,
        chunk_token_limit=config.CHUNK_TOKEN_LIMIT,
        count_by_text=config.COUNT_BY_TEXT,
        overlap=config.OVERLAP
    )

    documents, metadatas, empty_pages = html_processor.process_pages(pages)

    print(
        f'Processing completed\nNumber of documents: {len(documents)}\nNumber of empty pages: {len(empty_pages)}')

    print("Updating ChromaDB...")

    chroma_client = ChromaClient(
        chroma_host=config.CHROMA_HOST,
        chroma_port=config.CHROMA_PORT,
        embedder=embedder
    )

    update_result = chroma_client.update(
        collection_name=config.COLLECTION_NAME,
        documents=documents,
        metadatas=metadatas
    )

    print(update_result['message'])

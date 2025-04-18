import functions_framework
import config
from backing_up.backing_up import BackupClient
from updating.chroma_updating import ChromaClient
from fetching.html_processor import HtmlProcessor
from fetching.confluence_fetcher import ConfluenceFetcher
from embedding.embedder import VertexAITokenizer, VertexAIChromaEmbedder
from atlassian import Confluence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def event_handler(cloud_event):

    backup_client = BackupClient(
        source_storage_name=config.CHROMA_STORAGE_NAME,
        backup_storage_name=config.BACKUP_STORAGE_NAME
    )

    confluence_client = Confluence(
        url=config.CONFLUENCE_URL,
        username=config.CONFLUENCE_USERNAME,
        password=config.CONFLUENCE_PASSWORD
    )

    confluence_fetcher = ConfluenceFetcher(
        confluence_client=confluence_client,
        max_workers=config.MAX_WORKERS
    )

    tokenizer = VertexAITokenizer(
        model_name=config.VERTEXAI_MODEL_NAME,
        task_type=config.VERTEXAI_TASK_TYPE
    )
    embedder = VertexAIChromaEmbedder(
        model_name=config.VERTEXAI_MODEL_NAME,
        task_type=config.VERTEXAI_TASK_TYPE,
        dimensions=config.VERTEXAI_VECTOR_DIMENSIONS
    )

    chroma_client = ChromaClient(
        chroma_host=config.CHROMA_HOST,
        chroma_port=config.CHROMA_PORT,
        embedder=embedder
    )

    html_processor = HtmlProcessor(
        confluence_client=confluence_client,
        tokenizer=tokenizer,
        chunk_token_limit=config.CHUNK_OVERLAP,
        overlap=config.CHUNK_SIZE
    )

    logger.info("Starting backup...")

    backup_result = backup_client.backup(backups_number=config.BACKUPS_NUMBER)

    logger.info(backup_result['message'])

    logger.info("Fetching pages...")

    page_ids = confluence_fetcher.get_all_space_pages(
        space=config.CONFLUENCE_SPACE,
        exclude_roots=config.EXCLUDE_PAGES_IDS
    )

    pages = confluence_fetcher.get_pages_content(page_ids=page_ids)

    logger.info(f'Fetcting completed\nNumber of pages: {len(pages)}')

    logger.info("Processing pages...")

    documents, metadatas, empty_pages = html_processor.process_pages(
        pages, config.KEEP_TAGS)

    logger.info(
        f'Processing completed\nNumber of documents: {len(documents)}\nNumber of empty pages: {len(empty_pages)}')

    logger.info("Updating ChromaDB...")

    update_result = chroma_client.update(
        collection_name=config.COLLECTION_NAME,
        documents=documents,
        metadatas=metadatas
    )

    logger.info(update_result['message'])

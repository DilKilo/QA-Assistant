import base64
import json
from typing import Any, Dict
from vertexai.generative_models import GenerativeModel
from prompting.templates import PromptTemplate, SystemInstructions, SafetySettings
from retrieval.retriever import ChromaRetriever
from embedding.embedder import VertexAIChromaEmbedder
import functions_framework
import config
from google.auth import default
from googleapiclient.discovery import build
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/chat.messages']
creds, _ = default()
creds = creds.with_scopes(SCOPES)
chat = build('chat', 'v1', credentials=creds)


@functions_framework.cloud_event
def chat_app_handler(cloud_event: Any) -> Dict[str, Any]:
    """Handles messages from Google Chat via Pub/Sub.

    Args:
        cloud_event: Event object containing Pub/Sub message data

    Returns:
        Dictionary with acknowledgment for Pub/Sub
    """
    try:
        logger.info("Received Pub/Sub event")

        pubsub_message = cloud_event.data.get("message", {})
        if not pubsub_message.get("data"):
            logger.warning("No data in Pub/Sub message")
            return {"ack": "true"}

        data_encoded = pubsub_message["data"]
        data_bytes = base64.b64decode(data_encoded)
        chat_event = json.loads(data_bytes)

        logger.info(f"Processing Chat event type: {chat_event.get('type')}")

        if chat_event.get('type') == 'MESSAGE':
            query = chat_event.get('message', {}).get('text', '')
            space_name = chat_event.get('space', {}).get('name', '')
            logger.info(
                f"Received message: '{query}' from space: {space_name}")

            response_text = process_query(query)

            send_response(space_name, response_text)

        elif chat_event.get('type') == 'ADDED_TO_SPACE':
            space_name = chat_event.get('space', {}).get('name', '')
            logger.info(f"Added to space: {space_name}")

            welcome_text = "Thanks for adding me! I'll help answer your questions."
            send_response(space_name, welcome_text)

        return {"ack": "true"}
    except Exception as e:
        error_message = f"Error processing Google Chat event: {str(e)}"
        logger.error(error_message, exc_info=True)
        return {"ack": "true"}


def process_query(query: str) -> str:
    """Processes user query using RAG (Retrieval Augmented Generation).

    Args:
        query: User question text

    Returns:
        Generated response text based on retrieved context
    """
    try:
        logger.info(f"Processing query: {query}")

        embedder = VertexAIChromaEmbedder(
            model_name=config.VERTEXAI_MODEL_EMBEDDING_NAME,
            task_type=config.VERTEXAI_TASK_TYPE,
            dimensions=config.VECTOR_DIMENSIONS
        )

        retriever = ChromaRetriever(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedder
        )

        retrieval_result = retriever.retrieve(
            query=query,
            n_results=config.RETRIEVAL_RESULTS
        )

        context = retriever.format_context(retrieval_result)

        model = GenerativeModel(config.VERTEXAI_MODEL_NAME)

        system_instruction = SystemInstructions.qa_system_instruction()
        prompt = PromptTemplate.qa_prompt(query, context, system_instruction)

        safety_settings = SafetySettings.standard_settings()

        response = model.generate_content(
            prompt,
            generation_config=config.GENERATION_CONFIG,
            safety_settings=safety_settings
        )

        logger.info(f"Generated response for query: {query}")
        return response.text
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return f"Sorry, an error occurred while processing your request: {str(e)}"


def send_response(space_name: str, text: str) -> None:
    """Sends message back to Google Chat.

    Args:
        space_name: Google Chat space identifier
        text: Message text to send

    Returns:
        None
    """
    try:
        logger.info(f"Sending response to space: {space_name}")

        message = {
            'text': text
        }

        response = chat.spaces().messages().create(
            parent=space_name,
            body=message
        ).execute()

        logger.info(f"Message sent successfully, ID: {response.get('name')}")
    except Exception as e:
        logger.error(
            f"Error sending message to Google Chat: {str(e)}", exc_info=True)

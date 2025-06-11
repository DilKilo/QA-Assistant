from utils.google_chat_client import create_client_with_default_credentials
from utils.formater import TextFormater
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from prompting.templates import PromptTemplate, SystemInstructions, SafetySettings
from retrieval.retriever import ChromaRetriever
from embedding.embedder import VertexAIChromaEmbedder
import json
import base64
import json
import logging
from typing import Any, Dict

import config
import functions_framework
from embedding.embedder import VertexAIChromaEmbedder
from google.apps import chat_v1 as google_chat
from prompting.templates import PromptTemplate, SafetySettings, SystemInstructions
from retrieval.retriever import ChromaRetriever
from utils.formater import TextFormater
from utils.google_chat_client import create_client_with_default_credentials
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def chat_app(cloud_event) -> Dict[str, Any]:
    """
    Cloud Function handler for Google Chat app events.

    Processes incoming Chat messages via Pub/Sub, retrieves relevant information
    from a vector database, generates responses using VertexAI, and sends the
    responses back to the Chat space.

    Args:
        cloud_event: The CloudEvent containing the Chat message data.
    Returns:
        Dict[str, Any]: The response from the Chat API after creating a message.
    """
    chat = create_client_with_default_credentials(config.GOOGLE_CHAT_SCOPES)

    message_data = cloud_event.data.get("message").get("data")
    if not message_data:
        logger.error("No message data found in cloud event")
        return {"error": "No message data found"}

    event = json.loads(base64.b64decode(message_data).decode("utf-8"))

    processed_response = "Sorry, I couldn't process your request."

    if event and event.get("type") == "MESSAGE":
        query = event.get("message", {}).get("text")
        if query:
            processed_response = process_query(query)
    else:
        logger.info(f"Ignoring non-MESSAGE event of type: {event.get('type')}")
        return {"status": "ignored"}

    space_name = event.get("space", {}).get("name")
    thread_name = event.get("message", {}).get("thread", {}).get("name")

    if not space_name or not thread_name:
        logger.error(
            f"Missing space_name or thread_name: space={space_name}, thread={thread_name}"
        )
        return {"error": "Missing space or thread information"}

    request = google_chat.CreateMessageRequest(
        parent=space_name,
        message={"text": processed_response, "thread": {"name": thread_name}},
    )

    try:
        chat_response = chat.create_message(request)
        logger.info(f"Message sent successfully: {chat_response}")
        return chat_response
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}", exc_info=True)
        return {"error": f"Error sending message: {str(e)}"}


def process_query(query: str) -> str:
    """
    Processes a user query by retrieving relevant context and generating a response.

    This function handles the core processing logic:
    1. Embeds the query using VertexAI embeddings
    2. Retrieves relevant documents from a Chroma vector database
    3. Formats the context for the LLM prompt
    4. Generates a response using VertexAI generative models

    Args:
        query: The user's question or request text

    Returns:
        str: The generated response text, or an error message if processing fails
    """
    try:
        logger.info(f"Processing query: {query}")

        embedder = VertexAIChromaEmbedder(
            model_name=config.VERTEXAI_MODEL_EMBEDDING_NAME,
            task_type=config.VERTEXAI_TASK_TYPE,
            dimensions=config.VECTOR_DIMENSIONS,
        )

        retriever = ChromaRetriever(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedder,
        )

        retrieval_result = retriever.retrieve(
            query=query, n_results=config.RETRIEVAL_RESULTS
        )

        document_text = TextFormater.format_retrieval_documents(retrieval_result)
        system_instruction = SystemInstructions.qa_system_instruction()

        user_prompt = PromptTemplate.qa_prompt(query, document_text)
        content_user = Content(role="user", parts=[Part.from_text(user_prompt)])
        part_system = Part.from_text(system_instruction)

        model = GenerativeModel(
            model_name=config.VERTEXAI_MODEL_NAME,
            system_instruction=part_system,
        )

        safety_settings = SafetySettings.standard_settings()

        generative_config = GenerationConfig.from_dict(config.GENERATION_CONFIG)

        response = model.generate_content(
            contents=[content_user],
            generation_config=generative_config,
            safety_settings=safety_settings,
        )

        generation = json.loads(response.candidates[0].content.parts[0].text)
        generation_with_grounding = TextFormater.format_grounding_links(
            generation, retrieval_result["metadatas"]
        )

        result = TextFormater.format_retrieval_text_links(
            generation_with_grounding["answer"], retrieval_result["metadatas"]
        )

        logger.info(f"Generated response for query: {query}\n{result}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return f"Sorry, an error occurred while processing your request: {str(e)}"

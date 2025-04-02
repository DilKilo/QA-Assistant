import base64
from vertexai.generative_models import GenerativeModel
from prompting.templates import PromptTemplate, SystemInstructions, SafetySettings
from retrieval.retriever import ChromaRetriever
from embedding.embedder import VertexAIChromaEmbedder
import functions_framework
import config


@functions_framework.cloud_event
def event_handler(cloud_event):
    try:
        if isinstance(cloud_event.data, dict) and 'message' in cloud_event.data:
            message_data = cloud_event.data['message'].get('data', '')
            if message_data:
                query = base64.b64decode(message_data).decode('utf-8')
            else:
                query = ''
        else:
            if isinstance(cloud_event.data, bytes):
                query = cloud_event.data.decode('utf-8')
            elif isinstance(cloud_event.data, str):
                query = cloud_event.data
            else:
                query = str(cloud_event.data)

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

        print(response.text)

        return response.text

    except Exception as e:
        print(f"Failed to process request: {str(e)}")
        return {"error": str(e)}

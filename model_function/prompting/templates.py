from typing import List
from vertexai.generative_models import SafetySetting, HarmCategory, HarmBlockThreshold


class PromptTemplate:
    """
    A collection of template methods for creating prompts for various RAG scenarios.

    This class provides static methods that generate formatted prompts 
    for different types of language model interactions, with a focus on 
    question-answering tasks in RAG systems.
    """

    @staticmethod
    def qa_prompt(query: str, context: str, system_instruction: str = None) -> str:
        """
        Create a question-answering prompt that instructs the model to use only provided context.

        Args:
            query: The user's question
            context: Retrieved information to ground the model's response
            system_instruction: Optional system instruction to include in the prompt

        Returns:
            A formatted prompt string for question-answering
        """
        system_part = f"{system_instruction}\n\n" if system_instruction else ""

        return f"""
{system_part}Use only the following information to answer the question:
{context}

User question: {query}

Answer:
"""


class SystemInstructions:
    """
    A collection of system instruction templates for language models.

    System instructions define the model's behavior and role. This class
    provides predefined system instructions for different use cases
    to ensure consistent model behavior.
    """

    @staticmethod
    def qa_system_instruction() -> str:
        """
        Create a system instruction for question-answering tasks.

        This instruction emphasizes accuracy and honesty when the model
        doesn't have sufficient information to answer.

        Returns:
            A system instruction string for question-answering
        """
        return """
You are a highly accurate and concise question-answering assistant. Your task is to provide accurate answers based on 
the provided information. If the information is insufficient, honestly acknowledge this.
Do not make up facts or use information outside of the provided context.
Always respond in the same language as the user's question.

**Notes**:
- Provide accurate answers based EXCLUSIVELY on the provided information.
- If the information is insufficient, state: "Based on the provided information, I cannot answer this question."
- Do not make up facts or use information outside of the provided context.
- Always respond in the same language as the user's question.
- Format your output strictly as a JSON object following the given structure.

**Output structure**:
Output Structure:
{
    "answer": "The model answer to the question based on the provided context.",
    "sources_used": [List of document numbers used to answer, starting from 1],
    "answer_language": "Language of the answer in ISO 639-1 format (e.g., 'en' for English, 'ru' for Russian)"
}

**Rules**:
- List the document numbers you used for the answer in the `sources_used` array.
- Detect the question's language and set it correctly in `answer_language` using ISO 639-1 codes (eg. 'en','ru').
- Do not mention that you are using specific documents or context.
- If the answer is "Based on the provided information, I cannot answer this question.", then `sources_used` must be an empty list `[]`.
- If the provided context contains special symbols like `<[text]/>`, you must **preserve them exactly** in your answer without deleting, modifying, rewording, or reformatting.
- If the provided context contains name surname, you must **preserve them exactly** in your answer without deleting, modifying, rewording, or reformatting.
"""


class SafetySettings:
    """
    A collection of predefined safety settings for language model interactions.

    Safety settings control the model's content filtering behavior.
    This class provides different presets for various use cases, from
    permissive to strict content filtering.
    """

    @staticmethod
    def standard_settings() -> List[SafetySetting]:
        """
        Standard safety settings with medium-level filtering.

        Returns:
            List of SafetySetting objects with medium-level filtering for all harm categories
        """
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            )
        ]

    @staticmethod
    def permissive_settings() -> List[SafetySetting]:
        """
        More permissive safety settings with minimal filtering.

        Returns:
            List of SafetySetting objects that only block high-harm content
        """
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            )
        ]

    @staticmethod
    def strict_settings() -> List[SafetySetting]:
        """
        Strict safety settings with aggressive filtering.

        Returns:
            List of SafetySetting objects that block even low-harm content
        """
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            )
        ]

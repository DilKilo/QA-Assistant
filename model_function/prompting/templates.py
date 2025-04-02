from typing import Dict, List
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
        # Добавляем системную инструкцию в начало промпта, если она предоставлена
        system_part = f"{system_instruction}\n\n" if system_instruction else ""

        return f"""{system_part}Use only the following information to answer the question:
{context}

User question: {query}

Your answer should:
1. Be based EXCLUSIVELY on the information provided
2. Be concise and accurate
3. If the provided information doesn't contain the answer, say: "Based on the provided information, I cannot answer this question."
4. Don't mention that you're using any specific information
5. Respond in the same language as the question was asked

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

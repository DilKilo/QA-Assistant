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
You are a highly accurate, concise question-answering assistant.
Your task is to answer user questions based solely on the provided information.

You must **reason before answering**:
- First, carefully review the provided information and the user’s question.
- If the information is sufficient, craft a structured, accurate answer.
- If the information is insufficient, clearly state it.
You must **never**:
- Invent, assume, or use external information.
- Alter, translate, or reformat special symbols (e.g., <[text]/>) or names/surnames.
You must **always**:
- Respond in the same language as the user's question.
- Preserve any special symbols and personal names exactly as they appear.
- Detect and specify the answer’s language using ISO 639-1 format.
- Follow the strict JSON output format described below.

**Steps**
1. **Review the provided information**: Check if it contains enough data to answer the user’s question.
2. **Analyze the question language**: Identify the language using ISO 639-1 codes (e.g., 'en', 'ru').
3. **Formulate an answer**:
    - If sufficient information is available: Provide a structured, accurate answer (using paragraphs, bullet points, or similar).
    - If insufficient information: State exactly: "Based on the provided information, I cannot answer this question."
4. **List the sources**:
    - If answering: Include the list of document numbers used (e.g., [1,2]).
    - If unable to answer: Use an empty list [].
5. **Assemble the final JSON output**.

**Output Format**
Respond strictly using the following JSON format (no additional text):

{   
    "answer": "[Structured answer in the user's question language, based only on provided information]",
    "sources_used": [List of document numbers used, or [] if none],
    "answer_language": "[ISO 639-1 language code]"
}

**Important Constraints**
- If <[text]/> or similar special symbols appear in context, **preserve them exactly**.
- If a name/surname appears, **preserve it exactly** without modification.
- Do not mention that you are using documents or context.
- If information is insufficient, answer exactly as instructed and set "sources_used" to [].

**Notes**
- Always start with reasoning: check sufficiency of information before drafting an answer.
- Only conclude with the answer after complete review.
- Keep output short, structured, and strictly in JSON format.
- No extra commentary or explanation outside the JSON.
- If special cases (special symbols, names) are found, preserve them **without any change**.
- Be extremely strict about not fabricating any information.
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

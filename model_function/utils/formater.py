from typing import Dict, Any, List
import json


class TextFormater:
    @staticmethod
    def format_retrieval_documents(retrieval_result: Dict[str, Any]) -> str:
        if (
            not retrieval_result
            or "documents" not in retrieval_result
            or not retrieval_result["documents"]
        ):
            return ""
        if (
            not retrieval_result
            or "metadatas" not in retrieval_result
            or not retrieval_result["metadatas"]
        ):
            return ""

        documents_text = (
            retrieval_result["documents"][0]
            if isinstance(retrieval_result["documents"][0], list)
            else retrieval_result["documents"]
        )
        documents_metadata = (
            retrieval_result["metadatas"][0]
            if isinstance(retrieval_result["metadatas"][0], list)
            else retrieval_result["metadatas"]
        )

        text = ""
        for i, (document, metadata) in enumerate(
            zip(documents_text, documents_metadata)
        ):
            text += f"[Document {i + 1}] Title:[{metadata['title']}] {document}\n"
        return text

    @staticmethod
    def format_retrieval_text_links(
        response_answer: str, retrieval_metadata: List[Any]
    ) -> str:
        documents_metadata = (
            retrieval_metadata[0]
            if isinstance(retrieval_metadata[0], list)
            else retrieval_metadata
        )

        replacements = {}
        for metadata_item in documents_metadata:
            document_links = json.loads(metadata_item["links"])
            for topic, link in document_links.items():
                placeholder = f"<[{topic}]/>"
                replace_value = f"<{link}|{topic}>" if link else topic
                replacements[placeholder] = replace_value

        for placeholder, replacement in replacements.items():
            response_answer = response_answer.replace(placeholder, replacement)

        return response_answer

    @staticmethod
    def format_grounding_links(
        response_data: Dict[str, Any], retrieval_metadata: List[Any]
    ) -> Dict[str, Any]:
        enriched = response_data.copy()
        metadata_entries = (
            retrieval_metadata[0]
            if isinstance(retrieval_metadata[0], list)
            else retrieval_metadata
        )

        source_page_urls = [entry.get("page_url", "") for entry in metadata_entries]

        sources_used = enriched.get("sources_used", []) or []

        if sources_used:
            formatted_urls = []
            seen_urls = set()
            for source_id in sources_used:
                if 0 < source_id <= len(source_page_urls):
                    url = source_page_urls[source_id - 1]
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        formatted_urls.append(f"{url}")

            header = (
                "Источники:" if enriched.get("answer_language") == "ru" else "Sources:"
            )

            enriched["answer"] += f"\n\n{header}\n" + "\n".join(formatted_urls)
        return enriched

from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Callable, Optional, Dict, Any, Union
from functools import lru_cache
import re
from atlassian import Confluence
import time
import uuid


class ConfluenceResolver:
    """Implementation of LinkResolver for Confluence."""

    def __init__(self, confluence_client: Confluence):
        """
        Initialize the ConfluenceResolver with a Confluence client.

        Args:
            confluence_client: Authenticated Confluence client instance
        """
        self.confluence_client = confluence_client

    @lru_cache(maxsize=1024)
    def resolve_page_link(self, page_title: str, space_key: str) -> tuple:
        """
        Resolve a Confluence page link by its title and space key.

        Args:
            page_title: Title of the Confluence page
            space_key: Space key where the page is located

        Returns:
            Tuple with page title and link, or just title if resolution fails
        """
        if not space_key:
            space_key = "QD"

        try:
            page = self.confluence_client.get_page_by_title(
                space=space_key, title=page_title)

            if not page or page["status"] != "current":
                return page_title, None

            page = self.confluence_client.get_page_by_id(page_id=page["id"])
            page_link = page["_links"]["base"] + page["_links"]["webui"]

            return page_title, page_link
        except Exception as e:
            print(f"Error resolving page link: {e}")
            return page_title, None

    @lru_cache(maxsize=1024)
    def resolve_user_link(self, account_id: str) -> str:
        """
        Resolve a Confluence user link by account ID.

        Args:
            account_id: Confluence user account ID

        Returns:
            Formatted string with user name, or account ID if resolution fails
        """
        try:
            user = self.confluence_client.get_user_details_by_accountid(
                accountid=account_id)

            if not user:
                return f'User account_id: {account_id}'

            return f'{user["publicName"]}'
        except Exception as e:
            print(f"Error resolving user link: {e}")
            return f'User account_id: {account_id}'


class TokenCounter:
    """Counts tokens in text using a provided tokenizer function."""

    def __init__(self, tokenizer: Callable[[str], int]):
        """
        Initialize the TokenCounter with a tokenizer function.

        Args:
            tokenizer: A function that converts a string to token count
        """
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the provided tokenizer.

        Args:
            text: The input text to tokenize and count

        Returns:
            Number of tokens in the text
        """
        return self.tokenizer(text)


class HtmlCleaner:
    """Cleans and processes HTML content."""

    def __init__(self, confluence_resolver: ConfluenceResolver):
        """
        Initialize HtmlCleaner with a ConfluenceResolver.

        Args:
            confluence_resolver: A resolver for Confluence links and users
        """
        self.confluence_resolver = confluence_resolver

    def clean_html(self, html: str, keep_tags: Optional[set] = None) -> str:
        """
        Clean HTML by either keeping only specified tags or removing all tags.

        Args:
            html: HTML content to clean
            keep_tags: Set of tag names to preserve (if None, all tags are removed)

        Returns:
            Cleaned HTML or plain text
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            if keep_tags:
                for tag in soup.find_all():
                    if tag.name not in keep_tags:
                        tag.unwrap()
                cleaned_html = soup.prettify()
            else:
                cleaned_html = soup.get_text(separator=' ', strip=True)
                cleaned_html = re.sub(
                    r'[\xa0\u200b\t\n\r]+', ' ', cleaned_html)
                cleaned_html = re.sub(r'\s{2,}', ' ', cleaned_html).strip()

            return cleaned_html
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return re.sub(r'\s+', ' ', html).strip()

    def process_links(self, html: str) -> Dict[str, Any]:
        """
        Process all links in HTML, resolving references to pages and users.

        Args:
            html: HTML content with links

        Returns:
            HTML with links replaced by text representations
        """
        soup = BeautifulSoup(html, "html.parser")

        metadata_confluence = self._process_confluence_links(soup)

        metadata_html = self._process_html_links(soup)

        metadata = metadata_confluence | metadata_html

        return {"page_content": str(soup), "metadata": metadata}

    def _process_confluence_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Process Confluence-specific links in place.

        Args:
            soup: BeautifulSoup object containing the HTML with Confluence links
        """
        metadata = {}
        for tag in soup.find_all('ac:link'):
            try:
                ripage_tag = tag.find("ri:page")
                riuser_tag = tag.find("ri:user")

                if ripage_tag:
                    title = ripage_tag.get('ri:content-title', '')
                    space_key = ripage_tag.get('ri:space-key', '')
                    page_title, page_link = self.confluence_resolver.resolve_page_link(
                        title, space_key)

                    metadata[page_title] = page_link

                    replacement = f'<[{page_title}]/>'

                elif riuser_tag:
                    account_id = riuser_tag.get('ri:account-id', '')
                    replacement = self.confluence_resolver.resolve_user_link(
                        account_id)
                else:
                    replacement = tag.get_text()

                tag.replace_with(replacement)
            except Exception as e:
                print(f"Error processing confluence link: {e}")
                tag.replace_with(tag.get_text())

        return metadata

    def _process_html_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Process standard HTML links in place.

        Args:
            soup: BeautifulSoup object containing the HTML with standard links
        """
        metadata = {}
        for tag in soup.find_all('a'):
            try:
                href = tag.get('href', '')
                text = tag.get_text().strip()

                if not text:
                    tag.replace_with("")
                    continue

                if text.startswith("http"):
                    text = str(uuid.uuid4())

                replacement = f'<[{text}]/>'
                tag.replace_with(replacement)
                metadata[text] = href
            except Exception as e:
                print(f"Error processing html link: {e}")
                tag.replace_with(tag.get_text().strip())
        return metadata


class DocumentChunker:
    """
    Splits HTML documents into semantically meaningful chunks
    while preserving structure and respecting token limits.
    """

    def __init__(
        self,
        html_cleaner: HtmlCleaner,
        token_counter: TokenCounter,
        chunk_token_limit: int = 512,
        overlap: float = 0.0
    ):
        """
        Initialize the DocumentChunker with required components and settings.

        Args:
            html_cleaner: HTML cleaner component for processing HTML
            token_counter: Token counter component for measuring token counts
            chunk_token_limit: Maximum number of tokens per chunk
            overlap: Percentage of overlap between chunks (0.0-0.5)
        """
        self.html_cleaner = html_cleaner
        self.token_counter = token_counter
        self.chunk_token_limit = chunk_token_limit
        self.overlap = min(max(0.0, overlap), 0.5)

        self.chunks = []
        self.current_chunk = {'page_content': '', 'metadata': {}}
        self.last_chunk_content = None

    def count_tokens(self, html_string: str) -> int:
        """
        Count tokens in processed HTML string.

        Args:
            html_string: Raw HTML string

        Returns:
            Token count after processing
        """
        processed_chunk = self.html_cleaner.process_links(html_string)
        processed_chunk["page_content"] = self.html_cleaner.clean_html(
            processed_chunk["page_content"])

        return self.token_counter.count_tokens(processed_chunk["page_content"])

    def chunk_document(self, html: str) -> List[Dict[str, Any]]:
        """
        Split an HTML document into chunks while preserving structure.

        Args:
            html: The HTML string to chunk

        Returns:
            List of text chunks
        """
        self.chunks = []
        self.current_chunk = {'page_content': '', 'metadata': {}}
        self.last_chunk_content = None

        try:
            soup = BeautifulSoup(html, "html.parser")
            root = soup.body if soup.body else soup

            elements = [el for el in root.children if not (
                isinstance(el, NavigableString) and not el.strip()
            )]

            self._process_elements(elements)
            self._finalize_chunk()

            return self.chunks
        except Exception as e:
            print(f"Error chunking document: {e}")
            return self._fallback_chunking(html)

    def _process_elements(self, elements: List[Union[Tag, NavigableString]]) -> None:
        """
        Process a list of HTML elements for chunking.

        This method iterates through the elements, handling headers with their
        content specially, and processes the regular elements individually.

        Args:
            elements: List of HTML elements to process
        """
        i = 0
        while i < len(elements):
            element = elements[i]

            if self._is_header(element) and i + 1 < len(elements):
                next_element = elements[i + 1]

                if not self._is_header(next_element):
                    self._process_header_with_content(element, next_element)
                    i += 2
                    continue

            self._process_regular_element(element)
            i += 1

    def _is_header(self, element: Union[Tag, NavigableString]) -> bool:
        """
        Check if an element is a header tag.

        Args:
            element: HTML element to check

        Returns:
            True if element is a header tag (h1-h6), False otherwise
        """
        return (isinstance(element, Tag) and
                element.name in ["h1", "h2", "h3", "h4", "h5", "h6"])

    def _process_header_with_content(self, header: Tag, content: Union[Tag, NavigableString]) -> None:
        """
        Process a header element together with its content.

        Args:
            header: The header tag element
            content: The content element following the header
        """
        combined_html = str(header) + str(content)

        if self.count_tokens(self.current_chunk["page_content"] + combined_html) <= self.chunk_token_limit:
            self.current_chunk["page_content"] += combined_html
            return

        self._finalize_chunk()

        if self.count_tokens(self.current_chunk["page_content"] + combined_html) <= self.chunk_token_limit:
            self.current_chunk["page_content"] += combined_html
        else:
            self._split_and_add_content(combined_html)

    def _process_regular_element(self, element: Union[Tag, NavigableString]) -> None:
        """
        Process a non-header element.

        Args:
            element: The HTML element to process
        """
        element_html = str(element)

        if self.count_tokens(self.current_chunk["page_content"] + element_html) <= self.chunk_token_limit:
            self.current_chunk["page_content"] += element_html
        else:
            if self.count_tokens(element_html) > self.chunk_token_limit:
                self._split_and_add_content(element_html)
            else:
                self._finalize_chunk()
                self.current_chunk["page_content"] += element_html

    def _split_and_add_content(self, content: str) -> None:
        """
        Split content and add to chunks.

        Args:
            content: HTML content string to split and add
        """
        soup = BeautifulSoup(content, "html.parser")

        for element in list(soup.children):
            if isinstance(element, Tag):
                for part in self._split_element(element):
                    if self.count_tokens(self.current_chunk["page_content"] + part) > self.chunk_token_limit:
                        self._finalize_chunk()
                    self.current_chunk["page_content"] += part

    def _split_element(self, element) -> List[str]:
        """
        Recursively split an HTML element if it exceeds token limit.

        Args:
            element: BeautifulSoup element (Tag or NavigableString)

        Returns:
            List of HTML strings representing the split element.
        """
        element_html = str(element)
        if self.count_tokens(element_html) <= self.chunk_token_limit:
            return [element_html]

        if isinstance(element, NavigableString):
            return self._split_text_node(element)

        if isinstance(element, Tag):
            return self._split_tag(element)

        return [element_html]

    def _split_text_node(self, text_node: NavigableString) -> List[str]:
        """
        Split a text node into smaller chunks based on sentences and words.

        Args:
            text_node: Text node to split

        Returns:
            List of text chunks that fit within token limit
        """
        sentences = re.split(r'(?<=[.!?])\s+', str(text_node))

        if len(sentences) > 1:
            chunks = []
            current = ""

            for sentence in sentences:
                if not sentence.strip():
                    continue

                test_chunk = current + " " + sentence if current else sentence
                if self.count_tokens(test_chunk) <= self.chunk_token_limit:
                    current = test_chunk
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sentence

            if current:
                chunks.append(current.strip())

            if chunks:
                return chunks

        words = str(text_node).split()
        chunks = []
        current = ""

        average_tokens_per_word = max(
            1, self.count_tokens(str(text_node)) / len(words))
        window_size = int(self.chunk_token_limit / average_tokens_per_word)

        for i in range(0, len(words), window_size):
            word_group = words[i:i + window_size]
            word_chunk = " ".join(word_group)

            if self.count_tokens(word_chunk) <= self.chunk_token_limit:
                chunks.append(word_chunk)
            else:
                current = ""
                for word in word_group:
                    test_chunk = current + " " + word if current else word
                    if self._cached_count_tokens(test_chunk) <= self.chunk_token_limit:
                        current = test_chunk
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = word

                if current:
                    chunks.append(current.strip())

        return chunks

    def _split_tag(self, tag: Tag) -> List[str]:
        """
        Split a tag by processing its children.

        Args:
            tag: HTML tag to split

        Returns:
            List of HTML strings with the tag's content split into chunks
        """
        children = list(tag.children)
        parts = []
        current_group = ""

        for child in children:
            child_str = str(child)
            test_group = current_group + child_str

            if self.count_tokens(test_group) <= self.chunk_token_limit:
                current_group = test_group
            else:
                if current_group:
                    parts.append(current_group)

                if self.count_tokens(child_str) > self.chunk_token_limit:
                    parts.extend(self._split_element(child))
                else:
                    current_group = child_str

        if current_group:
            parts.append(current_group)

        wrapped_parts = []
        for part in parts:
            tag_copy = f"<{tag.name}"
            for attr, value in tag.attrs.items():
                tag_copy += f' {attr}="{value}"'
            tag_copy += f">{part}</{tag.name}>"
            wrapped_parts.append(tag_copy)

        return wrapped_parts

    def _finalize_chunk(self) -> None:
        """
        Add the current chunk to the list of chunks and reset it.

        This method processes the current chunk, cleans it, and adds it to 
        the chunks list if it contains non-empty content.
        """
        if self.current_chunk["page_content"].strip():
            self.current_chunk = self.html_cleaner.process_links(
                self.current_chunk["page_content"])
            self.current_chunk["page_content"] = self.html_cleaner.clean_html(
                self.current_chunk["page_content"])

            if self.current_chunk["page_content"].strip():
                self.chunks.append(self.current_chunk)
                self.last_chunk_content = self.current_chunk["page_content"]

        self.current_chunk = {'page_content': '', 'metadata': {}}

        if self.overlap > 0 and self.last_chunk_content:
            overlap_size = int(self.chunk_token_limit * self.overlap)
            if overlap_size > 0:
                words = self.last_chunk_content.split()
                overlap_words = words[-min(len(words), overlap_size):]
                overlap_text = " ".join(overlap_words)

                overlap_tokens = self.token_counter.count_tokens(overlap_text)
                if overlap_tokens <= self.chunk_token_limit:
                    self.current_chunk["page_content"] = overlap_text

    def _get_sliding_window_chunks(self, text: str) -> List[Dict[str, Any]]:

        if not text or self.count_tokens(text) <= self.chunk_token_limit:
            return [{'page_content': text, 'metadata': {}}]

        words = text.split()
        result_chunks = []

        average_tokens_per_word = max(1, self.count_tokens(text) / len(words))
        window_size = int(self.chunk_token_limit / average_tokens_per_word)

        if window_size <= 0:
            window_size = 1

        step_size = int(window_size * (1.0 - self.overlap))
        step_size = max(1, step_size)

        for i in range(0, len(words), step_size):
            window_words = words[i:i + window_size]
            window_text = " ".join(window_words)

            while self.count_tokens(window_text) > self.chunk_token_limit and len(window_words) > 1:
                window_words.pop()
                window_text = " ".join(window_words)

            if window_text.strip():
                processed_chunk = self.html_cleaner.process_links(window_text)
                processed_chunk["page_content"] = self.html_cleaner.clean_html(
                    processed_chunk["page_content"])

                if processed_chunk["page_content"].strip():
                    result_chunks.append(processed_chunk)

        return result_chunks

    def _fallback_chunking(self, html: str) -> List[Dict[str, Any]]:
        """
        Fallback method for chunking if the main method fails.

        Args:
            html: HTML content to chunk

        Returns:
            List of text chunks created using a simpler chunking strategy
        """
        processed_chunk = self.html_cleaner.process_links(html)
        processed_chunk["page_content"] = self.html_cleaner.clean_html(
            processed_chunk["page_content"])

        if self.overlap > 0:
            return self._get_sliding_window_chunks(processed_chunk["page_content"])

        paragraphs = re.split(r'\n\s*\n', processed_chunk["page_content"])

        chunks = []
        current_chunk = {'page_content': '', 'metadata': {}}

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            if self.count_tokens(current_chunk["page_content"] + paragraph) <= self.chunk_token_limit:
                current_chunk["page_content"] += " " + \
                    paragraph if current_chunk["page_content"] else paragraph
            else:
                if current_chunk["page_content"]:
                    chunks.append(current_chunk)

                if self.count_tokens(paragraph) > self.chunk_token_limit:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = {'page_content': '', 'metadata': {}}

                    for sentence in sentences:
                        if not sentence.strip():
                            continue

                        if self.count_tokens(current_chunk["page_content"] + sentence) <= self.chunk_token_limit:
                            current_chunk["page_content"] += " " + \
                                sentence if current_chunk["page_content"] else sentence
                        else:
                            if current_chunk["page_content"]:
                                chunks.append(current_chunk)
                            current_chunk = {
                                'page_content': sentence, 'metadata': {}}
                else:
                    current_chunk = {'page_content': paragraph, 'metadata': {}}

        if current_chunk["page_content"]:
            chunks.append(current_chunk)

        return chunks


class HtmlProcessor:
    """
    Facade class for HTML processing, providing a simplified interface
    for cleaning, processing links, and chunking HTML documents.
    """

    def __init__(
        self,
        confluence_client: Confluence,
        tokenizer: Callable[[str], int],
        chunk_token_limit: int = 512,
        overlap: float = 0.0
    ):
        """
        Initialize HtmlProcessor with component dependencies.

        Args:
            confluence_client: Authenticated Confluence client instance
            tokenizer: Function that counts tokens in a string
            chunk_token_limit: Maximum number of tokens per chunk
            overlap: Percentage of overlap between chunks (0.0-0.5)
        """
        self.confluence_resolver = ConfluenceResolver(confluence_client)
        self.token_counter = TokenCounter(tokenizer)
        self.html_cleaner = HtmlCleaner(self.confluence_resolver)
        self.document_chunker = DocumentChunker(
            self.html_cleaner,
            self.token_counter,
            chunk_token_limit,
            overlap
        )

    def clean_html(self, html: str, keep_tags: Optional[set] = None) -> str:
        """
        Clean HTML by removing or keeping specified tags.

        Args:
            html: HTML content to clean
            keep_tags: Set of tag names to preserve (if None, all tags are removed)

        Returns:
            Cleaned HTML or plain text
        """
        return self.html_cleaner.clean_html(html, keep_tags)

    def replace_link_tag(self, text: str) -> Dict[str, Any]:
        """
        Replace Confluence and html link tags with text representation.

        Args:
            text: HTML text containing Confluence links

        Returns:
            HTML with links replaced by text representations
        """
        processed = self.html_cleaner.process_links(text)
        return processed

    def count_tokens(self, html_string: str) -> int:
        """
        Count tokens in an HTML string after processing.

        Args:
            html_string: HTML content to count tokens in

        Returns:
            Number of tokens in the processed HTML
        """
        return self.document_chunker.count_tokens(html_string)

    def chunk_document(self, html: str) -> List[Dict[str, Any]]:
        """
        Split an HTML document into chunks.

        Args:
            html: HTML document to split

        Returns:
            List of text chunks suitable for embedding
        """
        return self.document_chunker.chunk_document(html)

    def process_pages(self, pages: List[Dict[str, Any]], keep_tags: Optional[set] = None) -> tuple:
        """
        Process a list of Confluence pages, chunking each page.

        Args:
            pages: List of page objects from Confluence API
            keep_tags: Set of tag names to preserve

        Returns:
            Tuple of (documents, metadatas, empty_pages) where:
            - documents: List of text chunks from all pages
            - metadatas: List of metadata dictionaries for each chunk
            - empty_pages: List of page IDs that couldn't be processed
        """
        documents = []
        metadatas = []
        empty_pages = []
        duration_times = []

        number_pages = len(pages)

        for idx, page in enumerate(pages):
            try:
                average_duration_time = sum(
                    duration_times) / len(duration_times) if duration_times else 0
                print(
                    f'Processing page: {idx}/{number_pages}\nAverage duration time: {average_duration_time}')

                start_time = time.time()
                page_id = page.get('id', 'unknown')

                html_text = page['body']['storage']['value']

                html_text = self.clean_html(html_text, keep_tags)
                chunks = self.chunk_document(html_text)

                if not chunks:
                    empty_pages.append(page_id)
                    continue

                page_url = page['_links']['base'] + page['_links']['webui']
                metadata = [
                    {
                        'title': page['title'],
                        'page_id': page_id,
                        'page_url': page_url,
                        'links': chunk['metadata']
                    } for chunk in chunks
                ]

                documents.extend(chunk["page_content"] for chunk in chunks)
                metadatas.extend(metadata)
                duration_times.append(time.time() - start_time)

            except Exception as e:
                page_id = page.get('id', 'unknown')
                page_title = page.get('title', 'unknown')
                empty_pages.append(page_id)
                print(
                    f"Error processing page {page_id} ({page_title}): {str(e)}")

        return documents, metadatas, empty_pages

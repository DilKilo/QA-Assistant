from typing import List, Set, Dict, Optional, Any
from atlassian import Confluence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


class ConfluenceFetcher:
    """
    Fetcher for retrieving data from Confluence API with optimized performance.

    This class provides methods to efficiently fetch pages and their content
    from Confluence, with support for parallel processing and caching.
    """

    def __init__(self, confluence_client: Confluence, max_workers: int = 5):
        """
        Initialize the Confluence fetcher.

        Args:
            confluence_client: Confluence client
            max_workers: Maximum number of parallel workers for API requests
        """
        self.confluence = confluence_client
        self.max_workers = max_workers

    @lru_cache(maxsize=128)
    def get_page_children(self, page_id: str, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Get all child pages of a given page with caching.

        Args:
            page_id: ID of the parent page
            batch_size: Number of results to fetch per API call

        Returns:
            List of child page objects
        """
        try:
            children = []
            start = 0

            while True:
                batch = self.confluence.get_page_child_by_type(
                    page_id, type="page", start=start, limit=batch_size)

                if not batch:
                    break

                children.extend(batch)
                start += len(batch)

                if len(batch) < batch_size:
                    break

            return children
        except Exception as e:
            print(
                f"Error fetching children of page {page_id}: {e}")
            raise

    def get_page_tree(self, root_id: str) -> Set[str]:
        """
        Get all page IDs in the tree rooted at the given page.

        Uses breadth-first search to traverse the page hierarchy.

        Args:
            root_id: ID of the root page

        Returns:
            Set of all page IDs in the tree
        """
        result = {root_id}
        queue = [root_id]

        while queue:
            current_batch = queue[:min(len(queue), self.max_workers * 2)]
            queue = queue[len(current_batch):]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                child_results = list(executor.map(
                    self.get_page_children, current_batch))

            for children in child_results:
                child_ids = [child['id'] for child in children]
                queue.extend(child_ids)
                result.update(child_ids)

        return result

    def get_excluded_pages(self, exclude_roots: List[str]) -> Set[str]:
        """
        Get all page IDs that should be excluded based on root exclusion pages.

        Args:
            exclude_roots: List of page IDs whose entire subtrees should be excluded

        Returns:
            Set of page IDs to exclude
        """
        if not exclude_roots:
            return set()

        result = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tree_results = executor.map(self.get_page_tree, exclude_roots)

        for tree in tree_results:
            result.update(tree)

        return result

    def get_all_space_pages(self, space: str, exclude_roots: Optional[List[str]] = None,
                            batch_size: int = 100) -> List[str]:
        """
        Get all page IDs from a space, with optional exclusions.

        Args:
            space: Space key
            exclude_roots: Optional list of page IDs to exclude (including their children)
            batch_size: Number of results to fetch per API call

        Returns:
            List of page IDs in the space, excluding any specified subtrees
        """

        all_pages = []
        start = 0

        while True:
            try:
                batch = self.confluence.get_all_pages_from_space(
                    space=space, start=start, limit=batch_size,
                    status="current", expand=None)

                if not batch:
                    break

                all_pages.extend(batch)
                start += len(batch)

                if len(batch) < batch_size:
                    break

            except Exception as e:
                print(
                    f"Error fetching pages from space {space}: {e}")
                raise

        page_ids = {page['id']
                    for page in all_pages if page.get('status') == 'current'}

        if exclude_roots:
            excluded_ids = self.get_excluded_pages(exclude_roots)
            page_ids -= excluded_ids
        return list(page_ids)

    def get_pages_content(self, page_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch content for multiple pages in parallel.

        Args:
            page_ids: Set of page IDs to fetch

        Returns:
            List of page objects with content
        """

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for page_id in page_ids:
                futures.append(
                    executor.submit(
                        self.confluence.get_page_by_id,
                        page_id=page_id,
                        expand="body.storage"
                    )
                )

            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error fetching page content: {e}")

        return results

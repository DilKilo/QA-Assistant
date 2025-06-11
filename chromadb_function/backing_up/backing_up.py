from typing import Any, Dict, List, Optional

from google.cloud import storage

from backing_up.utilities import get_current_timestamp


class BackupClient:
    """
    Class responsible for backing up ChromaDB data between Google Cloud Storage buckets.

    This class manages the process of creating backups from a source bucket to a destination bucket,
    maintaining a specified number of most recent backups.
    """

    def __init__(self, source_storage_name: str, backup_storage_name: str):
        """
        Initialize the backup service with source and destination bucket names.

        Args:
            source_storage_name: Name of the source GCS bucket
            backup_storage_name: Name of the backup GCS bucket
        """
        self.storage_client = storage.Client()
        self.source_storage_name = source_storage_name
        self.backup_storage_name = backup_storage_name

    def get_folders(self, bucket: storage.Bucket) -> Dict[str, List[storage.Blob]]:
        """
        Group blobs by their top-level folder (backup).

        Args:
            bucket: GCS bucket to search in

        Returns:
            Dictionary mapping folder names to lists of blobs in that folder
        """
        try:
            blobs = list(bucket.list_blobs())
        except Exception:
            raise

        folders: Dict[str, List[storage.Blob]] = {}

        for blob in blobs:
            parts = blob.name.split("/", 1)
            if len(parts) > 1:
                folder_name = parts[0] + "/"
                if folder_name not in folders:
                    folders[folder_name] = []
                folders[folder_name].append(blob)
        return folders

    def get_oldest_blob(self, bucket: storage.Bucket) -> Optional[storage.Blob]:
        """
        Find the oldest blob in the given bucket based on creation time.

        Args:
            bucket: GCS bucket to search in

        Returns:
            The oldest blob or None if bucket is empty
        """
        try:
            blobs = list(bucket.list_blobs())
        except Exception:
            raise

        if not blobs:
            return None

        oldest_blob = min(blobs, key=lambda blob: blob.time_created)
        return oldest_blob

    def delete_folder(self, bucket: storage.Bucket, folder_name: str) -> Dict[str, Any]:
        """
        Delete all blobs in a specific folder.

        Args:
            bucket: GCS bucket containing the folder
            folder_name: Name of the folder to delete
        Returns:
            Dictionary containing success status and message
        """
        result = {"success": False, "message": ""}

        try:
            blobs = list(bucket.list_blobs(prefix=folder_name))

            if not blobs:
                result["success"] = True
                result["message"] = (
                    f"No blobs found with prefix '{folder_name}' in bucket {bucket.name}"
                )
                return result

            for blob in blobs:
                blob.delete()

            result["success"] = True
            result["message"] = (
                f"Successfully deleted {len(blobs)} blobs with prefix '{folder_name}'"
            )
            return result

        except Exception:
            raise

    def backup(self, backups_number: int) -> Dict[str, Any]:
        """
        Perform backup from source to destination bucket.

        This method copies all blobs from source to destination bucket into a timestamped
        directory. If the number of backups exceeds backups_number, the oldest backup will
        be deleted.

        Args:
            backups_number: Maximum number of backups to keep

        Returns:
            Dictionary containing success status and message
        """

        result = {"success": False, "message": ""}

        try:
            source_bucket = self.storage_client.bucket(self.source_storage_name)
            destination_bucket = self.storage_client.bucket(self.backup_storage_name)

            if not source_bucket.exists():
                raise ValueError(
                    f"Source bucket {self.source_storage_name} does not exist"
                )
            if not destination_bucket.exists():
                raise ValueError(
                    f"Destination bucket {self.backup_storage_name} does not exist"
                )

            source_blobs = list(source_bucket.list_blobs())

            if not source_blobs:
                result["success"] = True
                result["message"] = "No blobs found in source bucket"
                return result

            backup_folders = self.get_folders(destination_bucket)

            if len(backup_folders) >= backups_number:
                oldest_blob = self.get_oldest_blob(destination_bucket)
                if oldest_blob:
                    folder_name = oldest_blob.name.split("/")[0]
                    result = self.delete_folder(destination_bucket, folder_name)

            current_time = get_current_timestamp()

            for blob in source_blobs:
                destination_blob_name = f"{current_time}/{blob.name}"
                source_blob = source_bucket.blob(blob.name)

                source_bucket.copy_blob(
                    source_blob, destination_bucket, destination_blob_name
                )

            result["success"] = True
            result["message"] = f"Successfully backed up {len(source_blobs)} blobs"
            return result

        except Exception:
            raise

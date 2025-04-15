from google.apps import chat_v1 as google_chat
from google.auth import default


def create_client_with_default_credentials(scopes: list[str]):
    credentials, _ = default(scopes=scopes)
    client = google_chat.ChatServiceClient(credentials=credentials)
    return client

import os
import logging
import requests
import time
import numpy as np
import json
from urllib.request import pathname2url
import rdflib
from sparql_handler import SPARQLHandler
from embedding_handler import EmbeddingHandler


class Bot:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.DEBUG)

        # Centralized loading of all necessary files (RDF graph, embeddings, dictionaries)
        self.load_files()

        # Initialize EmbeddingHandler with loaded embeddings and IDs
        self.embedding_handler = EmbeddingHandler(
            self.entity_embeddings,
            self.relation_embeddings,
            self.ent2id,
            self.rel2id,
            None  # Initially None, will update after initializing SPARQLHandler
        )

        # Initialize SPARQLHandler with RDF graph and dictionaries
        self.sparql_handler = SPARQLHandler(
            self.rdf_graph,
            self.entity_dict,
            self.property_dict,
            self.embedding_handler
        )

        # Link SPARQLHandler to EmbeddingHandler after initializing both
        self.embedding_handler.sparql_handler = self.sparql_handler

        # Initialize other bot attributes
        self.base_url = "https://speakeasy.ifi.uzh.ch"
        self.session_token = None
        self.cache = {}
        self.processed_messages = set()
        self.headers = {'Content-Type': 'application/json'}
        self.bot_alias = None
        self.last_timestamp = 0  # Track last timestamp for message fetching

    def load_files(self):
        # Define project root and paths for necessary files
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load RDF graph
        graph_path = os.path.join(project_root, 'dataset', '14_graph.nt')
        graph_uri = f'file:///{pathname2url(graph_path)}'.replace('\\', '/')
        self.rdf_graph = rdflib.Graph().parse(graph_uri, format="turtle")

        # Load entity dictionary
        entity_dict_path = os.path.join(project_root, 'dataset', 'entity_corrected.json')
        with open(entity_dict_path, 'r', encoding='utf-8') as f:
            self.entity_dict = json.load(f)

        # Load property dictionary
        properties_path = os.path.join(project_root, 'dataset', 'properties.json')
        with open(properties_path, 'r', encoding='utf-8') as f:
            self.property_dict = json.load(f)

        # Load embeddings and IDs
        embeddings_dir = os.path.join(project_root, 'dataset', 'ddis-graph-embeddings')
        self.entity_embeddings = np.load(os.path.join(embeddings_dir, 'entity_embeds.npy'))
        self.relation_embeddings = np.load(os.path.join(embeddings_dir, 'relation_embeds.npy'))

        # Parse entity and relation IDs
        with open(os.path.join(embeddings_dir, 'entity_ids.del'), 'r', encoding='ISO-8859-1') as f:
            self.ent2id = {line.split('\t')[1].strip(): int(line.split('\t')[0]) for line in f if
                           len(line.split('\t')) == 2}

        with open(os.path.join(embeddings_dir, 'relation_ids.del'), 'r', encoding='ISO-8859-1') as f:
            self.rel2id = {line.split('\t')[1].strip(): int(line.split('\t')[0]) for line in f if
                           len(line.split('\t')) == 2}

    def login(self):
        login_data = {"username": "*****", "password": "*****"}
        response = requests.post(f"{self.base_url}/api/login", json=login_data, headers=self.headers)
        if response.status_code == 200:
            self.session_token = response.cookies.get("SESSIONID", None)
            if self.session_token:
                self.headers['Cookie'] = f"SESSIONID={self.session_token}"
                logging.info("Logged in successfully.")
        else:
            logging.error("Failed to log in.")

    def get_chatrooms(self):
        response = requests.get(f"{self.base_url}/api/rooms", headers=self.headers)
        if response.status_code == 200:
            return response.json().get('rooms', [])
        return []

    def fetch_messages(self, room_id, last_timestamp):
        """
        Fetch messages from the specified chatroom starting from the last timestamp.
        """
        response = requests.get(f"{self.base_url}/api/room/{room_id}/{last_timestamp}", headers=self.headers)
        if response.status_code == 200:
            messages = response.json().get('messages', [])
            if messages:
                logging.debug(f"Fetched {len(messages)} new messages from room {room_id}.")
            return messages
        else:
            logging.error(f"Failed to fetch messages from room {room_id}. Status Code: {response.status_code}")
        return []

    def fetch_chatroom_info(self, room_id):
        response = requests.get(f"{self.base_url}/api/rooms", headers=self.headers)
        if response.status_code == 200:
            chatrooms = response.json().get('rooms', [])
            for room in chatrooms:
                if room['uid'] == room_id:
                    return room
        logging.error(f"Failed to fetch information for chatroom {room_id}.")
        return None

    def handle_message(self, room_id, message_id, content):
        """
        Handles a message by sending it to SPARQLHandler for processing
        and then posting the formatted result.
        """
        try:
            content_lower = content.lower()
            logging.debug(f"Handling message: {content_lower}")

            # Skip message if already processed
            if message_id in self.processed_messages:
                return

            # Mark the message as processed to prevent reprocessing
            self.processed_messages.add(message_id)

            # Delegate message handling to SPARQLHandler and get the result
            result = self.sparql_handler.handle_message(content_lower)

            # Post the result if available
            if result:
                self.post_message(room_id, result)
            else:
                # Default response if no result from SPARQLHandler
                self.post_message(room_id, "Sorry, I couldn't find any relevant information.")

        except Exception as e:
            logging.error(f"Error processing message {message_id}: {e}")
            self.post_message(room_id, "An error occurred while processing your request. Please try again.")

    def run(self):
        logging.info("Running the bot...")
        self.login()
        if not self.session_token:
            logging.error("No session token found. Exiting.")
            return

        greeted_rooms = set()  # Track which rooms have been greeted

        while True:
            chatrooms = self.get_chatrooms()
            if not chatrooms:
                logging.info("No chatrooms available. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            for chatroom in chatrooms:
                room_id = chatroom['uid']
                self.bot_alias = chatroom['alias']
                last_timestamp = 0  # Reset timestamp for each new room

                if room_id not in greeted_rooms:
                    # Send greeting message only once per room
                    self.post_message(room_id, "Hi there!")
                    greeted_rooms.add(room_id)

                while True:
                    # Fetch the latest chatroom info
                    chatroom_info = self.fetch_chatroom_info(room_id)

                    if not chatroom_info:
                        logging.info(f"Chatroom {room_id} info not available, skipping.")
                        break

                    remaining_time = chatroom_info.get('remainingTime', 0)
                    mark_as_no_feedback = chatroom_info.get('markAsNoFeedback', False)

                    # Check room status each time to avoid re-entering closed rooms
                    if remaining_time <= 0 or mark_as_no_feedback:
                        self.processed_messages = set()
                        logging.info(f"Exiting room {room_id} as it is either closed or marked for no feedback.")
                        break

                    # Fetch messages starting from the last known timestamp
                    messages = self.fetch_messages(room_id, last_timestamp)
                    if messages:
                        for message in messages:
                            message_id = message['ordinal']
                            author = message['authorAlias']
                            content = message['message']

                            logging.info(f"Processing message {message_id} from {author}: {content}")

                            # Skip messages from the bot itself or already processed
                            if author == self.bot_alias or message_id in self.processed_messages:
                                logging.info(
                                    f"Skipping message {message_id} as it is from the bot or already processed.")
                                continue

                            # Process the message and mark it as processed
                            self.handle_message(room_id, message_id, content)
                            self.processed_messages.add(message_id)

                        # Update last timestamp to the latest message timestamp
                        last_timestamp = max(msg['timeStamp'] for msg in messages)

                    else:
                        logging.info(f"No new messages in room {room_id}.")

                    # Sleep briefly before re-checking messages and room status
                    logging.info(f"Sleeping for 5 seconds before rechecking room {room_id}.")
                    time.sleep(5)

            logging.info("No more active chatrooms. Retrying in 10 seconds...")
            time.sleep(10)

    def post_message(self, room_id, message):
        """
        Post a raw message to the specified chatroom without JSON wrapping.
        """
        # If the message is a list or other iterable, convert it to a comma-separated string
        if isinstance(message, list):
            message = ', '.join(message)
        elif not isinstance(message, str):
            message = str(message)

        logging.info(f"Posting message to room {room_id}: {message}")
        try:
            # Post message as raw text
            response = requests.post(
                f"{self.base_url}/api/room/{room_id}",
                data=message,  # Send raw message as plain text
                headers={'Content-Type': 'text/plain', **self.headers}  # Use text/plain content type
            )

            # Check response status and content
            if response.status_code == 200:
                logging.info(f"Message posted successfully to room {room_id}.")
            else:
                logging.error(
                    f"Failed to post message to room {room_id}. Status Code: {response.status_code}, Response: {response.content}")
        except Exception as e:
            logging.error(f"Exception occurred while posting message to room {room_id}: {e}")


if __name__ == "__main__":
    bot = Bot()
    bot.run()

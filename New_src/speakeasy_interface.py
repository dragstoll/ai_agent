import requests
import yaml


class SpeakeasyInterface:
    def __init__(self):
        # Load configuration (API credentials, endpoints) from config.yaml
        with open('../config/config.yaml', 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.base_url = self.config['speakeasy_base_url']
        self.bot_account = self.config['bot_account']
        self.bot_password = self.config['bot_password']
        self.user_account = self.config['user_account']
        self.user_password = self.config['user_password']
        self.token = None

    def login(self):
        """Login to the Speakeasy platform using bot credentials."""
        login_url = f"{self.base_url}/login"
        payload = {
            "username": self.bot_account,
            "password": self.bot_password
        }
        response = requests.post(login_url, json=payload)
        if response.status_code == 200:
            self.token = response.json()['token']
            print(f"Login successful for bot: {self.bot_account}")
        else:
            print(f"Failed to log in, Status code: {response.status_code}")
            print(f"Response content: {response.content}")

    def get_chatrooms(self):
        """Get list of available chatrooms for the bot."""
        if not self.token:
            print("Bot is not logged in!")
            return

        chatroom_url = f"{self.base_url}/chatrooms"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(chatroom_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to fetch chatrooms")

    def post_message(self, chatroom_id, message):
        """Post a message to a specific chatroom."""
        if not self.token:
            print("Bot is not logged in!")
            return

        post_url = f"{self.base_url}/chatrooms/{chatroom_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "message": message
        }
        response = requests.post(post_url, headers=headers, json=payload)
        if response.status_code == 200:
            print("Message posted successfully!")
        else:
            print("Failed to post message")

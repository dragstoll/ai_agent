import rdflib
import os
import logging
import json
import openai  # For calling the ChatGPT API
from urllib.request import pathname2url
import requests
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from statsmodels.stats.inter_rater import fleiss_kappa
import base64
import io
from collections import defaultdict


# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SPARQLHandler:
    def __init__(self, rdf_graph, entity_dict, property_dict, embedding_handler):
        """
        Initialize the SPARQLHandler with pre-loaded resources and EmbeddingHandler.
        """
        logging.debug("Initializing SPARQLHandler...")

        # Set the OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Use pre-loaded RDF graph and dictionaries
        self.graph = rdf_graph
        self.entity_dict = entity_dict
        self.property_dict = property_dict
        self.embedding_handler = embedding_handler  # Store the embedding handler
        self.entity_url = self.load_json('../dataset/entity_url.json')
        self.property_url = self.load_json('../dataset/properties-url.json')
        #self.property_url = self.load_json('../dataset/properties-url.json')

        # Load and normalize the crowdsource dataset
        self.crowd_data = pd.read_csv('../dataset/crowd_data/crowd_data.tsv', sep='\t')
        self.crowd_data['Input1ID'] = self.crowd_data['Input1ID'].str.lower().str.strip()
        self.crowd_data['Input2ID'] = self.crowd_data['Input2ID'].str.lower().str.strip()

        # Load entity embeddings from JSON file
        try:
            with open('../dataset/entity_embeddings.json', 'r') as f:
                self.entity_embeddings = json.load(f)
            logging.info("Loaded entity embeddings successfully.")
        except Exception as e:
            logging.error(f"Error loading entity embeddings: {e}")
            self.entity_embeddings = None

    def extract_entity_and_property(self, question):
        """
        Use ChatGPT to extract the entity and property (role or other properties) from the question.
        """
        logging.debug("Using ChatGPT to extract entity and property from question: %s", question)

        try:
            # Call the ChatGPT API to process the question
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in the movie industry. Your task is to identify relevant movies, people, and related entities, and extract their associated properties in a structured JSON format."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"From the following question, extract the relevant entities and their associated property in JSON format. "
                            f"Please respond in the format: '{{\"entity\": [<entity1>, <entity2>, ...], \"property\": \"<property>\"}}'. "
                            f"Question: '{question}'"
                        )
                    }
                ]
            )

            # Extract the response content
            extracted_text = response['choices'][0]['message']['content']
            logging.debug(f"ChatGPT response: {extracted_text}")

            # Parse the JSON response for entity and property
            try:
                extracted_json = json.loads(extracted_text)
                entity = extracted_json.get("entity")
                property_ = extracted_json.get("property")

                # Log the extracted values
                logging.debug(f"Extracted Entity: {entity}, Property: {property_}")

                # Check for closest match using ChatGPT
                closest_match_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Here are the properties I have: {json.dumps(self.property_dict)}. Match this property to the closest one from the list and respond in JSON format as: '{{\"property\": \"<Closest Property>\", \"url\": \"<URL>\"}}'."
                        },
                        {
                            "role": "user",
                            "content": f"Find the closest match for the property: '{property_}'. Please respond in JSON format."
                        }
                    ]
                )

                # Extract the closest match response
                closest_match_text = closest_match_response['choices'][0]['message']['content']
                logging.debug(f"Closest match response: {closest_match_text}")

                # Parse the JSON response for the closest match
                closest_match_json = json.loads(closest_match_text)
                closest_property = closest_match_json.get("property")
                closest_url = closest_match_json.get("url")

                if closest_url in self.property_url:
                    property_ = closest_url  # Update property_ to the closest URL if found
                    logging.debug(f"Using closest URL for property: {property_}")
                elif closest_property.lower() in self.property_dict:
                    property_ = closest_property.lower()  # Keep the property label if found
                    logging.debug(f"Using closest property label for property: {property_}")

                return entity, property_

            except json.JSONDecodeError:
                logging.warning("Failed to decode JSON from ChatGPT response.")
                return None, None

        except Exception as e:
            logging.error(f"Error calling ChatGPT API: {e}")
            return None, None

    def get_entity_url(self, entity_label):
        """
        Look up the URL for the entity in the entity dictionary.
        If the entity is not found, use embeddings to find the closest match.
        Returns a tuple: (result, is_exact_match).
        """
        logging.debug(f"Looking up URL for Entity: {entity_label}")

        # Normalize entity label by stripping spaces and converting to lowercase
        entity_label = entity_label.strip().lower()
        entity_url = self.entity_dict.get(entity_label)  # Lookup the modified label in the dictionary
        logging.debug(f"Found Entity URL: {entity_url}")

        if entity_url:
            return entity_url, True  # Exact match found
        else:
            logging.warning(f"Entity '{entity_label}' not found in entity dictionary.")
            # Fallback to embeddings for closest match
            closest_label = self.find_closest_entity_url(entity_label)
            logging.debug(f"Closest match found for '{entity_label}': {closest_label}")
            return closest_label, False  # Closest match found

    def get_embedding(self, text):
        """
        Generate an embedding for the given text.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response['data'][0]['embedding'])

    def find_closest_entity_url(self, entity_label, threshold=0.85):
        """
        Find the closest matching entity using embeddings if the entity is not found in the dictionary.
        """
        if self.entity_embeddings is None:
            logging.error("Entity embeddings not loaded.")
            return None

        logging.info(f"Attempting to find closest match for entity: {entity_label}")

        # Generate embedding for the entity label
        query_embedding = self.get_embedding(entity_label).reshape(1, -1)

        max_similarity = -1
        closest_entity = None

        # Iterate over entities in the loaded embeddings dictionary
        for entity, embedding_list in self.entity_embeddings.items():
            entity_embedding = np.array(embedding_list).reshape(1, -1)  # Convert list to numpy array
            similarity = cosine_similarity(query_embedding, entity_embedding)[0][0]

            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                closest_entity = entity

        if closest_entity:
            logging.info(f"Closest entity found: {closest_entity} with similarity {max_similarity:.4f}")
            return self.get_entity_url(closest_entity.lower())
        else:
            logging.warning(f"No high-confidence match found for '{entity_label}' using embeddings.")
            return None

    def get_property_url(self, property_label):
        """
        Look up the URL for the property in the property dictionary.
        """
        logging.debug(f"Looking up URL for Property: {property_label}")

        try:
            property_url = self.property_dict[property_label]
            logging.debug(f"Found Property URL: {property_url}")
            return property_url
        except KeyError:
            logging.warning(f"Property '{property_label}' not found in property dictionary.")
            return None

    def execute_sparql(self, entity_url, property_url):
        """
        Execute SPARQL query on the knowledge graph using the entity and property URLs.
        """
        # Confirm the URLs are correctly formatted before building the query
        if not entity_url or not property_url:
            logging.error(f"Invalid entity or property URL. Entity: {entity_url}, Property: {property_url}")
            return "Error: Invalid entity or property URL."

        # Construct the SPARQL query
        query = f"""
        SELECT ?value WHERE {{
            <{entity_url}> <{property_url}> ?value .
        }} LIMIT 10
        """

        logging.debug(f"Executing SPARQL query: {query}")
        try:
            result = self.graph.query(query)
            logging.debug("SPARQL query executed successfully.")
            return result
        except Exception as e:
            logging.error("Error executing SPARQL query: %s", e)
            return f"Error executing SPARQL query: {str(e)}"

    def preprocess_text(self, text):
        # Remove punctuation, including double and single quotes, and make lowercase
        text = text.lower().replace('"', '').replace("'", "")
        return text

    def match_keywords(self, question, keywords):
        # Create a regex pattern to match each keyword as a standalone word or phrase
        pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
        match = bool(re.search(pattern, question))
        logging.debug(f"Keyword match for pattern '{pattern}': {match}")

        return bool(re.search(pattern, question))

    def handle_message(self, question):
        """
        Handle the user's question by extracting entity and property, looking them up, and querying the RDF graph.
        """
        question = self.preprocess_text(question)
        logging.debug("Handling message: %s", question)

        # Step 1: Extract entity and property using ChatGPT
        entity_label, property_label = self.extract_entity_and_property(question)

        # Normalize entity_label to handle both single and multiple entities
        entity_labels = [e.lower() for e in entity_label] if isinstance(entity_label, list) else [entity_label.lower()]

        # Ensure property_label is a string, extracting the key if it's a dictionary
        if isinstance(property_label, dict):
            property_label = next(iter(property_label.keys()))

        # Normalize property_label to lowercase if it exists and is not a URL
        if not property_label.startswith("http"):
            property_label = property_label.lower() if property_label else None

        # Retrieve all keyword lists
        keyword_lists = self.get_keyword_lists()
        results = []

        # Process each entity individually
        for entity in entity_labels:
            entity_result, is_exact_match = self.get_entity_url(entity) if entity else (None, False)

            # If we have a URL (exact match), use it; otherwise, treat it as a label
            if is_exact_match:
                entity_urls = [entity_result]  # Exact match found, using the URL directly
                logging.debug(f"Exact match found for '{entity}': {entity_result}")
            else:
                logging.debug(f"Similar match found for '{entity}': {entity_result[0][0]}")

                # Check if entity_result is a list and handle accordingly
                if isinstance(entity_result[0], list) and entity_result[0]:
                    entity = self.get_entity_label(entity_result[0][0])
                    entity_urls = entity_result[0]
                elif entity_result[0]:
                    entity = self.get_entity_label(entity_result[0][0])
                    entity_urls = [entity_result[0]]
                else:
                    entity_urls = []

                logging.debug(f"Using closest match for '{entity}' with URLs: {entity_urls}")

            main_answer = None

            # Check for image request
            if self.match_keywords(question, keyword_lists["image_keywords"]) or property_label == "image":
                logging.debug("Matched image keywords or recognized 'image' property.")
                images = []
                for entity_url in entity_urls:
                    try:
                        image = self.get_image(entity_url)
                        if image:
                            images.append(image)
                    except Exception as e:
                        logging.error(f"Error fetching image for URL {entity_url}: {e}")
                main_answer = "\n".join(images) if images else f"No images found for '{entity}'."

            # Check for recommendation request
            elif self.match_keywords(question, keyword_lists["recommendation_keywords"]):
                logging.debug("Matched recommendation keywords.")
                recommendations = self.get_recommendation(entity)
                main_answer = recommendations if recommendations else f"No recommendations found for '{entity}'."

            # Check for similarity search
            elif self.match_keywords(question, keyword_lists["similar_keywords"]):
                logging.debug("Matched similarity search keywords.")
                similar_entities = []
                for entity_url in entity_urls:
                    try:
                        similar_entity_label = self.embedding_handler.find_similar_entities_by_url(entity_url, top_k=1)
                        if similar_entity_label and similar_entity_label != entity:
                            similar_entities.append(similar_entity_label)
                    except Exception as e:
                        logging.error(f"Error finding similar entities for URL {entity_url}: {e}")
                main_answer = ", ".join(
                    similar_entities) if similar_entities else f"No similar entities found for '{entity}'."

            # Check for plot/story retrieval
            elif self.match_keywords(question, keyword_lists["plot_keywords"]):
                logging.debug("Matched plot/story keywords.")
                plots = []
                for entity_url in entity_urls:
                    try:
                        plot = self.get_plot(entity_url)
                        if plot:
                            plots.append(plot)
                    except Exception as e:
                        logging.error(f"Error fetching plot for URL {entity_url}: {e}")
                main_answer = "\n".join(plots) if plots else f"No plots found for '{entity}'."

            # Check for user rating retrieval
            elif self.match_keywords(question, keyword_lists["user_rating_keywords"]):
                logging.debug("Matched user rating keywords.")
                ratings = []
                for entity_url in entity_urls:
                    try:
                        rating = self.get_user_rating(entity_url)
                        if rating:
                            ratings.append(rating)
                    except Exception as e:
                        logging.error(f"Error fetching user rating for URL {entity_url}: {e}")
                main_answer = "\n".join(ratings) if ratings else f"No ratings found for '{entity}'."

            # Handle property-specific query if a property is specified
            elif property_label:
                if property_label.lower() in self.property_dict:
                    property_url = self.property_dict[property_label]
                elif property_label in self.property_url:
                    property_url = property_label
                else:
                    logging.warning(f"Property '{property_label}' not found in property_dict or property_url.")
                    main_answer = "The property does not exist in our records."

                if main_answer is None:
                    results_list = []
                    for entity_url in entity_urls:
                        # Check if entity_url is a list, and handle each URL within it if necessary
                        urls_to_query = entity_url if isinstance(entity_url, list) else [entity_url]

                        for single_url in urls_to_query:
                            if not isinstance(single_url, str):
                                logging.error(f"Expected a single URL in string format but received: {single_url}")
                                continue  # Skip non-string entries

                            try:
                                # Execute SPARQL for a single URL
                                result = self.execute_sparql(single_url, property_url)
                                if result:
                                    formatted_result = self.format_sparql_result(result)
                                    results_list.append(formatted_result)
                            except Exception as e:
                                logging.error(
                                    f"Error executing SPARQL query for URL {single_url} and property {property_url}: {e}")

                    if results_list:
                        readable_results = []
                        for result in results_list:
                            urls = result.split() if isinstance(result, str) and result.count("http") > 1 else [result]
                            readable_labels = [
                                self.get_property_label(url) or self.get_entity_label(url) or url
                                if url.startswith("http") else url
                                for url in urls
                            ]
                            readable_results.append("\n".join(readable_labels))
                        main_answer = "\n".join(readable_results)
                    else:
                        logging.debug("No results found for entity URLs, attempting similar relations.")
                        for entity_url in entity_urls:
                            urls_to_query = entity_url if isinstance(entity_url, list) else [entity_url]

                            for single_url in urls_to_query:
                                try:
                                    similar_relations = self.embedding_handler.find_similar_relations_by_url(
                                        property_url, single_url)
                                    if similar_relations:
                                        main_answer = similar_relations
                                        break
                                except Exception as e:
                                    logging.error(
                                        f"Error finding similar relations for URL {single_url} and property {property_url}: {e}")
                            if main_answer:
                                break
                        else:
                            main_answer = f"No similar relations found for '{entity}' with the specified property."


            # Handle case where property is not found
            else:
                logging.warning(f"Property '{property_label}' not found in property_dict or property_url.")
                main_answer = "The property does not exist in our records."

            # Final crowdsourcing check to add to main_answer if applicable
            if main_answer:
                predicate = property_label if property_label else ""
                crowdsourcing_text = self.get_crowdsource(entity, predicate)
                if crowdsourcing_text:
                    main_answer = f" {crowdsourcing_text}"

                results.append(main_answer)

        # Handle cases where no result was found
        if not results:
            return "Unable to process the question. No results were found."
        elif len(results) == 1:
            return results[0]
        else:
            return "\n\n".join(results)

    def format_sparql_result(self, result):
        """
        Format the SPARQL query result for output.
        """
        logging.debug("Formatting SPARQL result: %s", result)
        if isinstance(result, str):
            return result  # Handle error message from SPARQL execution

        formatted_result = []
        for row in result:
            row_result = [str(item) for item in row]
            formatted_result.append(", ".join(row_result))

        if not formatted_result:
            logging.debug("No results found.")
            return "No results found."

        return '\n'.join(formatted_result)


    def get_entity_label(self, entity_url):
        """
        Look up the label for the entity using the local knowledge graph.
        """
        """
                Look up the URL for the entity in the entity dictionary.
                """
        logging.debug(f"Looking up label for Entity: {entity_url}")

        try:
            entity_label = self.entity_url[entity_url]
            logging.debug(f"Found Entity label: {entity_label}")
            return entity_label
        except KeyError:
            logging.warning(f"Entity '{entity_url}' not found in entity dictionary.")
            return None

    def get_property_label(self, property_url):
        """
        Look up the label for the property using the local knowledge graph.
        """
        logging.debug(f"Looking up label for Property: {property_url}")

        try:
            property_label = self.property_url[property_url]
            logging.debug(f"Found Property label: {property_label}")
            return property_label
        except KeyError:
            logging.warning(f"Property '{property_url}' not found in property dictionary.")
            return None

    def load_json(self, file_path):
        """
        Load a JSON file with UTF-8 encoding.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            logging.debug("JSON file loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            return {}

    def get_plot(self, entity_url):
        """
        Retrieve the plot/story for the given entity URL from the plots CSV.
        """
        logging.debug(f"Fetching plot for entity URL: {entity_url}")

        try:
            # Load plot data from the CSV file
            plots_df = pd.read_csv("../dataset/plots.csv")

            # Filter for the specific entity URL
            filtered_plot = plots_df[plots_df['qid'] == entity_url]

            if not filtered_plot.empty:
                # Return the first matching plot text
                plot_text = filtered_plot.iloc[0]['plot']  # Assuming 'plot' is the correct column name
                return f"Plot for {self.get_entity_label(entity_url)}: {plot_text}"

            return f"No plot found for {self.get_entity_label(entity_url)}."

        except Exception as e:
            logging.error(f"Error reading plot data: {e}")
            return "Error retrieving plot information."

    import pandas as pd

    def get_user_rating(self, entity_url):
        """
        Retrieve user ratings for the given entity URL from the user ratings CSV.
        """
        logging.debug(f"Fetching user ratings for entity URL: {entity_url}")

        try:
            # Load user ratings from the CSV file
            ratings_df = pd.read_csv("../dataset/user-comments.csv")

            # Filter ratings for the specific entity URL
            filtered_ratings = ratings_df[ratings_df['qid'] == entity_url]

            if not filtered_ratings.empty:
                # Calculate total rating or average rating
                average_rating = filtered_ratings['rating'].mean()
                return f"Average User Rating for {self.get_entity_label(entity_url)}: {average_rating:.2f}"

            return f"No ratings found for {self.get_entity_label(entity_url)}."

        except Exception as e:
            logging.error(f"Error reading user ratings: {e}")
            return "Error retrieving user ratings."

    def get_recommendation(self, entity_name):
        # Load the JSON file with movie similarities
        logging.debug("Loading similar data JSON file for recommendations.")
        with open("../dataset/movies-similar.json", "r", encoding="utf-8") as f:
            similar_data = json.load(f)

        logging.debug(f"Inside recommendation function for entity: {entity_name}")

        # Check if the entity name is in the data
        lower_entity_name = entity_name.lower()
        if lower_entity_name in similar_data:
            recommendations = similar_data[lower_entity_name]
            logging.debug(f"Recommendations found for {entity_name}: {recommendations}")

            # Format recommendations to show only titles
            recommendation_text = f"Top recommendations for {entity_name}:\n"
            for rec in recommendations:
                recommendation_text += f"- {rec['title']}\n"
                logging.debug(f"Adding recommendation title: {rec['title']}")

            return recommendation_text

        logging.warning(f"No recommendations found for '{entity_name}'.")
        return f"No recommendations found for '{entity_name}'."

    def get_image(self, entity_name, min_width=0, min_height=0):
        # Load the JSON file with entity images
        with open("../dataset/images_entities.json", "r", encoding="utf-8") as f:
            image_data = json.load(f)

        # Check if entity_name is a list and extract the first URL if necessary
        if isinstance(entity_name, list):
            entity_name = entity_name[0]

            # If entity_name contains "http", convert it to a label
        if "http" in entity_name:
            entity_name = self.get_entity_label(entity_name)
        # Normalize entity_name to lowercase for case-insensitive matching
        entity_name_lower = entity_name.lower()
        image_data_lower = {key.lower(): value for key, value in image_data.items()}

        logging.debug(f"Looking for images for entity: {entity_name_lower}")

        # Define the priority order for image types
        type_priority = ["event", "user_avatar", "publicity", "poster", "behind_the_scenes", "still_frame"]

        # Check if the entity exists in the image data (case-insensitive)
        if entity_name_lower in image_data_lower:
            images = image_data_lower[entity_name_lower]
            logging.debug(f"Images found for {entity_name_lower}: {len(images)} images available" if isinstance(images,
                                                                                                                list) else "One image available")

            # Ensure that images is a list
            if not isinstance(images, list):
                logging.warning(
                    f"Expected list of images for {entity_name_lower}, but got {type(images)}. Converting to list.")
                images = [images]  # Convert single dict entry to a list if necessary

            # Sort images by priority and check dimensions
            for image_type in type_priority:
                for image in images:
                    # Handle potential issue if image is not a dictionary
                    if isinstance(image, dict) and image.get("type") == image_type:
                        logging.debug(
                            f"Checking image of type {image_type} for {entity_name_lower}, Dimensions: {image.get('w')}x{image.get('h')}")
                        if image.get("w", 0) >= min_width and image.get("h", 0) >= min_height:
                            img_path_no_ext = image["img"].rsplit('.', 1)[0]  # Remove the file extension
                            logging.info(f"Image path (without extension) found for {entity_name}: {img_path_no_ext}")

                            # Return the path in chat-compatible format without modifications
                            return f'image:{img_path_no_ext}'
        else:
            logging.warning(f"No images found for entity '{entity_name}' in the image data.")

        return f"No suitable image found for '{entity_name}' with the specified dimensions."

    def identify_malicious_workers(self, data):
        """
        Identify malicious workers based on multiple criteria
        """
        worker_stats = defaultdict(lambda: {
            'total_tasks': 0,
            'fast_tasks': 0,
            'disagreements': 0,
            'approval_rate': 0
        })

        hit_majorities = data.groupby('hitid')['answerlabel'].agg(lambda x: x.mode()[0])

        for _, row in data.iterrows():
            worker_id = row['workerid']
            worker_stats[worker_id]['total_tasks'] += 1
            worker_stats[worker_id]['approval_rate'] = float(row['lifetimeapprovalrate'].strip('%'))

            if row['worktimeinseconds'] < 10:
                worker_stats[worker_id]['fast_tasks'] += 1

            if row['answerlabel'] != hit_majorities[row['hitid']]:
                worker_stats[worker_id]['disagreements'] += 1

        malicious_workers = set()
        for worker_id, stats in worker_stats.items():
            if stats['total_tasks'] > 0:
                disagreement_rate = stats['disagreements'] / stats['total_tasks']
                fast_task_rate = stats['fast_tasks'] / stats['total_tasks']

                if (stats['approval_rate'] < 75 or
                        fast_task_rate > 0.5 or
                        disagreement_rate > 0.7):
                    malicious_workers.add(worker_id)

        return malicious_workers

    def calculate_fleiss_kappa(self, data):
        """Calculate Fleiss' Kappa for a batch of ratings"""
        hits = data['hitid'].unique()
        n_items = len(hits)
        n_raters = data.groupby('hitid').size().iloc[0]

        if n_items < 2 or n_raters < 2:
            return 0.0

        ratings = np.zeros((n_items, 2))

        for idx, hit_id in enumerate(hits):
            hit_data = data[data['hitid'] == hit_id]
            ratings[idx, 0] = (hit_data['answerlabel'] == 'CORRECT').sum()
            ratings[idx, 1] = (hit_data['answerlabel'] == 'INCORRECT').sum()

        n = ratings.sum(axis=1)
        P_i = (np.sum(ratings * (ratings - 1), axis=1)) / (n * (n - 1))
        P_bar = np.mean(P_i)

        P_j = ratings.sum(axis=0) / (n_items * n_raters)
        P_e = np.sum(P_j * P_j)

        try:
            kappa = (P_bar - P_e) / (1 - P_e)
            return max(min(float(kappa), 1.0), -1.0)
        except:
            return 0.0

    def get_crowdsource(self, subject, predicate):
        """
        Retrieve crowdsourced information for a given subject and predicate, returning it in brackets if available.
        """
        logging.debug("Starting get_crowdsource function")

        # Retrieve full URLs using get_entity_url() and get_property_url()
        if subject.startswith("http"):
            subject_url = subject
        else:
            subject_url, is_exact_match = self.get_entity_url(subject)

        if isinstance(subject_url, list) and subject_url:
            subject_url = subject_url[0]
        elif isinstance(subject_url, list) or subject_url is None:
            logging.warning("Subject URL list is empty or None.")
            return ""

        if predicate.startswith("http"):
            predicate_url = predicate
        else:
            predicate_url = self.get_property_url(predicate)

        if isinstance(predicate_url, list) and predicate_url:
            predicate_url = predicate_url[0]
        elif isinstance(predicate_url, list) or predicate_url is None:
            logging.warning("Predicate URL list is empty or None.")
            return ""

        logging.debug(f"Retrieved URLs - subject_url: {subject_url}, predicate_url: {predicate_url}")

        # Extract the last part of each URL
        try:
            subject_id = subject_url.split('/')[-1].lower()
            predicate_id = predicate_url.split('/')[-1].lower()
        except AttributeError as e:
            logging.error(f"Error splitting URL: {e}. subject_url: {subject_url}, predicate_url: {predicate_url}")
            return ""

        # Ensure proper renaming of columns in self.crowd_data
        crowd_data = self.crowd_data.rename(columns={
            'HITId': 'hitid',
            'HITTypeId': 'hittypeid',
            'Input1ID': 'input1id',
            'Input2ID': 'input2id',
            'Input3ID': 'input3id',
            'AnswerLabel': 'answerlabel',
            'LifetimeApprovalRate': 'lifetimeapprovalrate',
            'WorkTimeInSeconds': 'worktimeinseconds',
            'WorkerId': 'workerid'
        })

        # Filter out malicious workers first
        malicious_workers = self.identify_malicious_workers(crowd_data)
        valid_data = crowd_data[~crowd_data['workerid'].isin(malicious_workers)].copy()

        # Normalize and remove prefixes in dataset columns for consistency
        valid_data.loc[:, 'input1id'] = valid_data['input1id'].str.lower().str.strip().str.replace(r'^wd:', '',
                                                                                                   regex=True)
        valid_data.loc[:, 'input2id'] = valid_data['input2id'].str.lower().str.strip().str.replace(r'^wdt:', '',
                                                                                                   regex=True)

        # Pre-calculate kappa scores for all batches using valid_data
        batch_kappas = {}
        for batch_id, batch_data in valid_data.groupby('hittypeid'):
            if len(batch_data['hitid'].unique()) >= 2:
                kappa = self.calculate_fleiss_kappa(batch_data)
                batch_kappas[batch_id] = kappa
            else:
                batch_kappas[batch_id] = 0.0

        # Filter rows for the specific subject-predicate pair from valid_data
        relevant_rows = valid_data[
            (valid_data['input1id'] == subject_id) &
            (valid_data['input2id'] == predicate_id)
            ]

        if relevant_rows.empty or 'input3id' not in relevant_rows.columns or relevant_rows['input3id'].isna().all():
            logging.info("No valid crowdsourced data available for the given subject-predicate pair.")
            return ""

        # Group by hitid to get majority answers and distributions for each hit
        hit_data = []
        for hit_id, group in relevant_rows.groupby('hitid'):
            batch_id = group['hittypeid'].iloc[0]
            distribution = {
                'CORRECT': (group['answerlabel'] == 'CORRECT').sum(),
                'INCORRECT': (group['answerlabel'] == 'INCORRECT').sum(),
                'total_votes': len(group)
            }

            hit_data.append({
                'hitid': hit_id,
                'batchid': batch_id,
                'majorityanswer': group['answerlabel'].mode().iloc[0],
                'input1id': group['input1id'].iloc[0],
                'input2id': group['input2id'].iloc[0],
                'input3id': group['input3id'].iloc[0],
                'batch_agreement': batch_kappas[batch_id],
                'distribution': distribution
            })

        if not hit_data:
            return ""

        # Use the first hit's data (they should all be for the same subject-predicate pair)
        hit_info = hit_data[0]
        object_value = hit_info['input3id'].strip()
        agreement_score = hit_info['batch_agreement']
        distribution = hit_info['distribution']

        # Construct URLs to retrieve human-readable labels
        subject_label = self.get_entity_label(subject_url) or self.get_formatted_entity_label(subject_id)
        predicate_url = f'http://www.wikidata.org/prop/direct/{predicate_id.upper()}'
        predicate_label = self.format_predicate(self.get_property_label(predicate_url) or predicate_id)

        # Ensure object_value is formatted correctly
        object_label = self.get_formatted_entity_label(object_value)

        # Format the response using the format_response method
        response_text = self.format_response(subject_label, predicate_label, object_label)

        # Construct crowd_info
        crowd_info = (
            f"[Crowd, inter-rater agreement {agreement_score:.3f}, "
            f"The answer distribution for this specific task was {distribution['CORRECT']} support votes, "
            f"{distribution['INCORRECT']} reject votes]"
        )

        # Combine response_text and crowd_info
        return f"{response_text}. {crowd_info}"

    def format_response(self, subject_label, predicate_label, object_label):
        """Format response string based on predicate type"""
        predicate = predicate_label.lower()
        if predicate in ['box office', 'publication date']:
            return f"{subject_label} {predicate} {object_label}"
        if predicate.startswith('is indirectly'):
            return f"{subject_label} {predicate} {object_label}"
        return f"{subject_label} {predicate} {object_label}"

    def format_predicate(self, predicate_label):
        """Make predicates human readable"""
        if not predicate_label:
            return ''
        predicate = predicate_label.lower()
        if predicate == 'ddis:indirectsubclassof':
            return 'is indirectly a type of'
        if predicate.startswith('p'):
            return predicate
        return predicate

    def get_formatted_entity_label(self, entity_id):
        """Get human readable entity label from ID"""
        if not entity_id:
            return ''

        # Clean up the entity ID
        entity_id = str(entity_id).strip().upper()
        if entity_id.startswith('WD:'):
            entity_id = entity_id[3:]
        if entity_id.startswith('Q'):
            entity_url = f'http://www.wikidata.org/entity/{entity_id}'
            label = self.get_entity_label(entity_url)
            if label:
                return label
            return f'Entity {entity_id}'
        return entity_id

    def get_keyword_lists(self):
        return {
            "similar_keywords": [
                'similar', 'like', 'resemble', 'akin to', 'comparable to', 'close to', 'related to',
                'kind of like', 'sort of like', 'matches', 'almost like', 'as if it were',
                'in the same vein as', 'the same as', 'on par with', 'resembles', 'equivalent to',
                'similar to', 'inspired by', 'close in style', 'another one like', 'same kind of',
                'almost identical to', 'has a similar feel', 'the same vibe as', 'along the lines of',
                'something like', 'similar genre', 'similar storyline', 'reminds me of', 'feels like',
                'is close to', 'can be compared to', 'brings to mind', 'another version of',
                'matches the style of', 'in the spirit of', 'from the same genre', 'reminiscent of',
                'another option like', 'fits the profile of', 'has the same essence as', 'akin in theme to',
                'closely related', 'could be related', 'goes hand in hand with'
            ],
            "plot_keywords": [
                'story', 'plot', 'narrative', 'storyline', 'summary', 'what happens', 'what’s it about',
                'synopsis', 'tell me the story', 'describe the plot', 'give the plot', 'outline',
                'the tale of', 'background', 'account of', 'explanation of', 'main events',
                'sequence of events', 'core story', 'central story', 'plotline', 'what it’s about',
                'plot summary', 'story summary', 'core narrative', 'central narrative', 'explain the plot',
                'describe what happens', 'give me the storyline', 'gist of the story', 'central events',
                'main plot points', 'key events', 'what unfolds', 'theme of the story', 'overview of the plot',
                'basic plot', 'tell the story', 'how the story goes', 'central theme', 'principal storyline',
                'main idea', 'tell what happens', 'break down the plot', 'gist of the narrative',
                'main conflict', 'crux of the story', 'essence of the story', 'chronology of events',
                'what’s the main story', 'what’s the premise', 'describe the sequence', 'basic storyline'
            ],
            "user_rating_keywords": [
                'user rating', 'user ratings', 'ratings', 'reviews', 'user score', 'how people rated',
                'viewer opinions', 'audience rating', 'feedback', 'approval score', 'popularity score',
                'customer rating', 'critics opinion', 'user feedback', 'crowd opinion', 'how viewers felt',
                'public rating', 'community score', 'score by users', 'average score', 'general rating',
                'how it was rated', 'what people think', 'how viewers rated it', 'rating on it',
                'viewer score', 'what’s the score', 'overall rating', 'overall score', 'general feedback',
                'general opinion', 'average rating', 'community review', 'common rating', 'public opinion',
                'what the audience thinks', 'user average rating', 'how people scored', 'audience review',
                'viewer response', 'user popularity', 'what viewers say', 'audience feedback', 'community opinion',
                'user perception', 'general score', 'overall impression', 'critic score', 'what people say'
            ],
            "recommendation_keywords": [
                'recommend', 'suggest', 'advise', 'you think', 'would enjoy', 'recommendations for',
                'similar to my taste', 'I might like', 'ideas on what to watch', 'good options',
                'do you have any recommendations', 'any suggestions', 'other options', 'what else is good',
                'best picks', 'can you recommend', 'please suggest', 'suggestions similar to', 'top choices',
                'what do you recommend', 'anything similar', 'do you think I’d like', 'any good ones',
                'what’s a good option', 'what would you suggest', 'other recommendations', 'can you advise',
                'recommend something', 'good alternatives', 'find something like', 'what’s similar',
                'any similar ones', 'anything else', 'suggest movies like', 'give me options',
                'movies I’d enjoy', 'things you’d suggest', 'list some recommendations', 'you suggest',
                'can you give a list of', 'offer some suggestions', 'which you think is good', 'list of options',
                'might be interested in', 'good to watch', 'your suggestion', 'picks for me', 'suggestions on'
            ],
            "image_keywords": [
                'image of', 'photo of', 'picture of', 'show me', 'look like', 'appearance', 'portrait',
                'visual of', 'snapshot of', 'depiction of', 'what does he look like', 'what does she look like',
                'can I see', 'give me an image', 'display picture', 'visual representation', 'show picture',
                'any images', 'looks like', 'see a photo', 'show me what they look like', 'see their face',
                'how they look', 'any visuals', 'do you have a picture', 'view an image of', 'display an image',
                'see what they look like', 'show face of', 'appearance of', 'what’s their image', 'picture of them',
                'can you display a photo', 'show a snapshot', 'give me a picture', 'see what they look like',
                'what’s the face of', 'portrait view', 'facial appearance', 'body image', 'how they appear',
                'show me their appearance', 'appearance snapshot', 'look of', 'picture view', 'any portraits',
                'face image', 'facial picture', 'get a visual', 'get an image', 'photo display', 'portrait shot',
                'look like'
            ],
            "crowdsourcing_keywords": [
                'crowd', 'consensus', 'agreement', 'crowdsourced', 'popular answer', 'what most people said',
                'community opinion', 'what others think', 'the crowd’s take', 'common answer',
                'public consensus', 'majority opinion', 'common belief', 'group view', 'crowd consensus',
                'collective opinion', 'general agreement', 'what others believe', 'general opinion',
                'community perspective', 'public sentiment', 'group consensus', 'box office', 'earnings',
                'revenue', 'release date', 'premiere date', 'opening weekend', 'ratings', 'viewer ratings',
                'how it was rated', 'critic ratings', 'audience score', 'who directed', 'cast of',
                'actors in', 'main actor', 'main actress', 'director', 'producers', 'executive producers',
                'screenwriters', 'story by', 'writers of', 'plot summary', 'movie summary', 'budget of',
                'production costs', 'funding', 'how successful was', 'how popular is', 'do people like',
                'what do people think of', 'public reaction to', 'crowd rating', 'was it successful',
                'is it well-rated', 'does it have good reviews', 'was it well-received', 'how was it received',
                'general rating', 'common reviews', 'overall opinion on', 'overall thoughts on',
                'how much did it make', 'how much it grossed', 'total earnings', 'final box office',
                'how much it earned', 'what’s the revenue of', 'was it a hit', 'box office earnings',
                'money made by', 'how profitable was', 'when was it released', 'what year was it released',
                'who stars in', 'who was cast in', 'lead role', 'supporting actors', 'who is in the cast',
                'is it popular', 'do fans like', 'how well did it do', 'was it a flop', 'was it a blockbuster',
                'did it do well', 'success rate'
            ]
        }

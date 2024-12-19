import os
import numpy as np
import logging
import json


class EmbeddingHandler:
    def __init__(self, entity_embeddings, relation_embeddings, ent2id, rel2id, sparql_handler):
        """
        Initialize the EmbeddingHandler with pre-loaded embeddings, IDs, and SPARQL handler.
        Load the entity dictionary for faster label lookups.
        """
        logging.debug("Initializing EmbeddingHandler...")

        # Set pre-loaded embeddings and ID mappings
        self.entity_emb = entity_embeddings.astype('float32')
        self.relation_emb = relation_embeddings.astype('float32')
        self.ent2id = ent2id
        self.rel2id = rel2id

        # Reverse the ent2id to create id2ent and rel2id to create id2rel
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        # Store the SPARQLHandler for querying similar entities
        self.sparql_handler = sparql_handler

        # Load the entity dictionary
        with open('../dataset/entity.json', 'r', encoding='utf-8') as f:
            self.entity_dict = json.load(f)

        logging.debug("Loaded entity dictionary from 'entity.json'")

        # Normalize the embeddings to improve cosine similarity calculations
        self.normalized_entity_emb = self.normalize_embeddings(self.entity_emb)
        self.normalized_relation_emb = self.normalize_embeddings(self.relation_emb)

    def normalize_embeddings(self, embeddings):
        """Normalize the embeddings for efficient cosine similarity calculations."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)

    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two normalized vectors."""
        return np.dot(vec1, vec2)

    def find_top_k_similar(self, embedding, embeddings, top_k):
        """Find the top_k most similar embeddings using cosine similarity."""
        similarities = np.dot(embeddings, embedding.T).flatten()
        sorted_indices = np.argsort(similarities)[-top_k - 1:][::-1]
        return sorted_indices[1:top_k + 1]

    def find_similar_entities_by_url(self, entity_url, top_k=1):
        logging.debug(f"Finding similar entities for URL: {entity_url}")
        entity_id = self.ent2id.get(entity_url)
        if entity_id is None:
            logging.error(f"Entity URL '{entity_url}' not found in the embeddings.")
            return f"Entity URL '{entity_url}' not found in the embeddings."

        entity_embedding = self.normalized_entity_emb[entity_id]
        similar_indices = self.find_top_k_similar(entity_embedding, self.normalized_entity_emb, top_k)

        similar_entities = []
        for idx in similar_indices:
            similar_entity_url = self.id2ent.get(idx)
            if similar_entity_url == entity_url or not similar_entity_url:
                continue

            entity_label = self.sparql_handler.get_entity_label(similar_entity_url)
            if entity_label:
                logging.debug(f"Found Entity label for {similar_entity_url}: {entity_label}")
                similar_entities.append(entity_label)

        return "\n".join(similar_entities) if similar_entities else "No similar entities found."

    def find_similar_relations_by_url(self, relation_url, entity_url, top_k=5):
        logging.debug(f"Finding similar relations for URL: {relation_url}")
        relation_id = self.rel2id.get(relation_url)
        if relation_id is None:
            logging.error(f"Relation URL '{relation_url}' not found in the embeddings.")
            return f"Relation URL '{relation_url}' not found in the embeddings."

        relation_embedding = self.normalized_relation_emb[relation_id]
        similar_indices = self.find_top_k_similar(relation_embedding, self.normalized_relation_emb, top_k)

        for idx in similar_indices:
            similar_relation_url = self.id2rel.get(idx)
            if not similar_relation_url:
                logging.warning(f"Relation ID {idx} not found in id2rel mapping.")
                continue

            logging.debug(f"Found similar relation URL: {similar_relation_url}. Querying entity {entity_url}")
            sparql_result = self.sparql_handler.execute_sparql(entity_url, similar_relation_url)

            if sparql_result:
                formatted_result = self.format_sparql_result(sparql_result)
                return f"Embedded Answer: {self.sparql_handler.get_entity_label(formatted_result)}"

        return f"No results found for similar relations to '{relation_url}' with entity '{entity_url}'."

    def format_sparql_result(self, sparql_result):
        """
        Format the SPARQL result into a human-readable string.
        """
        if not sparql_result:
            return "No result found."

        results_list = []

        logging.debug(f"SPARQL result structure: {sparql_result}")

        try:
            for row in sparql_result:
                value = row.get('value') if isinstance(row, dict) else row[0]
                if value:
                    results_list.append(str(value))
        except Exception as e:
            logging.error(f"Error processing SPARQL result: {e}")
            return "Error processing results."

        return ", ".join(results_list) if results_list else "No valid results."

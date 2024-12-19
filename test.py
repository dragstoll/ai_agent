import pandas as pd
import numpy as np
from rdflib import Graph, URIRef, Literal
import logging
import json
from collections import defaultdict


class KnowledgeGraphProcessor:
    def __init__(self, properties_json_path, entities_json_path):
        self.property_url = self.load_json(properties_json_path)
        self.entity_url = self.load_json(entities_json_path)

    def load_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f'File not found: {file_path}')
            return {}
        except json.JSONDecodeError as e:
            logging.error(f'Error decoding JSON file {file_path}: {e}')
            return {}

    def get_property_label(self, property_url):
        return self.property_url.get(property_url, None)

    def get_entity_label(self, entity_url):
        return self.entity_url.get(entity_url, None)

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

    def format_response(self, subject_label, predicate_label, object_label):
        """Format response string based on predicate type"""
        predicate = predicate_label.lower()
        if predicate in ['box office', 'publication date']:
            return f"{subject_label} {predicate} {object_label}"
        if predicate.startswith('is indirectly'):
            return f"{subject_label} {predicate} {object_label}"
        return f"{subject_label} {predicate} {object_label}"

    def process_crowdsourced_data_with_labels(self, file_path, graph_path, graph_format='turtle'):
        try:
            data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

            column_mapping = {
                'HITId': 'hitid',
                'HITTypeId': 'hittypeid',
                'Input1ID': 'input1id',
                'Input2ID': 'input2id',
                'Input3ID': 'input3id',
                'AnswerLabel': 'answerlabel',
                'LifetimeApprovalRate': 'lifetimeapprovalrate',
                'WorkTimeInSeconds': 'worktimeinseconds',
                'WorkerId': 'workerid'
            }
            data = data.rename(columns=column_mapping)

            malicious_workers = self.identify_malicious_workers(data)
            valid_data = data[~data['workerid'].isin(malicious_workers)].copy()

            valid_data.loc[:, 'input1id'] = valid_data['input1id'].str.lower().str.strip().str.replace(r'^wd:', '',
                                                                                                       regex=True)
            valid_data.loc[:, 'input2id'] = valid_data['input2id'].str.lower().str.strip().str.replace(r'^wdt:', '',
                                                                                                       regex=True)

            batch_kappas = {}
            for batch_id, batch_data in valid_data.groupby('hittypeid'):
                if len(batch_data['hitid'].unique()) >= 2:
                    kappa = self.calculate_fleiss_kappa(batch_data)
                    batch_kappas[batch_id] = kappa
                else:
                    batch_kappas[batch_id] = 0.0

            hit_data = []
            for hit_id, group in valid_data.groupby('hitid'):
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

            majority_answers = pd.DataFrame(hit_data)

            g = Graph()
            g.parse(graph_path, format=graph_format)

            responses = []
            for _, row in majority_answers.iterrows():
                # Get subject label with proper entity resolution
                subject_url = f'http://www.wikidata.org/entity/{row["input1id"].upper()}'
                subject_label = self.get_entity_label(subject_url) or self.get_formatted_entity_label(row["input1id"])

                # Get predicate label
                predicate_url = f'http://www.wikidata.org/prop/direct/{row["input2id"].upper()}'
                predicate_label = self.format_predicate(self.get_property_label(predicate_url) or row["input2id"])

                # Get object label with proper entity resolution
                object_label = self.get_formatted_entity_label(row['input3id'])

                subject = URIRef(subject_url)
                predicate = URIRef(predicate_url)
                obj = Literal(object_label)

                if row['majorityanswer'] == 'CORRECT':
                    g.add((subject, predicate, obj))
                elif row['majorityanswer'] == 'INCORRECT':
                    g.remove((subject, predicate, None))
                    g.add((subject, predicate, obj))

                # Format response string
                response_text = self.format_response(subject_label, predicate_label, object_label)

                dist = row['distribution']
                crowd_info = (
                    f'[Crowd, inter-rater agreement {row["batch_agreement"]:.3f}, '
                    f'The answer distribution for this specific task was {dist["CORRECT"]} support votes, '
                    f'{dist["INCORRECT"]} reject votes]'
                )

                responses.append(f'{response_text}. {crowd_info}')

            updated_graph_path = 'updated_graph.nt'
            g.serialize(updated_graph_path, format=graph_format)

            return {
                'updated_graph_path': updated_graph_path,
                'responses': responses
            }

        except Exception as e:
            logging.error(f'Error processing data: {str(e)}')
            raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    processor = KnowledgeGraphProcessor(
        properties_json_path='dataset/properties-url.json',
        entities_json_path='dataset/entity_url.json'
    )

    file_path = 'dataset/crowd_data/crowd_data.tsv'
    graph_path = 'dataset/14_graph.nt'

    try:
        result = processor.process_crowdsourced_data_with_labels(file_path, graph_path)
        print('\nResponses:')
        for response in result['responses']:
            print(response)
    except Exception as e:
        print(f'Error: {str(e)}')
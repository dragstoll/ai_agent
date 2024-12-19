import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
from statsmodels.stats.inter_rater import fleiss_kappa


def process_crowdsourced_data(file_path, graph_path, graph_format="turtle"):
    """
    Process crowdsourced data, filter malicious workers, aggregate answers,
    compute Fleiss' Kappa, and update the knowledge graph.

    :param file_path: Path to the crowdsourced TSV data file.
    :param graph_path: Path to the existing knowledge graph file.
    :param graph_format: Format of the knowledge graph (e.g., "turtle").
    :return: Dictionary with summary statistics and updated graph.
    """
    # Step 1: Load the crowdsourced dataset
    data = pd.read_csv(file_path, sep='\t')

    # Step 1.1: Filter malicious workers
    threshold_approval_rate = 75  # Example threshold
    threshold_time = 10  # Minimum reasonable completion time in seconds
    data['LifetimeApprovalRate'] = data['LifetimeApprovalRate'].str.replace('%', '').astype(float)
    valid_data = data[(data['LifetimeApprovalRate'] >= threshold_approval_rate) &
                      (data['WorkTimeInSeconds'] >= threshold_time)]

    # Step 2: Aggregate Answers Using Majority Voting
    def majority_vote(group):
        return group['AnswerLabel'].value_counts().idxmax()

    majority_answers = valid_data.groupby('HITId').apply(
        lambda group: pd.Series({
            "MajorityAnswer": majority_vote(group),
            "Input1ID": group['Input1ID'].iloc[0],
            "Input2ID": group['Input2ID'].iloc[0],
            "Input3ID": group['Input3ID'].iloc[0],
            "AnswerDistribution": group['AnswerLabel'].value_counts().to_dict()
        })
    ).reset_index()

    # Step 3: Compute Inter-Rater Agreement
    def compute_kappa(batch):
        votes = batch['AnswerLabel'].value_counts()
        matrix = pd.DataFrame([votes], columns=['CORRECT', 'INCORRECT']).fillna(0)
        return fleiss_kappa(matrix)

    kappa_values = valid_data.groupby('HITTypeId').apply(compute_kappa).to_dict()

    # Step 4: Load and Update the Knowledge Graph
    g = Graph()
    g.parse(graph_path, format=graph_format)

    for idx, row in majority_answers.iterrows():
        subject = URIRef(row['Input1ID'])
        predicate = URIRef(row['Input2ID'])
        obj = URIRef(row['Input3ID']) if row['MajorityAnswer'] == "CORRECT" else Literal(row['Input3ID'])

        if row['MajorityAnswer'] == "CORRECT":
            g.add((subject, predicate, obj))
        elif row['MajorityAnswer'] == "INCORRECT":
            g.remove((subject, predicate, None))
            g.add((subject, predicate, obj))

    # Save the updated graph
    updated_graph_path = "updated_graph.nt"
    g.serialize(updated_graph_path, format=graph_format)

    # Step 5: Summary Statistics
    engagement_stats = {
        "total_tasks": valid_data['HITId'].nunique(),
        "total_workers": valid_data['WorkerId'].nunique(),
        "average_time": valid_data['WorkTimeInSeconds'].mean(),
        "average_approval_rate": valid_data['LifetimeApprovalRate'].mean(),
        "correct_percentage": (majority_answers['MajorityAnswer'] == 'CORRECT').mean() * 100,
        "kappa_values": kappa_values,
    }

    return engagement_stats



# Example usage
file_path = "../dataset/crowd_data/crowd_data_olat_P344FullstopCorrected.tsv"
graph_path = "../dataset/14_graph.nt"

result = process_crowdsourced_data(file_path, graph_path, graph_format="turtle")

print("Engagement Statistics:")
print(result)

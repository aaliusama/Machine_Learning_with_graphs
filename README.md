# Machine Learning with Graphs

This repository contains a collection of self‑contained Jupyter notebooks exploring different graph machine‑learning tasks.  Each notebook focuses on a specific problem, from analysing the robustness of transportation networks to recommending films, classifying scientific articles and predicting missing links in a social network.  The aim of this project is to showcase how classical graph algorithms and modern embedding techniques can be applied to real‑world datasets.

## Contents

| Notebook | Description |
| --- | --- |
| **European_Transport_Robustness.ipynb** | Measures how vulnerable a road network is to targeted failures by iteratively removing the most central cities and tracking the size of the largest connected component. |
| **Movie_PageRank_Recommendations.ipynb** | Builds a bipartite graph of movies and their attributes (genres, actors, directors, etc.) and uses personalised PageRank to generate film recommendations given an initial query. |
| **APS_Journal_Prediction_Node2Vec.ipynb** | Learns node embeddings for a large citation graph from the American Physical Society and trains classifiers to predict the journal in which a paper was published. |
| **Southern_Women_Link_Prediction.ipynb** | Demonstrates link prediction on a small bipartite social network (women attending events) using node2vec embeddings and supervised learning. |

All input files used by the notebooks can be found in the `data/` directory.

## Robustness of a European Transport Network

**Dataset:**

- `data/europe.net` – a Pajek file encoding the connectivity between 1039 cities in Europe and 1305 undirected road segments.

**Objective:**  Assess the resilience of the European road network under targeted attacks.  The approach is to compute betweenness centrality for all nodes (cities) and iteratively remove the most central ones.  After each removal the size of the largest connected component (LCC) is recomputed.

**Methodology:**

1. Read the graph from the Pajek file and convert it to a simple undirected NetworkX graph.
2. Compute the betweenness centrality for all nodes.
3. Repeatedly remove the node with the highest betweenness centrality (15 times in total).
4. Track the size of the LCC before and after removals and compute the percentage of the remaining nodes it contains.

**Key results:**

- Initially all 1039 cities form a single connected component (100 % connectivity).
- After greedily removing the 15 most central cities (`Brest`, `Warsaw`, `Saint Petersburg`, `Kiev`, `Niš`, `Gdańsk`, `Mukachevo`, `Chişinău`, `Trieste`, `Vinnytsia`, `Budapest`, `Zagreb`, `Oradea`, `Kherson` and `Suceava`) the LCC still contains 442 cities.
- This final component represents **43.2 %** of the cities that remain in the graph, illustrating that the network retains a surprisingly large core even under targeted disruption.

The notebook `European_Transport_Robustness.ipynb` contains these steps and can be used to investigate alternative removal strategies or to compute different centrality measures.

## Film Recommendations via Personalised PageRank

**Dataset:**

- `data/movies_graph.net` – a network of 6577 nodes and 16842 edges representing movies (1337) and attribute “mode” nodes.  Movies are connected to modes corresponding to genres, actors, directors, languages, etc.

**Objective:**  Given a query (for example a specific actor or genre) recommend films that are strongly connected in the graph.  This task is framed as personalised PageRank: starting from the query nodes a random walker distributes probability mass through the network and nodes with higher PageRank are deemed more relevant.

**Methodology:**

1. Load the bipartite graph and label nodes to distinguish movies from modes.  Mode node labels start with `m-`, while movie nodes are plain text.
2. Identify candidate starting nodes for the personalised PageRank vector given a free‑text query (e.g. “Tom Hanks”, “Action”, “Brad Pitt”).
3. Run personalised PageRank with restart probability β = 0.2 using NetworkX.
4. Rank movies by their PageRank score and present the top recommendations.

**Example recommendations:**

- **Query: “Moana” (movie)** – the algorithm suggests other family‑friendly animated films such as *Saving Santa*, *Frozen* and *Smallfoot*.
- **Query: “Tom Hanks” (actor)** – top ranked films include *Forrest Gump*, *Captain Phillips* and *Sully*.
- **Query: “Brad Pitt & George Clooney” (actors)** – the PageRank algorithm highlights ensemble heist comedies such as *Ocean’s 12*, *Burn After Reading* and *Ocean’s 13*.

These results show that personalised PageRank can capture nuanced relationships in a heterogeneous network and produce sensible recommendations for various types of queries.

## Journal Classification on APS Citations with Node2Vec

**Dataset:**

- `data/aps_citations.net` – a citation network of papers published by the American Physical Society containing **56473** nodes and **200353** edges.
- Papers are annotated with their Digital Object Identifier (DOI), year and the journal (one of nine Physical Review journals).
- The repository also contains precomputed files produced by the notebook: 128‑dimensional node embeddings (`aps_embeddings.pkl`), metadata (`aps_metadata.pkl`), a train/test split (`aps_train_test_split.pkl`) and the classification results (`aps_classification_results.pkl`).
- Note: It may take several hours to generate embeddings from scratch. You can download it from: https://drive.google.com/file/d/1HN2u04vzaoDllj8JFqRShLZEoQiIZG5k/view?usp=sharing

**Objective:**  Predict the journal of papers published in 2013 using only the citation network and embeddings learned from prior years (2010–2012).

**Methodology:**

1. Parse the Pajek file to build an undirected citation graph and extract attributes (year, journal, DOI) for each paper.
2. Split the data chronologically: papers from 2010–2012 form the training set (41810 nodes) and those from 2013 form the test set (14663 nodes).
3. Train a node2vec model to learn 128‑dimensional embeddings from the full unlabeled graph.
4. Train three classifiers on the training embeddings: logistic regression (tuning regularisation strength), random forest and linear SVM.
5. Evaluate accuracy, macro–F1 and (for logistic regression) top‑3 accuracy on the 2013 test set.

**Key results:**

| Model | Accuracy | Macro‑F1 | Top‑3 |
| --- | --- | --- | --- |
| **Logistic regression (C = 0.5)** | **0.58** | **0.49** | **0.92** |
| **Random forest** | **0.72** | **0.55** | – |
| **Linear SVM** | **0.70** | **0.52** | – |

The random forest classifier achieves the highest accuracy (≈ 72 %) and macro–F1 (≈ 0.55) on the unseen 2013 papers.  Although logistic regression performs worse overall, its top‑3 accuracy exceeds 91 %, meaning that the true journal is usually among the top three predicted labels. Check notebook for a detailed classification report per journal.

## Link Prediction in the Southern Women Network

**Dataset:**

- `data/southern_women.net` – a small bipartite graph describing attendance of 18 women (type 1/2) at 14 events (type 3).  The network contains 32 nodes and 89 edges.

**Objective:**  Predict which woman–event pairs that are not currently observed would be most likely to occur, by learning node embeddings and training a binary classifier on pairs of nodes.

**Methodology:**

1. Read the graph, identify women (type 1/2) and events (type 3) and compute positive examples (existing edges).
2. Generate an equal number of negative examples by sampling unconnected woman–event pairs.
3. Split the edges into a training set (≈ 160 edges) and a test set (≈ 36 edges), ensuring both positive and negative samples are represented.
4. For each pair, create a feature vector by concatenating the node2vec embeddings of the two nodes (embedding dimension 32, so features have length 64).
5. Train logistic regression and random forest classifiers using 10 random splits to evaluate stability of the results.

**Key results (averaged over 10 runs):**

| Model | Accuracy (mean ± SD) | ROC AUC (mean ± SD) | F1 Score (mean ± SD) |
| --- | --- | --- | --- |
| **Logistic regression** | 0.59 ± 0.12 | 0.75 ± 0.06 | 0.54 ± 0.18 |
| **Random forest** | 0.76 ± 0.10 | 0.92 ± 0.06 | 0.72 ± 0.11 |

The random forest consistently outperforms logistic regression, achieving an average accuracy of 76 % and ROC AUC around 0.92.  These results demonstrate the effectiveness of combining node embeddings with supervised models for link prediction on small networks.

## Usage

To explore the analyses yourself:

1. Clone or download this repository and change into its directory.
2. Create a Python environment with the required libraries.  At minimum you will need [`networkx`](https://networkx.org/), [`numpy`](https://numpy.org/), [`scikit‑learn`](https://scikit-learn.org/), and [`node2vec`](https://github.com/eliorc/node2vec). 
3. Start a Jupyter notebook server (`jupyter notebook`) and open any of the `.ipynb` files.
4. Ensure that the working directory is the repository root so that the notebooks can load the files in the `data/` directory.

The notebooks are designed to be self‑explanatory and include comments throughout.  Feel free to modify the code to experiment with alternative parameters (e.g. different centrality measures, embedding dimensions or classifier hyperparameters).

---

If you find this repository useful or have suggestions for improvement, please feel free to open an issue or submit a pull request.

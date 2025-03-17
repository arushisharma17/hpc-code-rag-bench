import matplotlib.pyplot as plt
import numpy as np

retrieval_sources = ["No sources", "Stackoverflow \n-\n GIST-Embedding",
                     "Stackoverflow \n-\n codesearch-distilroberta", "Github \n-\n GIST-Embedding",
                     "Github \n-\n codesearch-distilroberta"]
language_models = [
    "Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-Coder-1.5B-Instruct",
    "SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-Instruct",
    "llama-3.2-3B"
]

# Scores for each models
no_sources = [ 0.8241, 0.8955, 0.0380, 0.2017, 0.6168, 0.8361 ]
stackoverflow_gist = [ 0.7222, 0.8376, 0.0480, 0.2330, 0.6029, 0 ]
stackoverflow_codesearch = [ 0.7672, 0.7349, 0.0233, 0.1514, 0.5049, 0 ]
github_gist = [ 0.7999, 0.8495, 0.0231, 0.2564, 0.5275, 0 ]
github_codesearch = [ 0.7869, 0.8731, 0.0260, 0.1419, 0.5322, 0 ]

# Convert data for plotting
scores = np.array([no_sources, stackoverflow_gist, stackoverflow_codesearch, github_gist, github_codesearch])

x = np.arange(len(retrieval_sources))
width = 0.15  # Bar width

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))
for i, method in enumerate(language_models):
    ax.bar(x + i * width - (len(language_models) / 2) * width, scores[:, i], width, label=method)

# Labels and title
ax.set_xlabel("Retrieval Sources", weight='bold')
ax.set_ylabel("Score", weight='bold')
ax.set_title("code-bert-f3-score")
ax.set_xticks(x)
ax.set_xticklabels(retrieval_sources)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Methods")

# Show plot
plt.tight_layout()
plt.show()

# EEG Embedding Artifacts

Feature vectors and similarity matrices from the embedding crate.

## Regenerate

```bash
cd rust
cargo run -p pipeline --release -- --packets 10000
```

The pipeline extracts 11-dimensional feature vectors from EEG windows:
- 5 relative band powers (delta, theta, alpha, beta, gamma)
- 5 log-absolute band powers
- 1 RMS amplitude

## Visualization (Python)

Feature vectors can be exported to CSV and visualized with the Python pipeline:

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

features = np.loadtxt("embeddings.csv", delimiter=",")
proj = TSNE(n_components=2).fit_transform(features)
plt.scatter(proj[:, 0], proj[:, 1], s=1)
plt.title("EEG Feature Embeddings (t-SNE)")
plt.savefig("artifacts/embeddings/tsne.png")
```

## Cosine Similarity

The `embedding::cosine_similarity` function compares trial-to-trial consistency:
- Same EEG pattern: similarity ~1.0
- Different frequency bands (alpha vs beta): similarity < 0.99
- Orthogonal signals: similarity ~0.0

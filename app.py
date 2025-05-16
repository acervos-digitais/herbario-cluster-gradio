import gradio as gr
import json
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances

with open("./20250515_embeddings.json", "r") as ifp:
  data = json.load(ifp)

ids = np.array(list(data.keys()))
clip_embeddings = np.array([v["clip"] for v in data.values()])
siglip_embeddings = np.array([v["siglip2"] for v in data.values()])

def tsne_kmeans(emb_raw, n_clusters=8, n_components=3, perplexity=30):
  mTSNE = TSNE(n_components=n_components, perplexity=perplexity, random_state=10)
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_reduced = mTSNE.fit_transform(StandardScaler().fit_transform(emb_raw))
  emb_clusters = mCluster.fit_predict(emb_reduced)

  return emb_reduced, emb_clusters, mCluster.cluster_centers_

def get_cluster_data(n_clusters):
  embs, clusters, centers = tsne_kmeans(clip_embeddings, n_clusters=n_clusters)
  cluster_distances = euclidean_distances(embs, centers)

  i_c_d = zip(ids.tolist(), clusters.tolist(), cluster_distances.tolist())
  return {id: {"cluster": c, "distances": [round(d,3) for d in ds]} for  id,c,ds in i_c_d}

with gr.Blocks() as demo:
  gr.Interface(
    fn=get_cluster_data,
    inputs="number",
    outputs="json",
    allow_flagging="never",
  )

if __name__ == "__main__":
   demo.launch()

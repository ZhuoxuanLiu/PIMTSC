import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        self.faster_search = False

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        if self.faster_search:
            self.search_index.train(features)
        self.search_index.add(features)

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self.search_index.train(index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        super().__init__(on_gpu, num_workers)
        self.faster_search = True

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)


class NearestNeighbourInference(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, distance, index, class_indicator):
        # distance, index: [B, K]  K -> features[K] [D]
        # class_indicator [N, num_classes]
        bs = distance.shape[0]
        prob = torch.exp(-torch.tensor(distance))
        prob_indicator = torch.zeros(bs)
        for b in range(bs):
            prob_indicator[b] = torch.argmax(torch.sum(torch.einsum('kn, k -> kn', class_indicator[index[b]], prob[b]), dim=0))
        return prob_indicator.to(class_indicator)
                
                
                
class NearestNeighbourScorer(object):
    def __init__(self, n_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """

        self.n_neighbours = n_neighbours
        self.nn_method = nn_method
        self.nn_inference = NearestNeighbourInference()
        self.nn = lambda query: self.nn_method.run(
            n_neighbours, query
        )

    def fit(self, features) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            features: Tensor [N, B, D]
            features VECTORS produced by some backbone network 
        """
        self.features = features.reshape(-1, features.shape[-1]).cpu().numpy()
        self.nn_method.fit(self.features)

    def predict(
        self, query_features, class_indicator
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
            query_features: Tensor [B, D]
            query_features VECTORS produced by some backbone network 
        """
        query_features = query_features.numpy()
        query_distances, query_nns = self.nn(query_features)
        result = self.nn_inference(query_distances, query_nns, class_indicator) # [B]
        return result

    @staticmethod
    def _feature_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(self._feature_file(save_folder, prepend), self.features)

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._feature_file(load_folder, prepend)):
            self.features = self._load(self._feature_file(load_folder, prepend))
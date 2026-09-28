'''
Visual recommendation: nearest-neighbour search over the ResNet feature layer.

The 64-d global-pool features saved by train_n_test.py (FLAGS.fc_path, a .npy file) are
L2-normalised and indexed with cosine distance; each query image is matched to the k most
similar catalogue items.

Usage:
    python recommend.py --features data/catalog_fc.npy --images data/catalog_images.csv \
                        --query 0 --k 5 --out recommendations.png
'''
import argparse

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Recommender:
    def __init__(self, features, k=5):
        self.features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        self.index = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(self.features)

    def similar(self, query_idx, k=5):
        '''Indices of the k items most similar to catalogue item query_idx (excluding itself).'''
        _, idx = self.index.kneighbors(self.features[query_idx:query_idx + 1], n_neighbors=k + 1)
        return [i for i in idx[0] if i != query_idx][:k]

    def similar_to_vector(self, vector, k=5):
        '''Indices of the k catalogue items closest to an arbitrary feature vector.'''
        v = vector / (np.linalg.norm(vector) + 1e-12)
        _, idx = self.index.kneighbors(v.reshape(1, -1), n_neighbors=k)
        return list(idx[0])


def save_grid(images, out_path, tile=128):
    '''images: list of HxWx3 BGR arrays; the first is the query.'''
    tiles = [cv2.resize(img, (tile, tile)) for img in images]
    tiles[0] = cv2.copyMakeBorder(tiles[0][4:-4, 4:-4], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    gap = np.full((tile, 12, 3), 255, np.uint8)
    row = [tiles[0], gap, gap]
    for t in tiles[1:]:
        row += [t, gap]
    cv2.imwrite(out_path, np.hstack(row[:-1]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='.npy array of shape [N, 64]')
    ap.add_argument('--images', required=True, help='csv with an image_path column, same order as features')
    ap.add_argument('--query', type=int, default=0)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='recommendations.png')
    args = ap.parse_args()

    feats = np.load(args.features)
    paths = pd.read_csv(args.images)['image_path'].tolist()
    rec = Recommender(feats, k=args.k)
    idx = rec.similar(args.query, k=args.k)
    print('Query:', paths[args.query])
    for rank, i in enumerate(idx, 1):
        print('  %d. %s' % (rank, paths[i]))
    save_grid([cv2.imread(paths[args.query])] + [cv2.imread(paths[i]) for i in idx], args.out)
    print('Saved', args.out)

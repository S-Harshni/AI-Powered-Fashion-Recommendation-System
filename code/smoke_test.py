'''
End-to-end smoke test without the DeepFashion download.

Builds the real ResNet (simple_resnet.inference) and training loss from train_n_test.Train,
trains for a few hundred steps on synthetic 64x64 "garments" (6 classes that differ in colour
and stripe pattern), then extracts the 64-d feature layer and checks that nearest-neighbour
retrieval returns items of the same class. Proves the TF2 port, the training graph, feature
extraction and the recommender all work together.

Usage: python smoke_test.py [--steps 300] [--grid ../docs/smoke_test_recommendations.png]
'''
import argparse
import os
import sys

import cv2
import numpy as np

# hyper_parameters parses flags with known_only=True, so --steps/--grid pass through.
from train_n_test import Train, TRAIN_BATCH_SIZE, tf  # noqa: E402  (tf is tensorflow.compat.v1)
from simple_resnet import inference, NUM_LABELS  # noqa: E402
from hyper_parameters import FLAGS  # noqa: E402
from recommend import Recommender, save_grid  # noqa: E402

IMG = 64
PALETTE = [(40, 40, 200), (40, 160, 40), (200, 80, 40), (30, 190, 220), (150, 50, 150), (90, 90, 90)]


def make_garment(label, rng):
    '''Synthetic garment: class colour + class-specific stripe orientation/frequency + noise.'''
    y, x = np.mgrid[0:IMG, 0:IMG]
    angle = label * np.pi / NUM_LABELS
    freq = 0.15 + 0.05 * (label % 3)
    stripes = 0.5 + 0.5 * np.sin(freq * (x * np.cos(angle) + y * np.sin(angle)) + rng.uniform(0, 6.28))
    base = np.array(PALETTE[label], np.float32) * (0.75 + 0.5 * rng.uniform())
    img = stripes[..., None] * base[None, None, :] + rng.normal(0, 12, (IMG, IMG, 3))
    # Garment silhouette: a centred box with random size (the "bounding box" target).
    h, w = rng.integers(36, 60), rng.integers(30, 56)
    top, left = (IMG - h) // 2, (IMG - w) // 2
    mask = np.zeros((IMG, IMG, 1), np.float32)
    mask[top:top + h, left:left + w] = 1
    img = img * mask + 235 * (1 - mask)
    bbox = np.array([left, top, left + w, top + h], np.float32) / IMG
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def batch(rng, n):
    labels = rng.integers(0, NUM_LABELS, n)
    imgs, boxes = zip(*(make_garment(int(l), rng) for l in labels))
    raw = np.stack(imgs)
    x = (raw.astype(np.float32) - raw.mean()) / (raw.std() + 1e-6)  # same whitening idea as fashion_input
    return x, labels.astype(np.int32), np.stack(boxes), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--grid', default=os.path.join(os.path.dirname(__file__), '..', 'docs', 'smoke_test_recommendations.png'))
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(0)
    tf.set_random_seed(0)

    t = Train()
    logits, bbox, features = inference(t.image_placeholder, n=FLAGS.num_residual_blocks, reuse=False,
                                       keep_prob_placeholder=t.dropout_prob_placeholder)
    reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([t.loss(logits, bbox, t.label_placeholder, t.bbox_placeholder)] + reg)
    error = t.top_k_error(tf.nn.softmax(logits), t.label_placeholder, 1)
    train_op = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        first = None
        for step in range(args.steps):
            x, y, b, _ = batch(rng, TRAIN_BATCH_SIZE)
            _, l, e = sess.run([train_op, loss, error], {t.image_placeholder: x, t.label_placeholder: y,
                                                         t.bbox_placeholder: b, t.dropout_prob_placeholder: 0.5})
            first = l if first is None else first
            if step % 50 == 0 or step == args.steps - 1:
                print('step %4d  loss %.3f  train top-1 error %.2f' % (step, l, e))

        # Feature extraction on a held-out catalogue, then retrieval.
        feats, labels, raws = [], [], []
        for _ in range(6):
            x, y, b, raw = batch(rng, TRAIN_BATCH_SIZE)
            feats.append(sess.run(features, {t.image_placeholder: x, t.dropout_prob_placeholder: 1.0}))
            labels.append(y); raws.append(raw)
    feats, labels, raws = np.concatenate(feats), np.concatenate(labels), np.concatenate(raws)

    rec = Recommender(feats, k=5)
    hits = [np.mean(labels[rec.similar(i, 5)] == labels[i]) for i in range(len(feats))]
    precision = float(np.mean(hits))
    print('\nfeature matrix %s, precision@5 of nearest-neighbour retrieval: %.2f (chance %.2f)'
          % (feats.shape, precision, 1 / NUM_LABELS))

    rows = []
    for q in [int(np.where(labels == c)[0][0]) for c in range(NUM_LABELS)]:
        save_grid([raws[q][..., ::-1]] + [raws[i][..., ::-1] for i in rec.similar(q, 5)], '/tmp/_row.png')
        rows.append(cv2.imread('/tmp/_row.png'))
    os.makedirs(os.path.dirname(os.path.abspath(args.grid)), exist_ok=True)
    cv2.imwrite(args.grid, np.vstack([np.vstack([r, np.full((10, r.shape[1], 3), 255, np.uint8)]) for r in rows])[:-10])
    print('saved', os.path.abspath(args.grid))

    ok = l < first and precision > 0.8
    print('SMOKE TEST', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

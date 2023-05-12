# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import faiss
from datasets import load_sift1M


print("load data")

xb, xq, xt, gt = load_sift1M()
nq, d = xq.shape

ncent = 256

variants = [(name, getattr(faiss.ScalarQuantizer, name))
            for name in dir(faiss.ScalarQuantizer)
            if name.startswith('QT_')]

quantizer = faiss.IndexFlatL2(d)
for name, qtype in variants:

    print("============== test", name)

    for rsname, vals in [('RS_minmax',
                          [-0.4, -0.2, -0.1, -0.05, 0.0, 0.1, 0.5]),
                         ('RS_meanstd', [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]),
                         ('RS_quantiles', [0.02, 0.05, 0.1, 0.15]),
                         ('RS_optim', [0.0])]:
        for val in vals:
            print("%-15s %5g    " % (rsname, val), end=' ')
            index = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent,
                                                  qtype, faiss.METRIC_L2)
            index.nprobe = 16
            index.sq.rangestat = getattr(faiss.ScalarQuantizer,
                                      rsname)

            index.rangestat_arg = val

            index.train(xt)
            index.add(xb)
            t0 = time.time()
            D, I = index.search(xq, 100)
            t1 = time.time()

            for rank in 1, 10, 100:
                n_ok = (I[:, :rank] == gt[:, :1]).sum()
                print("%.4f" % (n_ok / float(nq)), end=' ')
            print("   %.3f s" % (t1 - t0))

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import re


def get_code_size(d, indexkey):
    """ size of one vector in an index in dimension d
    constructed with factory string indexkey"""

    if indexkey == "Flat":
        return d * 4

    if indexkey.endswith(",RFlat"):
        return d * 4 + get_code_size(d, indexkey[:-len(",RFlat")])

    if mo := re.match("IVF\\d+(_HNSW32)?,(.*)$", indexkey):
        return get_code_size(d, mo[2])

    if mo := re.match("IVF\\d+\\(.*\\)?,(.*)$", indexkey):
        return get_code_size(d, mo[1])

    if mo := re.match("IMI\\d+x2,(.*)$", indexkey):
        return get_code_size(d, mo[1])

    if mo := re.match("(.*),Refine\\((.*)\\)$", indexkey):
        return get_code_size(d, mo[1]) + get_code_size(d, mo[2])

    if mo := re.match('PQ(\\d+)x(\\d+)(fs|fsr)?$', indexkey):
        return (int(mo[1]) * int(mo[2]) + 7) // 8

    if mo := re.match('PQ(\\d+)\\+(\\d+)$', indexkey):
        return int(mo[1]) + int(mo[2])

    if mo := re.match('PQ(\\d+)$', indexkey):
        return int(mo[1])

    if indexkey in ["HNSW32", "HNSW32,Flat"]:
        return d * 4 + 64 * 4 # roughly

    if indexkey == 'SQ4':
        return (d + 1) // 2
    elif indexkey == 'SQ6':
        return (d * 6 + 7) // 8
    elif indexkey == 'SQ8':
        return d
    elif indexkey == 'SQfp16':
        return d * 2

    if mo := re.match('PCAR?(\\d+),(.*)$', indexkey):
        return get_code_size(int(mo[1]), mo[2])
    if mo := re.match('OPQ\\d+_(\\d+),(.*)$', indexkey):
        return get_code_size(int(mo[1]), mo[2])
    if mo := re.match('OPQ\\d+,(.*)$', indexkey):
        return get_code_size(d, mo[1])
    if mo := re.match('RR(\\d+),(.*)$', indexkey):
        return get_code_size(int(mo[1]), mo[2])
    raise RuntimeError(f"cannot parse {indexkey}")



def reverse_index_factory(index):
    """
    attempts to get the factory string the index was built with
    """
    index = faiss.downcast_index(index)
    if isinstance(index, faiss.IndexFlat):
        return "Flat"
    if isinstance(index, faiss.IndexIVF):
        quantizer = faiss.downcast_index(index.quantizer)

        if isinstance(quantizer, faiss.IndexFlat):
            prefix = "IVF%d" % index.nlist
        elif isinstance(quantizer, faiss.MultiIndexQuantizer):
            prefix = "IMI%dx%d" % (quantizer.pq.M, quantizer.pq.nbit)
        elif isinstance(quantizer, faiss.IndexHNSW):
            prefix = "IVF%d_HNSW%d" % (index.nlist, quantizer.hnsw.M)
        else:
            prefix = "IVF%d(%s)" % (index.nlist, reverse_index_factory(quantizer))

        if isinstance(index, faiss.IndexIVFFlat):
            return f"{prefix},Flat"
        if isinstance(index, faiss.IndexIVFScalarQuantizer):
            return f"{prefix},SQ8"

    raise NotImplementedError()

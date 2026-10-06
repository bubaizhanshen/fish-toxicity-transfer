"""Fold-local molecular and experimental-context preprocessing."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

SEED = 20261006


def feature_matrices(train, query, fp, cats, family):
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    enc.fit(train[cats].fillna("unknown"))
    projection = None
    if family == "mlp":
        chemical_ids = np.unique(train.chemical_index.to_numpy(int))
        components = min(128, len(chemical_ids) - 1, fp.shape[1] - 1)
        if components < 1:
            raise ValueError("No independent chemical basis for MLP projection")
        projection = TruncatedSVD(n_components=components, n_iter=5, random_state=SEED)
        projection.fit(fp[chemical_ids])

    def make(z):
        molecular = fp[z.chemical_index.to_numpy(int)]
        if projection is not None:
            molecular = sparse.csr_matrix(projection.transform(molecular))
        return sparse.hstack(
            [
                molecular,
                enc.transform(z[cats].fillna("unknown")),
                sparse.csr_matrix(
                    np.log10(z.duration_h.clip(lower=1).to_numpy())
                    .astype(np.float32)
                    .reshape(-1, 1)
                ),
            ],
            format="csr",
            dtype=np.float32,
        )

    a, b = (make(train), make(query))
    scaler = None
    if family in [
        "ridge",
        "elastic_net",
        "linear_svr",
        "nystroem_ridge",
        "rbf_svr_low",
        "mlp",
    ]:
        scaler = StandardScaler(with_mean=False)
        a = scaler.fit_transform(a)
        b = scaler.transform(b)
    if family == "mlp":
        a = a.toarray()
        b = b.toarray()
    return (a, b, dict(encoder=enc, projection=projection, scaler=scaler))

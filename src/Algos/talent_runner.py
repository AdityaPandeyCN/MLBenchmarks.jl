import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from TALENT.model.utils import get_method, set_seeds


def _args(config, model_type, batch_size, max_epoch, seed, save_path, cat_policy, num_policy, normalization, use_float):
    os.makedirs(save_path, exist_ok=True)
    config = copy.deepcopy(config)
    config.setdefault("training", {})
    config["training"].setdefault("n_bins", 2)
    return SimpleNamespace(
        config=config,
        model_type=model_type,
        batch_size=int(batch_size),
        max_epoch=int(max_epoch),
        seed=int(seed),
        save_path=save_path,
        cat_policy=cat_policy,
        num_policy=num_policy,
        normalization=normalization,
        num_nan_policy="mean",
        cat_nan_policy="new",
        use_float=bool(use_float),
    )


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def run_regression(
    model_type,
    config,
    x_train,
    y_train,
    x_eval,
    y_eval,
    x_test,
    y_test,
    batch_size,
    max_epoch,
    seed,
    save_path,
    cat_policy,
    num_policy,
    normalization,
    use_float=True,
):
    set_seeds(int(seed))
    dtype = np.float32 if use_float else np.float64
    x_train = np.asarray(x_train, dtype=dtype)
    x_eval = np.asarray(x_eval, dtype=dtype)
    x_test = np.asarray(x_test, dtype=dtype)
    y_train = np.asarray(y_train, dtype=dtype).reshape(-1)
    y_eval = np.asarray(y_eval, dtype=dtype).reshape(-1)
    y_test = np.asarray(y_test, dtype=dtype).reshape(-1)

    train_val_data = (
        {"train": x_train, "val": x_eval},
        None,
        {"train": y_train, "val": y_eval},
    )
    eval_data = ({"test": x_eval}, None, {"test": y_eval})
    test_data = ({"test": x_test}, None, {"test": y_test})
    info = {
        "task_type": "regression",
        "n_num_features": int(x_train.shape[1]),
        "n_cat_features": 0,
    }

    args = _args(
        config,
        model_type,
        batch_size,
        max_epoch,
        seed,
        save_path,
        cat_policy,
        num_policy,
        normalization,
        use_float,
    )
    method = get_method(model_type)(args, True)

    t0 = time.perf_counter()
    method.fit(train_val_data, info, train=True, config=config)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, _, pred_eval = method.predict(eval_data, info, "best-val")
    eval_infer_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, _, pred_test = method.predict(test_data, info, "best-val")
    test_infer_time = time.perf_counter() - t0

    return {
        "train_time": float(train_time),
        "eval_infer_time": float(eval_infer_time),
        "test_infer_time": float(test_infer_time),
        "best_nround": int(method.trlog.get("best_epoch", -1)) + 1,
        "device": str(args.device),
        "pred_eval": _to_numpy(pred_eval),
        "pred_test": _to_numpy(pred_test),
    }


def _load_matrix(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _load_vector(path):
    return np.loadtxt(path, delimiter=",", ndmin=1).reshape(-1)


def main(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)

    config = spec["config"]
    if spec["model_type"] == "modernNCA":
        config["model"].setdefault("num_embeddings", None)

    result = run_regression(
        spec["model_type"],
        config,
        _load_matrix(spec["x_train_path"]),
        _load_vector(spec["y_train_path"]),
        _load_matrix(spec["x_eval_path"]),
        _load_vector(spec["y_eval_path"]),
        _load_matrix(spec["x_test_path"]),
        _load_vector(spec["y_test_path"]),
        spec["batch_size"],
        spec["max_epoch"],
        spec["seed"],
        spec["save_path"],
        spec["cat_policy"],
        spec["num_policy"],
        spec["normalization"],
        spec["use_float"],
    )

    np.savetxt(spec["pred_eval_path"], result["pred_eval"], delimiter=",")
    np.savetxt(spec["pred_test_path"], result["pred_test"], delimiter=",")
    with open(spec["result_path"], "w") as f:
        for key in ("train_time", "best_nround", "device"):
            f.write(f"{key}={result[key]}\n")


if __name__ == "__main__":
    main(sys.argv[1])

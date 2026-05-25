# -*- coding: utf-8 -*-
"""
MNIST 데이터 로드 유틸리티.

데이터 파일이 로컬에 없으면 TensorFlow/Keras 공개 URL에서 내려받아 `data/`에 저장합니다.
"""

import os
import urllib.request

import numpy as np


def load_mnist(data_dir="data", validation_split=0.2, seed=42):
    """
    MNIST 손글씨 숫자 데이터셋을 로드합니다.
    data/mnist.npz가 있으면 로컬 파일을 사용하고, 없으면 URL에서 다운로드 후 data/에 저장합니다.
    훈련 데이터 중 validation_split 비율만큼은 검증 데이터로 분리합니다.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        - x: (N, 784) float32, 0~1 정규화
        - y: (N,) int, 0~9 레이블
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "mnist.npz")

    if not os.path.isfile(local_path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, local_path)

    with np.load(local_path) as data:
        x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
        x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
        y_train = data["y_train"]
        y_test = data["y_test"]

    if not 0 <= validation_split < 1:
        raise ValueError("validation_split은 0 이상 1 미만이어야 합니다.")

    val_size = int(len(x_train) * validation_split)
    if val_size > 0:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(x_train))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
    else:
        x_val = x_train[:0]
        y_val = y_train[:0]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

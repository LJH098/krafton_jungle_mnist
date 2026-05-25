# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.
    network = model

    iter_nums = epochs
    train_size = x_train.shape[0]
    train_loss_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        y_pred = network.forward(x_batch, train=True)
        loss = cross_entropy_loss(y_pred, y_batch)

        current_batch_size = x_batch.shape[0]
        dout = y_pred.copy()
        dout[np.arange(current_batch_size), y_batch] -= 1
        dout /= current_batch_size

        network.backward(dout)
        optimizer.update(network.params, network.grads)
        train_loss_list.append(loss)

    return train_loss_list



def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

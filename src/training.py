# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np
# from dataset.mnist import load_mnist
from data import load_mnist
from network import NeuralNetwork

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
    # 데이터 읽기
    network = model

    # 하이퍼파라미터
    train_size = x_train.shape[0]
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(epochs):

        epoch_loss = 0
        batch_num = 0

        indices = np.random.permutation(train_size)

        for j in range(0, train_size, batch_size):
            



            # 미니배치 획득
            x_batch = x_train[j:j + batch_size]
            t_batch = y_train[j:j + batch_size]
        
            # forward/loss/backward/update를 수행
        

            Loss = model.loss(x_batch, t_batch)
            y_pred = model.forward(x_batch, train=True)

            SWL_dout = y_pred.copy()
            SWL_dout[np.arange(x_batch.shape[0]), t_batch] -= 1
            SWL_dout /= x_batch.shape[0]

            grad = model.backward(SWL_dout)
            
            optimizer.update(model.params, model.grads)
        
            # 학습 경과 기록
            loss = network.loss(x_batch, t_batch)

            epoch_loss  += loss
            batch_num += 1
        train_loss_list.append(epoch_loss / batch_num)
    

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

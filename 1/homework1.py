import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['font.family'] = ['Microsoft Yahei']
#ax = plt.axes(projection='3d')

# 生成训练数据集，每类100个数据


def get_train_data(data_size=100):
    data_label = np.zeros((2 * data_size, 1))  # class 1
    x1 = np.reshape(np.random.normal(1, 0.6, data_size), (data_size, 1))
    y1 = np.reshape(np.random.normal(1, 0.8, data_size), (data_size, 1))
    data_train = np.concatenate((x1, y1), axis=1)
    data_label[0:data_size, :] = 0
    # class 2
    x2 = np.reshape(np.random.normal(-1, 0.3, data_size), (data_size, 1))
    y2 = np.reshape(np.random.normal(-1, 0.5, data_size), (data_size, 1))
    data_train = np.concatenate(
        (data_train, np.concatenate((x2, y2), axis=1)), axis=0)
    data_label[data_size:2 * data_size, :] = 1
    #ax.scatter3D(x1, y1, np.ones(100))
    #ax.scatter3D(x2, y2, -np.ones(100))
    plt.scatter(x1, y1, label='训练集1')
    plt.scatter(x2, y2, label='训练集2')
    return data_train, data_label


# 生成测试数据集，每类10个
def get_test_data(data_size=10):
    testdata_label = np.zeros((2 * data_size, 1))  # class 1
    x1 = np.reshape(np.random.normal(1, 0.6, data_size), (data_size, 1))
    y1 = np.reshape(np.random.normal(1, 0.8, data_size), (data_size, 1))
    data_test = np.concatenate((x1, y1), axis=1)
    testdata_label[0:data_size, :] = 0
    # class 2
    x2 = np.reshape(np.random.normal(-1, 0.3, data_size), (data_size, 1))
    y2 = np.reshape(np.random.normal(-1, 0.5, data_size), (data_size, 1))
    data_test = np.concatenate(
        (data_test, np.concatenate((x2, y2), axis=1)), axis=0)
    testdata_label[data_size:2 * data_size, :] = 1
    plt.scatter(x1, y1, label='测试集1', marker='^')
    plt.scatter(x2, y2, label='测试集2', marker='^')
    return data_test, testdata_label


def linear():
    x, y = get_train_data()
    num_feature = x.shape[1]
    w = np.zeros((num_feature, 1))
    b = 0
    for i in range(10000):
        h = x.dot(w) + b
        loss = 0.5 * np.sum(np.square(h - y)) / x.shape[0]
        dw = x.T.dot((h-y)) / x.shape[0]
        db = np.sum((h-y)) / x.shape[0]
        w += -0.001*dw
        b += -0.001*db
        if i % 500 == 0:
            print('iters = %d,loss = %f' % (i, loss))

    X, Y = np.meshgrid(np.linspace(-3, 4, 50), np.linspace(-3, 4, 50))
    ax.plot_surface(X, Y, w[0] * X + w[1] * Y + b)
    plt.show()


def lda():
    x, y = get_train_data()
    test_x, test_y = get_test_data()
    train_data_x = x[(y == 0).flatten()]
    train_data_y = x[(y == 1).flatten()]
    train_data = np.append(train_data_x, train_data_y, axis=0)
    test_data_x = test_x[(test_y == 0).flatten()]
    test_data_y = test_x[(test_y == 1).flatten()]
    mu_x = np.mean(train_data_x, axis=0)
    mu_y = np.mean(train_data_y, axis=0)
    sw = np.dot((train_data_x - mu_x).T, train_data_x - mu_x) + \
        np.dot((train_data_y - mu_y).T, train_data_y - mu_y)
    w = np.dot(np.linalg.inv(sw), (mu_x - mu_y)).reshape(1, -1)  # (1, p)
    print(w)
    # 投影
    test_xp = w.dot(test_data_x.T)
    test_yp = w.dot(test_data_y.T)
    test_xp = w.T.dot(test_xp) / np.dot(w, w.T)
    test_yp = w.T.dot(test_yp) / np.dot(w, w.T)
    plt.scatter(test_xp[0], test_xp[1],
                label='测试集1投影', marker='v')
    plt.scatter(test_yp[0], test_yp[1],
                label='测试集2投影', marker='v')
    X = np.linspace(-3, 4, 50)
    Y = X*w[0, 1]/w[0, 0]
    #Y = X*w[0,1]
    plt.plot(X, Y, label='投影线')
    midx = (w.dot(mu_x.T) + w.dot(mu_y.T)) / 2
    mid_y = -midx * w[0, 0] / w[0, 1]
    X_ = np.linspace(-1, 1, 100)
    plt.plot(X_, (-w[0, 0] / w[0, 1] * (X_ - midx) +
             mid_y).flatten(), label='分类线')
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # linear()
    lda()

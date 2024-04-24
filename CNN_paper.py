# pylint: disable=maybe-no-member
# from plot_model import plot_results
import torch
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import torch.utils.data as Data
from sklearn.model_selection import train_test_split




def read_data():
    flux =[]
    scls = []
    scls = onehot(scls)
    print(flux.shape,scls.shape)
    fluxTR, fluxTE, clsTR, clsTE = train_test_split(flux, scls, test_size=0.3)
    Xtrain1 = torch.from_numpy(fluxTR)  # numpy 转成 torch 类型
    Xtest1 = torch.from_numpy(fluxTE)
    ytrain1 = torch.from_numpy(clsTR)
    ytest1 = torch.from_numpy(clsTE)
    torch_dataset_train = Data.TensorDataset(Xtrain1, ytrain1)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    print(clsTE.shape[0])
    data_loader_train = torch.utils.data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_test
def onehot(classes):
    ''' Encodes a list of descriptive labels as one hot vectors '''
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(classes)
    labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded) + 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    return onehot_encoded
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 40, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 60, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(60, 80, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(80),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(80, 100, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(100),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(100, 120, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(120),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(120 * 1 * 2, 1024),
        )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(1024, num_class),
            torch.nn.Softmax())

    # 前向传播
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = x6.view(-1, 120 * 1 * 2)
        x8 = self.dense1(x7)
        x9 = self.dense2(x8)
        return x9

def get_variable(x):
    x = Variable(x)
    return x

def run_CNN_module(device, num_class, num_epochs, batch_size, learning_rate, train, test):
    # 对模型进行训练和参数优化
    cnn_model = CNN_Model()
    cnn_model = cnn_model.to(device=device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(num_epochs):
        for data in train:
            optimizer.zero_grad()
            X_train, y_train = data
            X_train, y_train = get_variable(X_train), get_variable(y_train)
            optimizer.zero_grad()
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_train = X_train.to(device=device)
            y_train = y_train.to(device=device)
            outputs = cnn_model(X_train)
            _, pred = torch.max(outputs.data, 1)
            y_train = torch.max(y_train, 1)[1]
            lossTR = loss_func(outputs, y_train)
            lossTR.backward()
            optimizer.step()

        for data in test:
            cnn_model = cnn_model.eval()
            X_test, y_test = data
            X_test, y_test = get_variable(X_test), get_variable(y_test)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_test = X_test.to(device=device)
            y_test = y_test.to(device=device)
            outputs = cnn_model(X_test)
            _, pred = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            y_test = torch.max(y_test, 1)[1]

if __name__ == "__main__":
    device = "cuda:0"
    num_class = 2
    num_epochs = 2000
    batch_value = 0
    batch_size = 400
    learning_rate = 0.0001
    train, test= read_data()
    run_CNN_module(device, num_class, num_epochs, batch_size, learning_rate, train, test)




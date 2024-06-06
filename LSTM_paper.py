# pylint: disable=maybe-no-member
# from plot_model import plot_results
import torch
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def onehot(classes):
    ''' Encodes a list of descriptive labels as one hot vectors '''
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(classes)
    labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded) + 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    return onehot_encoded
def read_data():
    flux = []
    scls = []
    scls = onehot(scls)
    print(flux.shape, scls.shape)

    fluxTR, fluxTE, clsTR, clsTE = train_test_split(flux, scls, test_size=0.3)
    Xtrain1 = torch.from_numpy(fluxTR)  # numpy 转成 torch 类型
    Xtest1 = torch.from_numpy(fluxTE)
    ytrain1 = torch.from_numpy(clsTR)
    ytest1 = torch.from_numpy(clsTE)
    torch_dataset_train = Data.TensorDataset(Xtrain1, ytrain1)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    data_loader_train = torch.utils.data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_test

def get_variable(x):
    x = Variable(x)
    return x
class LSTM_Model(torch.nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=25, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm3 = torch.nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm4 = torch.nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=(1, 1), kernel_size=(1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 40, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=(1, 2), kernel_size=(1, 2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 20, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=(1, 1), kernel_size=(1, 2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 1, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=(1, 2), kernel_size=(1, 2)))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=(1, 2), kernel_size=(1, 2)))
        self.dense = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(148*9, 1024),
            torch.nn.Linear(1024, num_class),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = x.view(-1, 148, 25)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = x.view(-1,1,148,50)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.contiguous().view(-1, 148*9)
        x = self.dense(x)
        return x


def run_LSTM_module(device, num_class, num_epochs, batch_size, learning_rate, train, test):
    # 对模型进行训练和参数优化
    lstm_model = LSTM_Model().to(device)

    # cnn_model = cnn_model.to(device=device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(num_epochs):
        for data in train:
            optimizer.zero_grad()
            X_train, y_train = data
            X_train, y_train = get_variable(X_train), get_variable(y_train)
            X_train = X_train.view(-1,X_train.size(2), X_train.size(3))  # Reshape input to have 3 dimensions
            X_train = X_train.to(device=device)
            y_train = y_train.to(device=device)
            outputs = lstm_model(X_train)
            _, pred = torch.max(outputs.data, 1)
            y_train = torch.max(y_train, 1)[1]
            lossTR = loss_func(outputs, y_train)
            lossTR.backward()
            optimizer.step()
        for data in test:
            X_test, y_test = data
            X_test, y_test = get_variable(X_test), get_variable(y_test)
            X_test = X_test.view(-1, X_test.size(2),X_test.size(3))  # Reshape input to have 3 dimensions
            X_test = X_test.to(device=device)
            y_test = y_test.to(device=device)
            outputs = lstm_model(X_test)
            _, pred = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            y_test = torch.max(y_test, 1)[1]

if __name__ == "__main__":
    device = "cuda:0"
    num_class = 2
    num_epochs = 3000
    batch_size = 10
    learning_rate = 0.001
    train, test= read_data()
    run_LSTM_module(device, num_class, num_epochs, batch_size, learning_rate, train, test)
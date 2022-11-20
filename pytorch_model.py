import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
#-------------------------------------------------------------------
'''
    手写数字识别案例 https://zhuanlan.zhihu.com/p/137571225
    实现tqdm https://blog.csdn.net/wxd1233/article/details/118371404
    实现logging  https://zhuanlan.zhihu.com/p/166671955
    实现learning rate 变化
    实现模型自定义初始化参数 https://blog.csdn.net/dss_dssssd/article/details/83990511
    实现模型的参数打印 https://blog.csdn.net/weixin_45250844/article/details/103099067 https://blog.csdn.net/qq_33590958/article/details/103544175
    实现参数量的计算 https://blog.csdn.net/confusingbird/article/details/103914102
'''
#-------------------------------------------------------------------

#-----------------------init----------------------------------------
# 记录等级设置，默认为warning
#%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s 时间、用户名、记录等级、消息、函数名
logging.basicConfig(level=logging.INFO, filename='log.txt',
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# pycharm 需要的额外步骤，来让日记显示在控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(ch)
# 创建summarywriter对象，设置tensboard记录文件的地址
tb_writer = SummaryWriter(log_dir="runs/tensorboard")
#-----------------------parameters----------------------------------
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
model_parameter = 0
model_parameter_size = 0

#-----------------------model parameters----------------------------------

#-----------------------datasetloader-------------------------------------
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
#----------------------------------net-------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # 自定义初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight.item(), 1)
            #     nn.init.constant_(m.bias.item(), 0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()

if model_parameter_size:
    for param_tensor in network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        print(param_tensor, '\t', network.state_dict()[param_tensor].size())

if model_parameter:
    for param_tensor in network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        print(param_tensor, '\t', network.state_dict()[param_tensor])
# 计算总参数量
total = sum([param.nelement() for param in network.parameters()])
print("Number of parameter: %.2f" % total)

#-------------------------------optimizer----------------------------------
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
network = network.to(device)
#------------------------------visualization------------------------------------
# model structure visualization
# pictures size:(28, 28, 1) .Model need a dimension to represent batch size, so i need to add a dimension.
# I choose (1, 1, 28, 28) as input.(batchs, channels, height, width)
test_input = torch.zeros((1, 1, 28, 28), device=device)
tb_writer.add_graph(network, test_input)


#-------------------------------train-------------------------------------------
def train(epoch):
    network.train()
    learning_rate_present = optimizer.state_dict()['param_groups'][0]['lr']
    logging.info(f'learning rate={learning_rate_present}')
    # leave=False进度只显示在一行
    loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in loop:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # 设置进度条前的信息
        loop.set_description(f'Epoch [{epoch}/{n_epochs}]')
        # 设置进度条后的提示
        loop.set_postfix(loss=loss.item())
        if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

#-------------------------------test-------------------------------------------
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)

            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    logging.info('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc, test_loss


for epoch in range(1, n_epochs + 1):
    train(epoch)
    acc, loss = test()
    lr_scheduler.step()
    # tensorboard add scalar. Input do not accept tensor.
    tb_writer.add_scalar('loss', loss, epoch)
    tb_writer.add_scalar('acc', acc, epoch)

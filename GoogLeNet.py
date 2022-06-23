import torch
from torchsummary import summary

class Inception(torch.nn.Module):
    def __init__(self,in_channels=56,ch1=64,ch3_reduce=96,ch3=128,ch5_reduce=16,ch5=32,pool_proj=32):
        super(Inception, self).__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,ch1,kernel_size=1),
            torch.nn.BatchNorm1d(ch1)
        )

        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, ch3_reduce, kernel_size=1),
            torch.nn.BatchNorm1d(ch3_reduce),
            torch.nn.Conv1d(ch3_reduce, ch3, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(ch3),
        )

        self.branch5 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, ch5_reduce, kernel_size=1),
            torch.nn.BatchNorm1d(ch5_reduce),
            torch.nn.Conv1d(ch5_reduce, ch5, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(ch5),
        )

        self.branch_pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            torch.nn.Conv1d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch3(x),self.branch5(x),self.branch_pool(x)],1)



class GoogLeNet(torch.nn.Module):
    def __init__(self,in_channels=2,in_sample_points=224,classes=5):
        super(GoogLeNet, self).__init__()

        self.features=torch.nn.Sequential(
            torch.nn.Linear(in_sample_points,224),
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,padding=1),
            torch.nn.Conv1d(64,192,3,padding=1),
            torch.nn.MaxPool1d(3,2,padding=1),
            Inception(192,64,96,128,16,32,32),
            Inception(256,128,128,192,32,96,64),
            torch.nn.MaxPool1d(3,2,padding=1),
            Inception(480,192,96,208,16,48,64),
        )



        self.classifer_max_pool = torch.nn.MaxPool1d(5,3)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,1024),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512,classes),
        )

        self.Inception_4b = Inception(512,160,112,224,24,64,64)
        self.Inception_4c = Inception(512,128,128,256,24,64,64)
        self.Inception_4d = Inception(512,112,144,288,32,64,64)


        self.classifer1 = torch.nn.Sequential(
            torch.nn.Linear(2112,1056),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(1056,528),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(528,classes),
        )

        self.Inception_4e = Inception(528,256,160,320,32,128,128)
        self.max_pool = torch.nn.MaxPool1d(3,2,1)

        self.Inception_5a = Inception(832,256,160,320,32,128,128)
        self.Inception_5b = Inception(832,384,192,384,48,128,128)

        self.avg_pool = torch.nn.AvgPool1d(7,stride=1)
        self.dropout = torch.nn.Dropout(0.4)
        self.classifer2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, classes),
        )


    def forward(self,x):
        x = self.features(x)

        y = self.classifer(self.classifer_max_pool(x).view(-1,2048))

        x = self.Inception_4b(x)
        x = self.Inception_4c(x)
        x = self.Inception_4d(x)

        y1 = self.classifer1(self.classifer_max_pool(x).view(-1,2112))

        x = self.Inception_4e(x)
        x = self.max_pool(x)
        x = self.Inception_5a(x)
        x = self.Inception_5b(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(-1,1024)
        x = self.classifer2(x)

        return x,y,y1

class GoogLeNetLoss(torch.nn.Module):
    def __init__(self):
        super(GoogLeNetLoss, self).__init__()
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self,data,label):
        c2_loss = self.CrossEntropyLoss(data[0],label)
        c0_loss = self.CrossEntropyLoss(data[1],label)
        c1_loss = self.CrossEntropyLoss(data[2],label)

        loss = c2_loss + 0.3*(c0_loss+c1_loss)

        return loss




if __name__ == '__main__':
    model = GoogLeNet()
    input = torch.randn(size=(2,2,224))
    # [c2,c0,c1] = model(input)
    output = model(input)
    criterion = GoogLeNetLoss()
    label = torch.tensor([1,0])
    print(f"损失为:{criterion(output,label)}")
    print(f"输出结果为{output}")
    print(model)
    summary(model=model, input_size=(2, 224), device='cpu')

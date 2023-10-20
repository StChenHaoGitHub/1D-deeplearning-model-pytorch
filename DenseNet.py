import torch

class DenseLayer(torch.nn.Module):
    def __init__(self,in_channels,middle_channels=128,out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels,middle_channels,1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels,out_channels,3,padding=1)
        )
    def forward(self,x):
        return torch.cat([x,self.layer(x)],dim=1)


class DenseBlock(torch.nn.Sequential):
    def __init__(self,layer_num,growth_rate,in_channels,middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels+i*growth_rate,middele_channels,growth_rate)
            self.add_module('denselayer%d'%(i),layer)

class Transition(torch.nn.Sequential):
    def __init__(self,channels):
        super(Transition, self).__init__()
        self.add_module('norm',torch.nn.BatchNorm1d(channels))
        self.add_module('relu',torch.nn.ReLU(inplace=True))
        self.add_module('conv',torch.nn.Conv1d(channels,channels//2,3,padding=1))
        self.add_module('Avgpool',torch.nn.AvgPool1d(2))


class DenseNet(torch.nn.Module):
    def __init__(self,layer_num=(6,12,24,16),growth_rate=32,init_features=64,in_channels=1,middele_channels=128,classes=5):
        super(DenseNet, self).__init__()
        self.feature_channel_num=init_features
        self.conv=torch.nn.Conv1d(in_channels,self.feature_channel_num,7,2,3)
        self.norm=torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu=torch.nn.ReLU()
        self.maxpool=torch.nn.MaxPool1d(3,2,1)

        self.DenseBlock1=DenseBlock(layer_num[0],growth_rate,self.feature_channel_num,middele_channels)
        self.feature_channel_num=self.feature_channel_num+layer_num[0]*growth_rate
        self.Transition1=Transition(self.feature_channel_num)

        self.DenseBlock2=DenseBlock(layer_num[1],growth_rate,self.feature_channel_num//2,middele_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[1]*growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2],growth_rate,self.feature_channel_num//2,middele_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[2]*growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3],growth_rate,self.feature_channel_num//2,middele_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[3]*growth_rate

        self.avgpool=torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num//2, classes),

        )


    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.view(-1,self.feature_channel_num)
        x = self.classifer(x)

        return x



if __name__ == '__main__':
    input = torch.randn(size=(1,1,224))
    model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=5)
    output = model(input)
    print(output.shape)
    


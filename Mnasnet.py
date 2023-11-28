import torch

class SE_block(torch.nn.Module):
    def __init__(self,in_channel,ratio=1):
        super(SE_block, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.Hardsigmoid = torch.nn.Hardsigmoid(inplace=True)
        self.Relu = torch.nn.ReLU(inplace=True)

    def forward(self,input):
        b,c,_ = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Hardsigmoid(x)
        x = x.view([b,c,1])

        return input*x




class conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, keral,stride=1, groups=1,use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        padding = keral//2
        self.conv = torch.nn.Conv1d(in_channels, out_channels, keral, stride,padding, groups=groups)
        self.bath = torch.nn.BatchNorm1d(out_channels)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        if x.size()[-1] != 1:
            x = self.bath(x)
        if self.use_activation:
            x = self.activation(x)
        return x




class SepConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv = conv(in_channels,in_channels,3,stride,out_channels,True)
        self.conv1 = conv(in_channels,out_channels,1,use_activation=False)

    def forward(self,x):
        x = self.conv(x)
        x = self.conv1(x)
        return x




class MBConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,keral,stride,t=3,use_attention = False):
        super().__init__()
        self.use_attention = use_attention
        self.conv = conv(in_channels,in_channels*t,1)
        self.conv1 = conv(in_channels*t,in_channels*t,keral,stride=stride,groups=in_channels*t)
        self.attention = SE_block(in_channels*t)
        self.conv2 = conv(in_channels*t,out_channels,1,use_activation=False)

    def forward(self,x):
        x = self.conv(x)
        x = self.conv1(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.conv2(x)

        return x

class MnasNetA1(torch.nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()

        self.fearures = torch.nn.Sequential(
            conv(in_channels,32,3,stride=2,use_activation=False),
            SepConv(32,16,1),
            MBConv(16,16,3,2,6),
            MBConv(16,24,3,1,6),
            MBConv(24,24,5,2,3,True),
            MBConv(24,24,5,2,3,True),
            MBConv(24,40,5,2,3,True),
            MBConv(40, 40, 3, 2,6),
            MBConv(40, 40, 3, 1,6 ),
            MBConv(40, 40, 3, 1,6),
            MBConv(40, 80, 3, 1,6),
            MBConv(80, 80, 3, 1,6,True),
            MBConv(80, 112, 3, 1,6,True),
            MBConv(112, 112, 5, 2,6,True),
            MBConv(112, 112, 5, 1,6,True),
            MBConv(112, 160, 5, 1,6,True),
            MBConv(160, 160, 3, 2,6),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(160,80),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(80, classes),
        )
    def forward(self,x):
        x = self.fearures(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MnasNetA1(3,5)
    input = torch.randn(1,3,224)
    output = model(input)
    print(output.size())




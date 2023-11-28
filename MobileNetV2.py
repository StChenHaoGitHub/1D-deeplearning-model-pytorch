# 全文注释
import torch


class conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, keral,stride=1, groups=1):
        super().__init__()
        padding = 0 if keral==1  else 1

        self.conv = torch.nn.Conv1d(in_channels, out_channels, keral, stride,padding, groups=groups)
        self.bath = torch.nn.BatchNorm1d(out_channels)
        self.relu6 = torch.nn.ReLU6()



    def forward(self,x):
        x = self.conv(x)
        if x.size()[-1] != 1:
            x = self.bath(x)
        x = self.relu6(x)
        return x


class bottleneck(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride,t):
        super().__init__()
        self.conv = conv(in_channels,in_channels*t,1)
        self.conv1 = conv(in_channels*t,in_channels*t,3,stride=stride,groups=in_channels*t)
        self.conv2 = conv(in_channels*t,out_channels,1)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self,x):
        x1 = self.conv(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x1 += x

        return x1



class MobileNetV2(torch.nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()

        self.fearures = torch.nn.Sequential(
            conv(in_channels,32,keral=3,stride=2),
            bottleneck(32,16,stride=1,t=1),

            bottleneck(16,24,stride=2,t=6),
            bottleneck(24,24,stride=1,t=6),

            bottleneck(24, 32,stride=2, t=6),
            bottleneck(32, 32,stride=1, t=6),
            bottleneck(32, 32,stride=1, t=6),

            bottleneck(32, 64,stride=2, t=6),
            bottleneck(64, 64,stride=1, t=6),
            bottleneck(64, 64,stride=1, t=6),
            bottleneck(64, 64,stride=1, t=6),

            bottleneck(64, 96,stride=1, t=6),
            bottleneck(96, 96,stride=1, t=6),
            bottleneck(96, 96,stride=1, t=6),

            bottleneck(96, 160,stride=2, t=6),
            bottleneck(160, 160,stride=1, t=6),
            bottleneck(160, 160,stride=1, t=6),

            bottleneck(160, 320,stride=1, t=6),
            conv(320,1280,1,stride=1),
            torch.nn.AdaptiveAvgPool1d(1)

        )

        self.classifier = torch.nn.Sequential(
            conv(1280,out_channels=classes,keral=1),
            torch.nn.Flatten()

        )

    def forward(self,x):
        x = self.fearures(x)
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    # model = conv(3,20,1)
    # model = bottleneck(32,32,1)
    model = MobileNetV2(32,5)
    input = torch.randn((1,32,224))
    output = model(input)
    print(output.size())

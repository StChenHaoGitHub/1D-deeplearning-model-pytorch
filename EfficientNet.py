import torch
from thop import profile

class SEModule(torch.nn.Module):
    def __init__(self,in_channel,ratio=4):
        super(SEModule, self).__init__()
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


class MBConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=4):
        super(MBConvBlock, self).__init__()
        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(expanded_channels)
        # Depthwise convolution
        self.depthwise_conv = torch.nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(expanded_channels)
        # Squeeze and Excitation (SE) phase
        self.se = SEModule(expanded_channels, se_ratio)
        # Linear Bottleneck
        self.linear_bottleneck = torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        # Skip connection if input and output channels are the same and stride is 1
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.02)

    def forward(self, x):
        identity = x
        # Expansion phase
        x = self.leakyrelu(self.bn1(self.expand_conv(x)))
        # Depthwise convolution phase
        x = self.leakyrelu(self.bn2(self.depthwise_conv(x)))
        # Squeeze and Excitation phase
        x = self.se(x)
        # Linear Bottleneck phase
        x = self.bn3(self.linear_bottleneck(x))

        # Skip connection
        if self.use_skip_connection:
            x = identity + x

        return x


class EfficientNetB0(torch.nn.Module):
    def __init__(self, in_channels=3,classes=1000):
        super(EfficientNetB0, self).__init__()

        # Initial stem convolution
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.02)
        )

        # Building blocks
        self.blocks = torch.nn.Sequential(
            MBConvBlock(32, 16, 1, 3, 1),

            MBConvBlock(16, 24, 6, 3, 2),
            MBConvBlock(24, 24, 6, 3, 1),

            MBConvBlock(24, 40, 6, 5, 2),
            MBConvBlock(40, 40, 6, 5, 1),

            MBConvBlock(40, 80, 6, 3, 2),
            MBConvBlock(80, 80, 6, 3, 1),
            MBConvBlock(80, 80, 6, 3, 1),


            MBConvBlock(80, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),

            MBConvBlock(112, 192, 6, 5, 2),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 192, 6, 5, 1),

            MBConvBlock(192, 320, 6, 3, 1),
        )

        # Head
        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(1280),
            torch.nn.LeakyReLU(0.02)
        )

        # Global average pooling and classifier
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(1280, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # input = torch.randn((1,1,224))
    model = EfficientNetB0(2,200)

    input = torch.randn(1, 2, 200)
    flops, params = profile(model, inputs=(input,))

    print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

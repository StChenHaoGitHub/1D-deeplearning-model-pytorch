import torch


class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv1d, self).__init__()

        # 深度卷积
        self.depthwise = torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # 逐点卷积
        self.pointwise = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Entry(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.beforeresidual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,32,3,2,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 3, 2, 1),
            torch.nn.ReLU()
        )

        self.residual_branch1 = torch.nn.Conv1d(64, 128, 1, 2)
        self.residual_model1 = torch.nn.Sequential(
            SeparableConv1d(64,128,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(128, 128, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )

        self.residual_branch2 = torch.nn.Conv1d(256, 256, 1, 2)
        self.residual_model2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(256,256,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(256, 256, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )

        self.residual_branch3 = torch.nn.Conv1d(512, 728, 1, 2)
        self.residual_model3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(512,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )


    def forward(self,x):
        x = self.beforeresidual(x)

        x1 = self.residual_branch1(x)
        x = self.residual_model1(x)
        x = torch.cat([x,x1],dim=1)

        x1 = self.residual_branch2(x)
        x = self.residual_model2(x)
        x = torch.cat([x,x1],dim=1)

        x1 = self.residual_branch3(x)
        x = self.residual_model3(x)
        # x = torch.cat([x,x1],dim=1)
        x = x+x1

        return x


class Middleflow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
        )

    def forward(self,x):
        return x + self.layers(x)


class Exitflow(torch.nn.Module):
    def __init__(self,classes):
        super().__init__()

        self.residual = torch.nn.Conv1d(728,1024,1,2)
        self.residual_model = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 1024, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )
        self.last_layer = torch.nn.Sequential(
            SeparableConv1d(1024,1536,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(1536, 2048, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,classes)
        )


    def forward(self,x):
        x = self.residual_model(x) + self.residual(x)
        x = self.last_layer(x)

        return x



class Xception(torch.nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Entry(in_channels),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Exitflow(classes)
        )
    def forward(self,x):
        return self.layers(x)



if __name__ == '__main__':
    input = torch.randn((1,3,224))
    # model = SeparableConv1d(12,12,3,1,1)
    # model = Entry(12)
    # model = Middleflow()
    model = Xception(3,5)
    output = model(input)
    print(output.size())


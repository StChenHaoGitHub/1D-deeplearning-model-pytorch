import torch

class channel_shuffle(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, l = x.size()
        group_channel = c // self.groups
        x = x.reshape(b, self.groups, group_channel, l)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(b, c, l)
        return x

class shuffuleBlock(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, stride=2, group=3):
        super(shuffuleBlock, self).__init__()
        self.stride = stride  # Added to store the stride value

        if stride == 2:
            self.res_layer = torch.nn.AvgPool1d(3, stride, 1)

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            channel_shuffle(groups=group),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, stride, padding=1, groups=group),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        if self.stride == 2:
            return torch.cat((self.res_layer(x), self.layer(x)), 1)
        else:
            return self.layer(x)



class shufuleNetV1_G3(torch.nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,24,3,2,1),
            torch.nn.Conv1d(24,120,3,2,1),
            torch.nn.MaxPool1d(3,2,1),
            #
            shuffuleBlock(120,120//4,120,2,3),
            #
            shuffuleBlock(240,240//4,240,1,3),
            shuffuleBlock(240,240//4,240,1,3),
            shuffuleBlock(240,240//4,240,1,3),
            #
            shuffuleBlock(240, 240 // 4, 240, 2,3),
            #
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),
            shuffuleBlock(480, 480 // 4, 480, 1,3),

            shuffuleBlock(480, 480 // 4, 480, 2,3),

            shuffuleBlock(960, 960 // 4, 960, 1,3),
            shuffuleBlock(960, 960 // 4, 960, 1,3),
            shuffuleBlock(960, 960 // 4, 960, 1,3),

            torch.nn.AdaptiveAvgPool1d(1)

        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(960,classes)
        )


    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
if __name__ == "__main__":
    x = torch.randn(1, 2, 200)
    # model = shuffuleBlock(300, 300 // 4, 300, 2, 3)
    model = shufuleNetV1_G3(2, 125)
    output = model(x)
    print(output.size())

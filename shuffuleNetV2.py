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

class shuffuleV2Block(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, stride=2):
        super(shuffuleV2Block, self).__init__()
        self.stride = stride  # Added to store the stride value
        self.In_channel = In_channel
        self.Out_channel = Out_channel

        if self.stride == 2:
            self.left = torch.nn.Sequential(
                torch.nn.Conv1d(self.In_channel, self.In_channel, 3, self.stride, padding=1, groups=self.In_channel),
                torch.nn.BatchNorm1d(self.In_channel),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.In_channel, Out_channel, 1),
                torch.nn.BatchNorm1d(Out_channel),
                torch.nn.ReLU(),
            )
        else:
            self.In_channel = self.In_channel//2
            self.Out_channel = self.Out_channel//2

        self.right = torch.nn.Sequential(
            torch.nn.Conv1d(self.In_channel, Med_channel, 1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, self.stride, padding=1, groups=Med_channel),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, self.Out_channel, 1),
            torch.nn.BatchNorm1d(self.Out_channel),
            torch.nn.ReLU(),
        )
        self.shuffule = channel_shuffle(2)

    def forward(self, x):
        if self.stride == 2:
            xl = self.left(x)
            xr = self.right(x)
            x_out = torch.cat((xl, xr), 1)

        else:
            xl,xr = x.chunk(2,dim=1)
            xr = self.right(xr)
            x_out = torch.cat((xl, xr), 1)

        return self.shuffule(x_out)




class shufuleNetV2(torch.nn.Module):
    def __init__(self,in_channels = 2, classes = 125):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,24,3,2,1),
            torch.nn.MaxPool1d(3,2,1),
            shuffuleV2Block(24,24,58,2),

            shuffuleV2Block(116,116//4,116,1),
            shuffuleV2Block(116,116//4,116,1),
            shuffuleV2Block(116,116//4,116,1),

            shuffuleV2Block(116, 116, 116, 2),

            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),
            shuffuleV2Block(232, 232//4, 232, 1),

            shuffuleV2Block(232, 232, 232, 2),
            shuffuleV2Block(464, 464//4, 464, 1),
            shuffuleV2Block(464, 464//4, 464, 1),
            shuffuleV2Block(464, 464//4, 464, 1),

            torch.nn.Conv1d(464,1024,1),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024,classes)

        )

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
if __name__ == "__main__":
    x = torch.randn(1, 2, 200)
    # model = shuffuleV2Block(300, 300 // 4, 300, 1)
    model = shufuleNetV2(2, 125)
    output = model(x)
    print(output.size())

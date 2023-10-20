import torch

class AlexNet(torch.nn.Module):
   def __init__(self,in_channels,classes,input_sample_points):
       super(AlexNet, self).__init__()

       self.input_channels = in_channels
       self.input_sample_points = input_sample_points

       self.features = torch.nn.Sequential(

           torch.nn.Conv1d(in_channels,64,kernel_size=11,stride=4,padding=2),
           torch.nn.BatchNorm1d(64),
           torch.nn.ReLU(inplace=True),
           #torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
           torch.nn.MaxPool1d(kernel_size=3,stride=2),

           torch.nn.Conv1d(64, 192, kernel_size=5, padding=2),
           torch.nn.BatchNorm1d(192),
           torch.nn.ReLU(inplace=True),
           #torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
           torch.nn.MaxPool1d(kernel_size=3, stride=2),

           torch.nn.Conv1d(192, 384, kernel_size=3, padding=1),
           torch.nn.BatchNorm1d(384),
           torch.nn.ReLU(inplace=True),
           torch.nn.Conv1d(384, 256, kernel_size=3, padding=1),
           torch.nn.ReLU(inplace=True),
           torch.nn.BatchNorm1d(256),
           torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
           torch.nn.BatchNorm1d(256),
           torch.nn.ReLU(inplace=True),
           torch.nn.MaxPool1d(kernel_size=3, stride=2),
   		#自适应平均池化不管输入多少输出一定为6
           torch.nn.AdaptiveAvgPool1d(6),
       )

       self.classifier = torch.nn.Sequential(

           torch.nn.Dropout(0.5),
           torch.nn.Linear(1536,1024),
           torch.nn.ReLU(inplace=True),

           torch.nn.Dropout(0.5),
           torch.nn.Linear(1024, 1024),
           torch.nn.ReLU(inplace=True),
           torch.nn.Linear(1024,classes),

       )

   def forward(self,x):

       x = self.features(x)
       x = x.view(-1,1536)
       x = self.classifier(x)
       return x


if __name__ == '__main__':
   model = AlexNet(in_channels=1, input_sample_points=224, classes=5)
   input = torch.randn(size=(1, 1, 224))
   output = model(input)
   print(output.shape)

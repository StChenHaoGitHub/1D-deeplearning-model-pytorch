import torch

class ZFNet(torch.nn.Module):
   def __init__(self,input_channels,input_sample_points,classes):
       super(ZFNet, self).__init__()

       self.input_channels = input_channels
       self.input_sample_points = input_sample_points

       self.features = torch.nn.Sequential(
           torch.nn.Conv1d(input_channels,96,kernel_size=7,stride=2),
           torch.nn.BatchNorm1d(96),
           torch.nn.MaxPool1d(kernel_size=3,stride=2),
           torch.nn.Conv1d(96, 256, kernel_size=5, stride=2),
           torch.nn.BatchNorm1d(256),
           torch.nn.MaxPool1d(kernel_size=3, stride=2),

           torch.nn.Conv1d(256, 384, kernel_size=3, padding=1),
           torch.nn.BatchNorm1d(384),
           torch.nn.Conv1d(384, 384, kernel_size=3, padding=1),
           torch.nn.BatchNorm1d(384),
           torch.nn.Conv1d(384, 256, kernel_size=3, padding=1),
           torch.nn.BatchNorm1d(256),
           torch.nn.MaxPool1d(kernel_size=3, stride=2),
       )

       self.After_features_channels = 256
       self.After_features_sample_points = (((((((((input_sample_points-7)//2 + 1)-3)//2+1)-5)//2+1)-3)//2+1)-3)//2+1
       self.classifier = torch.nn.Sequential(

           torch.nn.Linear(self.After_features_channels*self.After_features_sample_points,1024),
           torch.nn.ReLU(inplace=True),
           torch.nn.Dropout(0.5),

           torch.nn.Linear(1024, 1024),
           torch.nn.ReLU(inplace=True),
           torch.nn.Dropout(0.5),

           torch.nn.Linear(1024,classes),
       )

   def forward(self,x):
       if x.size(1)!=self.input_channels or x.size(2)!=self.input_sample_points:
           raise Exception('输入数据维度错误,输入维度应为[Batch_size,{},{}],实际输入维度为{}'.format(self.input_channels,self.input_sample_points,x.size()))

       x = self.features(x)
       x = x.view(-1,self.After_features_channels*self.After_features_sample_points)
       x = self.classifier(x)
       return x


if __name__ == '__main__':
   model = ZFNet(input_channels=1, input_sample_points=224, classes=5)
   input = torch.randn(size=(1, 1, 224))
   output = model(input)
   print(output.shape)

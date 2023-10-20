import torch

class LeNet(torch.nn.Module):
   def __init__(self, input_channels, input_sample_points, classes):
       super(LeNet, self).__init__()

       self.input_channels = input_channels
       self.input_sample_points = input_sample_points

       self.features = torch.nn.Sequential(
           torch.nn.Conv1d(input_channels, 20, kernel_size=5),
           torch.nn.BatchNorm1d(20),
           torch.nn.MaxPool1d(2),
           torch.nn.Conv1d(20, 50, kernel_size=5),
           torch.nn.BatchNorm1d(50),
           torch.nn.MaxPool1d(2),
       )

       self.After_features_channels = 50
       # 根据公式计算出通过所有的卷积层和池化层后输出的通道中样本点的数量
       # self.After_features_sample_points = ((input_sample_points - 5 + 1) // 2 - 5 + 1) // 2
       self.After_features_sample_points = ((input_sample_points-4)//2-4) // 2


       self.classifier = torch.nn.Sequential(
           torch.nn.Linear(self.After_features_channels * self.After_features_sample_points, 512),
           torch.nn.ReLU(),
           torch.nn.Linear(512, classes),
           torch.nn.ReLU()
       )

   def forward(self, x):
       # check the dimensionality 
       # 检查输入样本维度是否有错误
       if x.size(1) != self.input_channels or x.size(2) != self.input_sample_points:
           raise Exception(
               'Input dimensionality is wrong,Input dimensionality should be [Batch_size,{},{}],Actually is {}'.format(self.input_channels, self.input_sample_points,x.size()))

       x = self.features(x)
       x = x.view(-1, self.After_features_channels * self.After_features_sample_points)
       x = self.classifier(x)
       return x


if __name__ == '__main__':
   model = LeNet(input_channels=1, input_sample_points=224, classes=5)
   input = torch.randn(size=(1, 1, 224))
   output = model(input)
   print(output.shape)


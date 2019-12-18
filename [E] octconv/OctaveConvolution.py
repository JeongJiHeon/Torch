class OctConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, a_in = 0.25, a_out = 0.25, stride = 1, padding = 0, bias = True, dilation = 1, groups = 1, padding_mode = 'zero', act = 'relu', norm = 'batch'):
        super(OctConv2d, self).__init__()

        
        assert 0 <= a_in <= 1, ' a_in must be in [0,1]'
        assert 0 <= a_out <=1, 'a_out must be in [0,1]'
        
        assert in_channels * a_in - int(in_channels * a_in) == 0, 'not int'
        assert out_channels * a_out - int(out_channels * a_out) == 0, 'not int'
        
        
        self.in_channels_H = int(in_channels * (1-a_in))
        self.out_channels_H = int(out_channels * (1-a_out))
        
        self.a_in = a_in
        self.a_out = a_out


        HH, HL, LH, LL = None, None, None, None
        if a_in == 0:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                           stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            HL = nn.Conv2d(self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size, 
                           stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
        
        elif a_out == 0:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                            stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LH = nn.Conv2d(in_channels - self.in_channels_H, self.out_channels_H, kernel_size = kernel_size,
                            stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
        elif a_in == 0 and a_out == 0:
            HH = nn.Conv2d(self.in_channels_H, out_channels_H, kernel_size = kernel_size, 
                            stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)



        else:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                          stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            HL = nn.Conv2d(self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size, 
                          stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LL = nn.Conv2d(in_channels - self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size,
                          stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LH = nn.Conv2d(in_channels - self.in_channels_H, self.out_channels_H, kernel_size = kernel_size,
                          stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
        self.W = [HH, HL, LH, LL]
        if act == 'relu':
            self.Activation = nn.ReLU()
        elif act == 'sigmoid':
            self.Activation = nn.Sigmoid()
        elif act == 'leakyrelu':
            self.Activation = nn.LeakyReLU()
            
            
        if norm == 'batchnorm':
            self.Normalization = nn.BatchNorm2d
        if norm == 'instance':
            self.Normalization = nn.InstanceNorm2d
        
        
    def forward(self, inputs):
        X_H , X_L = (inputs[:, :self.in_channels_H], inputs[:, self.in_channels_H:]) if type(inputs) != tuple else inputs
        X = [X_H.to(device), F.max_pool2d(X_H,2).to(device), X_L.to(device), X_L.to(device)]

        Y = []
        for num, weight in enumerate(self.W):
            Y.append(0 if type(weight) == type(None) else weight(X[num]))

        if type(Y[2]) != int:
            Y[2] = nn.Upsample(scale_factor=2, mode = 'nearest')(Y[2])
            
        Y_H = self.Activation(Y[0] + Y[2]) if isinstance(Y[0] + Y[2], torch.Tensor) else 0
        Y_L = self.Activation(Y[1] + Y[3]) if isinstance(Y[1] + Y[3], torch.Tensor) else 0
        
        Y_H = self.Normalization(Y_H.size(1)).to(device)(Y_H) if isinstance(Y_H, torch.Tensor) else 0
        Y_L = self.Normalization(Y_L.size(1)).to(device)(Y_L) if isinstance(Y_L, torch.Tensor) else 0
        
        
        
        return (Y_H, Y_L) if self.a_out != 0 else Y_H
    
    
    
class OctCnn(nn.Module):
    def __init__(self, a = 0.5, kernel_size = 4, stride = 2, padding = 1):
        super(OctCnn, self).__init__()
        oct_model = [OctConv2d(3, 32, kernel_size = kernel_size, a_in = 0, a_out = a, stride = stride, padding = padding),
                OctConv2d(32, 64, kernel_size = kernel_size, a_in = a, a_out = a, stride = stride, padding = padding), 
                OctConv2d(64, 128, kernel_size = kernel_size, a_in = a, a_out = 0, stride = stride, padding = padding)]
        linear_model = [nn.Linear(128*4*4, 500).to(device), nn.ReLU(), nn.BatchNorm1d(500).to(device),
                       nn.Linear(500, 250).to(device), nn.ReLU(), nn.BatchNorm1d(250).to(device),
                       nn.Linear(250, 10).to(device), nn.Sigmoid(), nn.Softmax(dim = 1)]
        self.oct_model = nn.Sequential(*oct_model)
        self.linear_model = nn.Sequential(*linear_model)
        
    def forward(self, inputs):
        output =  self.oct_model(inputs)
        output = output.view(output.size(0), -1)
        return self.linear_model(output)
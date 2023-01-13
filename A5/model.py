import torch
import torch.nn as nn
import torch.functional as F
import math

def show_size(text: str, output_size, debug: bool=False):
    if debug:
        print(f"{text} size: ", output_size)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

class DecBlock(nn.Module):
    def __init__(self, input, middle, output):
        super(DecBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input, middle, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle, output, kernel_size=3, padding=1)
        self.up = Interpolate(scale_factor=2, mode='bilinear')

    def forward(self, x):
        output = self.up(x)
        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        return output

class ResBlock(nn.Module):
    def __init__(self, input: int, output: int, conv1_str: int):
        super(ResBlock, self).__init__()
        self.conv1_str = conv1_str
        self.input = input
        self.output = output
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input, output, kernel_size=3, stride=self.conv1_str, padding=1)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(input)
        self.bn2 = nn.BatchNorm2d(output)

        self.downsample = nn.Sequential(nn.Conv2d(input, output, kernel_size=1, stride=2),
                                        nn.BatchNorm2d(output))
    def forward(self, x):
        add = x
        output = self.bn1(x)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv2(output)

        if self.conv1_str != 1 or self.input != self.output:
            add = self.downsample(x)

        return output + add

class ResUNet(nn.Module):
    def __init__(self, Block, DecBlock):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Layer_2 = self.Layer(64, 64, 1, 3, Block)
        self.Layer_3 = self.Layer(64, 128, 2, 4, Block)
        self.Layer_4 = self.Layer(128, 256, 2, 6, Block)
        self.Layer_5 = self.Layer(256, 512, 2, 3, Block)
        self.center = DecBlock(512, 512, 256)
        self.Dec_Layer_5 = DecBlock(768, 512, 256)
        self.Dec_Layer_4 = DecBlock(512, 256, 128)
        self.Dec_Layer_3 = DecBlock(256, 128, 64)
        self.Dec_Layer_2 = DecBlock(128, 64, 32)
        self.Dec_Layer_1 = DecBlock(32, 32, 32)
        self.Dec_Layer_0 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, 8, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        debug = True
        # show_size("input", x.size(), debug)
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        conv1 = self.pool1(output)
        show_size("layer_1", conv1.size(), debug)
        conv2 = self.Layer_2(conv1)
        show_size("layer_2", conv2.size(), debug)
        conv3 = self.Layer_3(conv2)
        show_size("layer_3", conv3.size(), debug)
        conv4 = self.Layer_4(conv3)
        show_size("layer_4", conv4.size(), debug)
        conv5 = self.Layer_5(conv4)
        show_size("layer_5", conv5.size(), debug)
        output = self.pool2(conv5)
        show_size("pool", output.size(), debug)
        output = self.center(output)
        show_size("center", output.size(), debug)
        d_conv5 = self.Dec_Layer_5(torch.cat([output, conv5], 1))
        show_size("dec_layer_5", d_conv5.size(), debug)
        d_conv4 = self.Dec_Layer_4(torch.cat([d_conv5, conv4], 1))
        show_size("dec_layer_4", d_conv4.size(), debug)
        d_conv3 = self.Dec_Layer_3(torch.cat([d_conv4, conv3], 1))
        show_size("dec_layer_3", d_conv3.size(), debug)
        d_conv2 = self.Dec_Layer_2(torch.cat([d_conv3, conv2], 1))
        show_size("dec_layer_2", d_conv2.size(), debug)
        d_conv1 = self.Dec_Layer_1(d_conv2)
        show_size("dec_layer_1", d_conv1.size(), debug)
        d_conv0 = self.Dec_Layer_0(d_conv1)
        show_size("dec_layer_0", d_conv0.size(), debug)
        output = self.final(d_conv0)
        show_size("output", output.size(), debug)

        return output


    def Layer(self, input: int, output: int, conv1_str: int, num_layer: int, Block):
        ResBlocks = list()
        for i in range(num_layer):
            if i == 0:
                ResBlocks.append(Block(input, output, conv1_str))
            else:
                ResBlocks.append(Block(output, output, 1))

        return nn.Sequential(*ResBlocks)


if __name__=="__main__":
    input = torch.rand(4, 3, 512, 512)
    model = ResUNet(Block=ResBlock, DecBlock=DecBlock)
    model(input)
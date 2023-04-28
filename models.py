import torch

class InceptionA(torch.nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.branch_1x1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)

        self.branch_5x5_1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)
        self.branch_5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch_3x3_1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)
        self.branch_3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_ch, 24, kernel_size=1)

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)

        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch_1x1, branch_5x5, branch_3x3, branch_pool] # 16 + 24 + 24 + 24
        return torch.cat(outputs, 1)


class GoogleNet(torch.nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(10)
        self.incep2 = InceptionA(20)
        self.mp = torch.nn.MaxPool2d(2)
        # pos 0 - self
        # pos 1 - enemy
        self.fc = torch.nn.Linear(14872, 3)

    def forward(self, x):
        in_size = x.size(0)
        x = torch.nn.functional.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = torch.nn.functional.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

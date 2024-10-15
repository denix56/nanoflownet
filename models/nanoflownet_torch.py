from .layers import *

FILTER_FACTOR = 2


class NanoFlowNet(nn.Module):
    def __init__(self, activation: str = 'relu'):
        super(NanoFlowNet, self).__init__()
        self.activation = activation

        self.conv2 = nn.Sequential(
            SeparableConv2d(2, 8 * FILTER_FACTOR, 3, stride=2, padding='same', bias=True),
            act_func(activation))
        self.stage3 = StageBlock(8 * FILTER_FACTOR, 32 * FILTER_FACTOR, activation)
        self.stage3_out = FlowHead(32 * FILTER_FACTOR, 8 * FILTER_FACTOR, 1, activation, before_sigmoid=True)

        self.stage4 = StageBlock(32 * FILTER_FACTOR, 64 * FILTER_FACTOR, activation)
        self.stage5 = StageBlock(64 * FILTER_FACTOR, 128 * FILTER_FACTOR, activation)

        self.avg = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Conv2d(128 * FILTER_FACTOR, 16 * FILTER_FACTOR,
                                      kernel_size=1, stride=1, padding='same', bias=True),
                            act_func('relu'))
        self.stage5_arm = AttentionRefinementModule(128 * FILTER_FACTOR, 16 * FILTER_FACTOR, activation)
        self.stage5_arm_up = nn.Sequential(
            SeparableConv2d(16 * FILTER_FACTOR, 16 * FILTER_FACTOR, 3, stride=1, padding='same', bias=True),
            act_func(activation))
        self.stage5_out = FlowHead(16 * FILTER_FACTOR, 8 * FILTER_FACTOR, 2, activation)

        self.stage4_arm = AttentionRefinementModule(64 * FILTER_FACTOR, 16 * FILTER_FACTOR, activation)
        self.stage4_arm_up = nn.Sequential(
            SeparableConv2d(16 * FILTER_FACTOR, 16 * FILTER_FACTOR, 3, stride=1, padding='same', bias=True),
            act_func(activation))
        self.stage4_out = FlowHead(16 * FILTER_FACTOR, 8 * FILTER_FACTOR, 2, activation)

        self.ffm = FeatureFusionModule(48 * FILTER_FACTOR, 32 * FILTER_FACTOR, activation)
        self.final = FlowHead(32 * FILTER_FACTOR, 32 * FILTER_FACTOR, 2, activation)

        self.init_weights()

    def forward(self, x):
        conv2 = self.conv2(x)
        stage3 = self.stage3(conv2)
        stage3_out = self.stage3_out(stage3)
        stage3_out = F.interpolate(stage3_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)

        stage5_arm_sum = self.stage5_arm(stage5) * self.avg(stage5)
        stage5_arm_up = F.interpolate(stage5_arm_sum, size=stage4.shape[2:], mode='bilinear', align_corners=True)
        stage5_arm_up = self.stage5_arm_up(stage5_arm_up)
        stage5_out = self.stage5_out(stage5_arm_up)

        stage4_arm_sum = self.stage4_arm(stage4) + stage5_arm_up
        stage4_arm_up = F.interpolate(stage4_arm_sum, size=stage3.shape[2:], mode='bilinear', align_corners=True)
        stage4_arm_up = self.stage4_arm_up(stage4_arm_up)
        stage4_out = self.stage4_out(stage4_arm_up)

        ffm = self.ffm(stage4_arm_up, stage3)
        conv_out = self.final(ffm)

        if self.training:
            return conv_out, stage4_out, stage5_out, stage3_out
        else:
            return conv_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(
                    self.activation if not isinstance(m, Conv2dBeforeSigmoid) else 'sigmoid'))
                nn.init.constant_(m.bias, 0.0)

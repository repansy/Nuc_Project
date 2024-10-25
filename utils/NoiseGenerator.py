import torch
import torch.fft
import random
import numpy as np


class NoiseGenerator:
    def __init__(self, image_sim, noise_level):
        self.image = image_sim
        self.level = noise_level

    def gaussion_noise(self, amplitude):
        """
        info: this function gives gaussion noise mean=0,std=level/255,
        :param level: noiseL
        :return noisy_tensor: simple gaussion noise mean=0,std=level/255
        """
        image = self.image
        level = self.level
        noisy_tensor = torch.FloatTensor(image.size()).normal_(mean=level, std=amplitude/255.)
        return noisy_tensor

    def stripe_noise(self, rand, direction, dense):
        """
        info: this function now gives basic stripe nose, about size,level,direction,dense but sill have some functions have
        not been solved about dimension,channel and etc..
        :param rand: bool , stripes appear in an order such as same gap..
        :param direction: char, 'row' and 'col'
        :param dense: 0-1, and it will determine stripes in noise
        :param dimension: ?
        :param channel:   ?
        :return: noisy tensor: stripe noise
        """
        image = self.image
        level = self.level
        [batch, channel, row, col] = image.size()
        if rand:
            temp = torch.zeros_like(image)
            if direction == 'col':
                stripes = round(dense * col)
                rand = np.random.randint(0, col - 1, [stripes, 1])
                for index in rand:
                    # Attention： 同时注意这里幅值的选择'除以25'并没有原因，只是经验结果 ##
                    rand_noise = random.uniform(0, 1) / level
                    # Attetion: 针对于不同的输入图像的向量格式，应有不同的位置 ##
                    temp[:, :, :, index] = rand_noise
            elif direction == 'row':
                stripes = round(dense * row)
                rand = np.random.randint(0, row - 1, [stripes, 1])
                for index in rand:
                    rand_noise = random.uniform(0, 1) / 25
                    temp[:, :, index, :] = rand_noise
            noisy_tensor = temp
        elif not rand:
            frequency = round(dense * 10)
            amplitude = round(level * 10)
            # 获取图像的行数和列数
            noise = torch.FloatTensor(image.size()).normal_(mean=level, std=amplitude/255.)
            freq_tensor = torch.fft.fftn(noise)
            # 创建一个掩码矩阵，其中包含一些高值的峰,
            # Attention： 同时注意这里频率和幅值的选择并没有原因，只是猜测和经验结果 ##
            mask = torch.zeros_like(freq_tensor)
            mask[:, :, 0, frequency] = mask[:, :, 0, -frequency] = amplitude  # 这些峰的位置和值决定了条纹的频率和强度
            # 遍历每个像素点
            # 将掩码矩阵与频率域的张量相乘
            noisy_freq_tensor = freq_tensor * mask
            # 将噪声张量从频率域转换回空间域
            noisy_tensor = torch.fft.ifftn(noisy_freq_tensor).real  # 只取实部，忽略虚部
            # 根据正弦函数生成噪声值
            # noise[:, j] = mean + amplitude * np.sin(2 * np.pi * frequency * j)
            # 将噪声矩阵和原始图像相加，得到带有条纹噪声的图像
        else:
            # 给出一张空白原图
            noisy_tensor = torch.FloatTensor(image.size())
        return noisy_tensor

    def nuc_stripe_noise(self, rand_level, lex_order):
        # 最好做成数据集的格式方便使用
        # 将已经找好的几张做成patch随机拼接来模拟真实噪声
        # 由于图像数量的局限性，暂时只有rand_level(决定图像随机拼接的程度),是否满足环境温度/照度乱序输出
        # and what if i can use the k and bias to stimulate noise
        # and this is the litter and un-predicted additive noise
        image = self.image
        level = self.level
        noisy_tensor = torch.FloatTensor(image.size()).normal_(mean=level, std=25 / 255.)
        if lex_order:
            print(1)
        elif not lex_order:
            print(0)
        else:
            print('not right input on lex_order,please bool')
        return noisy_tensor
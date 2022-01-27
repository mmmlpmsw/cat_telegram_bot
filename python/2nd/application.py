#!/usr/bin/env python
# coding: utf-8


import torch.nn as nn
import torch.utils.data
import torchvision.transforms.functional as tvtf
from flask import Flask


nc = 3
nz = 100
ngf = 64
ngpu = 1


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


emotion_paths = {
    'worry': 'models/worry_cats_generator.pb',
    'anger': 'models/worry_cats_generator.pb',
    'hate': 'models/worry_cats_generator.pb',
    'empty': 'models/empty_cats_generator.pb',
    'neutral': 'models/neutral_cats_generator.pb',
    'relief': 'models/relief_cats_generator.pb',
    # 'love': 'models/love_cats_generator.pb',
    # 'happiness': 'models/love_cats_generator.pb',
    # 'fun': 'models/love_cats_generator.pb',
    'surprise': 'models/surprise_cats_generator.pb',
    'enthusiasm': 'models/surprise_cats_generator.pb',
    'sadness': 'models/sadness_cats_generator.pb',
    'boredom': 'models/sadness_cats_generator.pb',
}


emotion_generators = {}


def init_generators():
    for emotion, path in emotion_paths.items():
        netG = Generator(ngpu).to(device)
        netG.load_state_dict(torch.load(path))
        netG.eval()

        emotion_generators[emotion] = netG


def execute(emotion):
    netG = emotion_generators[emotion]

    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    fake = netG(fixed_noise).detach().cpu()
    fake = fake[0]

    fake_min, fake_max = fake.min(), fake.max()
    fake_minmax = fake_max - fake_min
    fake = (fake - fake_min) / fake_minmax

    return tvtf.to_pil_image(fake, mode='RGB')


init_generators()

# webapp

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    from flask import request, send_file
    from io import BytesIO

    emotion = request.args.get('emotion')
    if emotion is None or emotion not in emotion_paths:
        emotion = 'neutral'

    img = execute(emotion)

    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run()

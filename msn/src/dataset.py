import os
import torch
import numpy as np
from torchvision.transforms import v2

# -----------------
# 1. Classes de Transformação
# -----------------

class SpectralJitter(object):
    """
    Aplica uma perturbação aleatória no brilho de cada banda espectral.
    """
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def __call__(self, sample: torch.Tensor):
        device = sample.device
        add = torch.rand(sample.shape[0], device=device) * self.intensity
        mul = torch.rand(sample.shape[0], device=device) * self.intensity + 1.0

        sample = sample * mul[:, None, None] + add[:, None, None]
        return sample

class RandomSimetry(object):
    """
    Aplica rotações de 90 graus e inversões horizontais/verticais aleatórias.
    """
    def __call__(self, sample):
        # Assume input is (Bands, Height, Width)
        if torch.rand(1) > 0.5:
            sample = torch.rot90(sample, 1, [1, 2]) # Rotação no plano H/W

        if torch.rand(1) > 0.5:
            sample = torch.flip(sample, [1]) # Inversão horizontal

        if torch.rand(1) > 0.5:
            sample = torch.flip(sample, [2]) # Inversão vertical

        return sample

class Clip(object):
    """
    Limita os valores do tensor para o intervalo [0.0, 1.0].
    """
    def __call__(self, sample):
        return torch.clamp(sample, 0.0, 1.0)

# -----------------
# 2. Classe para Geração de Visualizações Múltiplas
# -----------------

class MultiViewTransform(object):
    """
    Cria múltiplas visualizações de uma única imagem, aplicando diferentes
    conjuntos de transformações.
    """
    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- Gera as visualizações aleatórias (random views)
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- Gera as visualizações focais (focal views)
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views

def make_transforms(params):
    """
    Cria as pipelines de transformação para as visualizações aleatórias e focais.
    """
    rand_transform = v2.Compose([
        v2.RandomResizedCrop(params['rand_size'], scale=params['rand_crop_scale']),
        v2.ToDtype(torch.float32, scale=True),
        SpectralJitter(intensity=params['spectral_jitter_strength']),
        RandomSimetry(),
        Clip(),
    ])
    
    focal_transform = v2.Compose([
        v2.RandomResizedCrop(params['focal_size'], scale=params['focal_crop_scale']),
        v2.ToDtype(torch.float32, scale=True),
        SpectralJitter(intensity=params['spectral_jitter_strength']),
        Clip(),
    ])
    
    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=params['rand_views'] + 1, # +1 para incluir a "target view"
        focal_views=params['focal_views']
    )
    return transform

# -----------------
# 3. Classe do Dataset (adaptada para .npy)
# -----------------

class HyperspectralImageFolder(torch.utils.data.Dataset):
    """
    Carrega imagens hiperespectrais a partir de arquivos .npy.
    """
    def __init__(self, root, transform):
        if isinstance(root, str):
            self.roots = [root]
        else:
            self.roots = list(root)
        self.transform = transform
        self.imgs = []
        for r in self.roots:
            files = [os.path.join(r, f) for f in sorted(os.listdir(r)) if f.endswith('.npy')]
            self.imgs.extend(files)

    def __getitem__(self, index):
        path = self.imgs[index]
        try:
            data = np.load(path)
            # Converte o array NumPy para um tensor PyTorch.
            img = torch.from_numpy(data.astype(np.float32))

            
            timg = self.transform(img)
            return timg
        except Exception as e:
            print(f"Erro ao carregar imagem {path}: {e}")
            return None

    def __len__(self):
        return len(self.imgs)
    
# -----------------
# 4. Inicialização dos dados
# -----------------

def init_data(params):
    """
    Função de inicialização principal. Cria o dataset e o DataLoader.
    """
    transform = make_transforms(params)
    dataset = HyperspectralImageFolder(root=params['data_root'], transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['num_workers'] > 0,
        persistent_workers=params['num_workers'] > 0,
    )
    return data_loader

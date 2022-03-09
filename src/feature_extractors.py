import sys
import torch
from torch import nn
from typing import List


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    elif model_type == 'mae':
        print("Creating MAE Feature Extractor...")
        feature_extractor = FeatureExtractorMAE(**kwargs)
    elif model_type == 'swav':
        print("Creating SwAV Feature Extractor...")
        feature_extractor = FeatureExtractorSwAV(**kwargs)
    elif model_type == 'swav_w2':
        print("Creating SwAVw2 Feature Extractor...")
        feature_extractor = FeatureExtractorSwAVw2(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        import guided_diffusion.guided_diffusion.dist_util as dist_util
        from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        self.model.to(dist_util.dev())
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorMAE(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained MAE
    '''
    def __init__(self, num_blocks=12, **kwargs):
        super().__init__(**kwargs)

        # Save features from deep encoder blocks 
        for layer in self.model.blocks[-num_blocks:]:
            layer.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer)

    def _load_pretrained_model(self, model_path, **kwargs):
        import mae
        from functools import partial
        sys.path.append(mae.__path__[0])
        from mae.models_mae import MaskedAutoencoderViT

        # Create MAE with ViT-L-8 backbone 
        model = MaskedAutoencoderViT(
            img_size=256, patch_size=8, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        self.model = model.eval().to(device)

    @torch.no_grad()
    def forward(self, x, **kwargs):
        _, _, ids_restore = self.model.forward_encoder(x, mask_ratio=0)
        ids_restore = ids_restore.unsqueeze(-1)
        sqrt_num_patches = int(self.model.patch_embed.num_patches ** 0.5)
        activations = []
        for block in self.feature_blocks:
            # remove cls token 
            a = block.activations[:, 1:]
            # unshuffle patches
            a = torch.gather(a, dim=1, index=ids_restore.repeat(1, 1, a.shape[2])) 
            # reshape to obtain spatial feature maps
            a = a.permute(0, 2, 1)
            a = a.view(*a.shape[:2], sqrt_num_patches, sqrt_num_patches)

            activations.append(a)
            block.activations = None
        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorSwAV(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained SwAVs 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layers = [self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]

        # Save features from sublayers
        for layer in layers:
            for l in layer[::2]:
                l.register_forward_hook(self.save_hook)
                self.feature_blocks.append(l)

    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50

        model = resnet50(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False) 
        self.model = model.module.eval()

    @torch.no_grad()
    def forward(self, x, **kwargs):
        self.model(x)

        activations = []
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations
    

class FeatureExtractorSwAVw2(FeatureExtractorSwAV):
    ''' 
    Wrapper to extract features from twice wider pretrained SwAVs 
    '''
    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50w2

        model = resnet50w2(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False) 
        self.model = model.module.eval()


def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])
    
    return torch.cat(resized_activations, dim=0)

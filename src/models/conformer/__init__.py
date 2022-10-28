
from .conformer import Conformer

def Conformer_tiny_patch16(args):
        

    model = Conformer(patch_size=16, 
                    channel_ratio=1,
                    embed_dim=384,
                    depth=12,
                    num_heads=6, 
                    mlp_ratio=4, 
                    qkv_bias=True,
                    num_classes=args.num_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path)
    return model

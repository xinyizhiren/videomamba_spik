from models.videomamba import VisionMamba


class CleanVideoMamba(VisionMamba):
    def forward_single(self, video):
        features = self.forward_features(video)
        return self.head(self.head_drop(features))

    def forward(self, view1, view2=None, return_view_logits=False):
        view1_logits = self.forward_single(view1)
        if view2 is None:
            return view1_logits

        view2_logits = self.forward_single(view2)
        fused_logits = 0.5 * (view1_logits + view2_logits)
        if return_view_logits:
            return fused_logits, view1_logits, view2_logits
        return fused_logits


def create_videomamba_small_clean(
        num_classes,
        img_size=224,
        num_frames=16,
        tubelet_size=1,
        drop_path=0.0,
        fc_drop_rate=0.0,
        use_mean_pooling=True):
    return CleanVideoMamba(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        kernel_size=tubelet_size,
        num_frames=num_frames,
        num_classes=num_classes,
        drop_path_rate=drop_path,
        fc_drop_rate=fc_drop_rate,
        use_mean_pooling=use_mean_pooling,
    )

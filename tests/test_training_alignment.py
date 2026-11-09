"""Integration checks for architecture, checkpoint transfer and paper training settings."""
import argparse
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import timm
from torch.utils.data import DataLoader, TensorDataset

from dataset_mae_cnn import get_data
from finetune_training import add_training_arguments, build_optimizer, fit_downstream, task_loss
from load_model import load_pretrained_encoder, load_pretrained_model
from model.swin import VisionUlt, build_encoder
from sononexus_downstream import (
    LoRAQKV, SonoNexusLoRAClassifier, SonoNexusLoRAUNetSegmenter, load_frozen_lora_backbone,
)
from training_utils import warmup_cosine_scheduler


class TrainingAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        # A 16px fixture keeps backward/serialization tests inexpensive;
        # stage depths and widths remain the full pretraining architecture.
        cls.pretrained = VisionUlt(img_size=16).eval()
        cls.checkpoint = cls.root / 'pretrained.pth'
        torch.save({'state_dict': {'module.' + key: value
                                  for key, value in cls.pretrained.state_dict().items()}}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        del cls.pretrained
        cls.directory.cleanup()
        gc.collect()

    def test_restored_constructor_matches_original_at_224(self):
        # The original main-model constructor independently checks parameters,
        # grids, masks and shifts; the deeper server model is an ablation.
        torch.manual_seed(137)
        original = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k', pretrained=False, features_only=True,
        ).eval()
        original.patch_embed.proj = torch.nn.Conv2d(3, 128, kernel_size=2, stride=2)
        torch.manual_seed(137)
        restored = build_encoder().eval()
        self.assertEqual(restored.patch_embed.patch_size, (4, 4))
        self.assertEqual(restored.patch_embed.grid_size, (56, 56))
        self.assertEqual(restored.patch_embed.proj.kernel_size, (2, 2))
        self.assertEqual(set(original.state_dict()), set(restored.state_dict()))
        for key, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, original.state_dict()[key]), key)
        original_modules = dict(original.named_modules())
        for name, module in restored.named_modules():
            if hasattr(module, 'shift_size'):
                self.assertEqual(module.input_resolution, original_modules[name].input_resolution)
                self.assertEqual(module.shift_size, original_modules[name].shift_size)
                expected = original_modules[name].attn_mask
                self.assertTrue(module.attn_mask is None if expected is None
                                else torch.equal(module.attn_mask, expected))
        self.assertEqual(restored.get_submodule('layers_3.blocks.1').shift_size, (0, 0))
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected_features, actual_features = original(x), restored(x)
        self.assertEqual([tuple(f.shape) for f in actual_features], [
            (1, 112, 112, 128), (1, 56, 56, 256), (1, 28, 28, 512), (1, 14, 14, 1024),
        ])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_features, actual_features)))

    def test_pretraining_two_views_backward(self):
        x, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
        mask = torch.zeros(2, 1, 16, 16)
        mask[:, :, :8] = 1
        recon, masked, visible, contrast = self.pretrained(x, mask, target)
        self.assertEqual(recon.shape, target.shape)
        self.assertTrue(torch.isfinite(torch.stack((masked, visible, contrast))).all())
        (masked + visible + contrast).backward()
        self.assertGreater(self.pretrained.decoder_fc.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.pretrained.model.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.pretrained.zero_grad(set_to_none=True)

    def test_full_checkpoint_and_encoder_only_roundtrip(self):
        encoder = load_pretrained_encoder(self.checkpoint, img_size=16)
        expected = self.pretrained.model.state_dict()
        self.assertEqual(set(encoder.state_dict()), set(expected))
        self.assertFalse(any('decoder' in name for name, _ in encoder.named_modules()))
        for key, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)
        raw = self.root / 'encoder.pth'
        torch.save(encoder.state_dict(), raw)
        del encoder
        encoder = load_pretrained_encoder(raw, img_size=16)
        self.assertTrue(torch.equal(encoder.patch_embed.proj.weight, expected['patch_embed.proj.weight']))
        del encoder
        inference = load_pretrained_model(self.checkpoint, device='cpu', img_size=16)
        x, mask = torch.randn(1, 3, 16, 16), torch.zeros(1, 1, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(inference(x, mask), self.pretrained.reconstruct(x, mask)))
        del inference

    def test_incomplete_encoder_is_rejected(self):
        partial = dict(self.pretrained.model.state_dict())
        partial.pop('patch_embed.proj.weight')
        path = self.root / 'incomplete.pth'
        torch.save(partial, path)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            load_pretrained_encoder(path, img_size=16)

    def test_lora_preserves_key_slice(self):
        adapter = LoRAQKV(torch.nn.Linear(16, 48), rank=8, alpha=16)
        x = torch.randn(2, 5, 16)
        before = adapter.base(x).detach()
        self.assertTrue(torch.equal(adapter(x), before))
        optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=0.01)
        adapter(x).square().mean().backward()
        optimizer.step()
        after = adapter(x).detach()
        self.assertTrue(torch.equal(after[..., 16:32], before[..., 16:32]))
        self.assertFalse(torch.equal(after[..., :16], before[..., :16]))
        self.assertFalse(torch.equal(after[..., 32:], before[..., 32:]))
        self.assertIsNone(adapter.base.weight.grad)

    def test_real_encoder_transfer_all_downstream_heads(self):
        encoder = load_frozen_lora_backbone(self.checkpoint, torch.device('cpu'), img_size=16)
        self.assertEqual(sum(isinstance(module, LoRAQKV) for module in encoder.modules()), 24)
        self.assertTrue(all('lora_' in name for name, param in encoder.named_parameters() if param.requires_grad))
        self.assertFalse(any('decoder' in name for name, _ in encoder.named_modules()))
        x = torch.randn(2, 3, 16, 16)
        classifier = SonoNexusLoRAClassifier(encoder, num_outputs=3)
        logits = classifier(x)
        task_loss(logits, torch.tensor([0, 2]), 'classification').backward()
        self.assertTrue(any(param.grad is not None and param.grad.abs().sum() > 0
                            for name, param in encoder.named_parameters() if 'lora_b' in name))
        classifier.zero_grad(set_to_none=True)
        diagnosis = SonoNexusLoRAClassifier(encoder, num_outputs=2)
        self.assertEqual(diagnosis(x).shape, (2, 2))
        segmenter = SonoNexusLoRAUNetSegmenter(encoder)
        logits = segmenter(x)
        self.assertEqual(logits.shape, (2, 1, 16, 16))
        task_loss(logits, torch.randint(0, 2, logits.shape).float(), 'segmentation').backward()
        self.assertGreater(segmenter.seg_head.weight.grad.abs().sum().item(), 0)
        del encoder, classifier, diagnosis, segmenter, logits
        gc.collect()

    def test_data_two_views_masks_and_paper_augmentations(self):
        folder = self.root / 'images'
        folder.mkdir(exist_ok=True)
        for index in range(2):
            rng = np.random.default_rng(index)
            Image.fromarray(rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)).save(folder / f'{index}.png')
        loader = get_data(str(folder), batch_size=2, num_workers=0, img_size=32)
        views, masks, ratios = next(iter(loader))
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0].shape, (2, 3, 32, 32))
        self.assertFalse(torch.equal(views[0], views[1]))
        self.assertTrue(((ratios >= 0.1) & (ratios <= 0.8)).all())
        self.assertEqual(masks.shape, (2, 1, 32, 32))
        self.assertTrue(torch.equal(masks, masks[:, :, ::8, ::8].repeat_interleave(8, 2).repeat_interleave(8, 3)))
        first = loader.dataset.transform.transforms
        second = loader.dataset.second_transform.transforms
        self.assertEqual(first[0].scale, (0.2, 1.0))
        self.assertEqual((first[4].p, second[4].p), (0.5, 0.1))

    def test_optimizer_groups_warmup_and_early_stopping(self):
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, str(self.root / 'best.pt'))
        args = parser.parse_args([])
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 3)
                self.head = torch.nn.Linear(3, 2)
            def forward(self, x):
                return self.head(self.encoder(x))
        model = TinyModel()
        optimizer = build_optimizer(model, args)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [2e-4, 1e-3])
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)
        scheduler = warmup_cosine_scheduler(optimizer, epochs=50, warmup_epochs=5)
        rates = []
        for epoch in range(50):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], 2e-4 / 5)
        self.assertAlmostEqual(rates[4], 2e-4)
        self.assertGreater(rates[6], rates[-1])
        loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
        args.epochs, args.patience = 5, 2
        scores = iter([0.8, 0.7, 0.6])
        with patch('finetune_training.validation_auc', side_effect=lambda *_: next(scores)):
            fit_downstream(model, loader, loader, torch.device('cpu'), args, 'classification')
        saved = torch.load(args.output, weights_only=True)
        self.assertEqual(saved['epoch'], 1)
        self.assertEqual(saved['validation_auc'], 0.8)


if __name__ == '__main__':
    unittest.main()

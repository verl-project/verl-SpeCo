# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted original numerical validation; no external source imports.
import unittest

import torch

from verl_speco.ops.deepseek_v4_hc import (
    hc_post_explicit_backward_chunk,
    hc_pre_explicit_backward_chunk,
)


def hc_pre_reference(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    norm_eps,
    hc_eps,
    sinkhorn_iters,
):
    x_float = x.float()
    x_flat = x_float.flatten(start_dim=-2)
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    logits = torch.nn.functional.linear(x_flat, hc_fn.float()) * rsqrt
    pre_logits = logits[:, :hc_mult]
    post_logits = logits[:, hc_mult : 2 * hc_mult]
    comb_logits = logits[:, 2 * hc_mult :]
    pre = (
        torch.sigmoid(hc_scale[0].float() * pre_logits + hc_base[:hc_mult].float())
        + hc_eps
    )
    post = 2 * torch.sigmoid(
        hc_scale[1].float() * post_logits + hc_base[hc_mult : 2 * hc_mult].float()
    )
    comb = (hc_scale[2].float() * comb_logits + hc_base[2 * hc_mult :].float()).reshape(
        -1, hc_mult, hc_mult
    )
    comb = torch.softmax(comb, dim=-1) + hc_eps
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * x_float).sum(dim=-2)
    return collapsed.to(x.dtype), post, comb


class HCExplicitBackwardTests(unittest.TestCase):
    def assert_explicit_matches_autograd(self, x_dtype):
        generator = torch.Generator().manual_seed(314159)
        tokens, hc_mult, hidden = 5, 3, 7
        output_features = 2 * hc_mult + hc_mult * hc_mult
        x = torch.randn(
            tokens,
            hc_mult,
            hidden,
            generator=generator,
        ).to(x_dtype)
        hc_fn = torch.randn(
            output_features,
            hc_mult * hidden,
            generator=generator,
        )
        hc_scale = torch.randn(3, generator=generator)
        hc_base = torch.randn(output_features, generator=generator)
        grad_outputs = (
            torch.randn(tokens, hidden, generator=generator).to(x_dtype),
            torch.randn(tokens, hc_mult, generator=generator),
            torch.randn(tokens, hc_mult, hc_mult, generator=generator),
        )
        reference_inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (x, hc_fn, hc_scale, hc_base)
        )
        reference_outputs = hc_pre_reference(
            *reference_inputs,
            hc_mult,
            1e-6,
            1e-6,
            3,
        )
        expected = torch.autograd.grad(
            reference_outputs,
            reference_inputs,
            grad_outputs,
        )
        actual = hc_pre_explicit_backward_chunk(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            *grad_outputs,
            hc_mult,
            1e-6,
            1e-6,
            3,
        )
        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=2e-5,
                atol=2e-5,
            )

    def test_float32_matches_autograd(self):
        self.assert_explicit_matches_autograd(torch.float32)

    def test_bfloat16_input_matches_autograd(self):
        self.assert_explicit_matches_autograd(torch.bfloat16)

    def test_pre_backward_skips_unrequested_gradients(self):
        generator = torch.Generator().manual_seed(2718)
        tokens, hc_mult, hidden = 3, 2, 5
        output_features = 2 * hc_mult + hc_mult * hc_mult
        values = (
            torch.randn(tokens, hc_mult, hidden, generator=generator),
            torch.randn(output_features, hc_mult * hidden, generator=generator),
            torch.randn(3, generator=generator),
            torch.randn(output_features, generator=generator),
        )
        grad_outputs = (
            torch.randn(tokens, hidden, generator=generator),
            torch.randn(tokens, hc_mult, generator=generator),
            torch.randn(tokens, hc_mult, hc_mult, generator=generator),
        )
        reference_inputs = tuple(
            value.detach().clone().requires_grad_(True) for value in values
        )
        expected_scale = torch.autograd.grad(
            hc_pre_reference(
                *reference_inputs,
                hc_mult,
                1e-6,
                1e-6,
                2,
            ),
            reference_inputs[2],
            grad_outputs,
        )[0]

        actual = hc_pre_explicit_backward_chunk(
            *values,
            *grad_outputs,
            hc_mult,
            1e-6,
            1e-6,
            2,
            need_x=False,
            need_hc_fn=False,
            need_hc_scale=True,
            need_hc_base=False,
        )

        self.assertIsNone(actual[0])
        self.assertIsNone(actual[1])
        torch.testing.assert_close(actual[2], expected_scale)
        self.assertIsNone(actual[3])

    def test_post_backward_matches_autograd_and_skips_inputs(self):
        generator = torch.Generator().manual_seed(1618)
        tokens, hc_mult, hidden = 4, 3, 5
        values = (
            torch.randn(tokens, hidden, generator=generator),
            torch.randn(tokens, hc_mult, hidden, generator=generator),
            torch.randn(tokens, hc_mult, generator=generator),
            torch.randn(tokens, hc_mult, hc_mult, generator=generator),
        )
        grad_output = torch.randn(
            tokens,
            hc_mult,
            hidden,
            generator=generator,
        )
        reference_inputs = tuple(
            value.detach().clone().requires_grad_(True) for value in values
        )
        x, residual, post, comb = reference_inputs
        output = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2),
            residual,
        )
        expected = torch.autograd.grad(
            output,
            reference_inputs,
            grad_output,
        )

        actual = hc_post_explicit_backward_chunk(
            *values,
            grad_output,
        )
        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(actual_grad, expected_grad)

        skipped = hc_post_explicit_backward_chunk(
            *values,
            grad_output,
            need_x=False,
            need_residual=True,
            need_post=False,
            need_comb=False,
        )
        self.assertIsNone(skipped[0])
        torch.testing.assert_close(skipped[1], expected[1])
        self.assertIsNone(skipped[2])
        self.assertIsNone(skipped[3])


if __name__ == "__main__":
    unittest.main()

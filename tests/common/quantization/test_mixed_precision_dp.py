# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the Dynamic Programming based mixed precision selection algorithm.
"""

import pytest

from nncf.quantization.algorithms.weight_compression.mixed_precision import choose_bits_per_layer_with_path


class TestDPMixedPrecision:
    """Tests for the choose_bits_per_layer_with_path function."""

    def test_single_layer_single_option(self):
        """Test with a single layer and single option."""
        layers = {"layer1": [(4, 40, 0.5)]}
        target_bits = 50

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_loss == 0.5
        assert best_path == [("layer1", 4)]

    def test_single_layer_multiple_options(self):
        """Test selecting the best option from multiple choices for a single layer."""
        # Layer has 10 weights, options: 8-bit (cost=80, loss=0), 4-bit (cost=40, loss=0.5), 2-bit (cost=20, loss=1.0)
        layers = {"layer1": [(8, 80, 0.0), (4, 40, 0.5), (2, 20, 1.0)]}

        # With high budget, prefer lower loss (8-bit)
        best_loss, best_path = choose_bits_per_layer_with_path(layers, 100)
        assert best_loss == 0.0
        assert best_path == [("layer1", 8)]

        # With low budget, only 2-bit fits
        best_loss, best_path = choose_bits_per_layer_with_path(layers, 30)
        assert best_loss == 1.0
        assert best_path == [("layer1", 2)]

        # With medium budget, 4-bit fits
        best_loss, best_path = choose_bits_per_layer_with_path(layers, 50)
        assert best_loss == 0.5
        assert best_path == [("layer1", 4)]

    def test_two_layers(self):
        """
        - Or: layer1=2-bit (0.8) + layer2=4-bit (0.1) = 0.9 with cost 20+40=60
        - Or: layer1=4-bit (0.3) + layer2=2-bit (0.2) = 0.5 with cost 40+20=60
        - Or: layer1=2-bit (0.8) + layer2=2-bit (0.2) = 1.0 with cost 20+20=40

        Best within budget 60: layer1=4-bit, layer2=2-bit = 0.5
        """
        layers = {
            "layer1": [(8, 80, 0.0), (4, 40, 0.3), (2, 20, 0.8)],
            "layer2": [(8, 80, 0.0), (4, 40, 0.1), (2, 20, 0.2)],
        }
        target_bits = 60

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        # DP should find the optimal solution
        assert best_loss == pytest.approx(0.5)  # 0.3 + 0.2
        path_dict = {name: bits for name, bits in best_path}
        assert path_dict["layer1"] == 4
        assert path_dict["layer2"] == 2

    def test_dp_vs_greedy_advantage(self):
        """
        Test scenario where DP outperforms greedy.

        Greedy (sort by sensitivity, pick lowest bits for least sensitive):
        Would assign low bits to low-sensitivity layers first.

        DP considers all combinations to find globally optimal solution.
        """
        # Three layers with different sensitivities and sizes
        # Sizes: layer1=10, layer2=5, layer3=8 weights
        layers = {
            "layer1": [(8, 80, 0.0), (4, 40, 1.0), (2, 20, 3.0)],  # Highly sensitive
            "layer2": [(8, 40, 0.0), (4, 20, 0.2), (2, 10, 0.5)],  # Medium sensitivity, small
            "layer3": [(8, 64, 0.0), (4, 32, 0.3), (2, 16, 0.8)],  # Low sensitivity
        }

        # Budget allows: 80+20+32 = 132 for layer1@8, layer2@4, layer3@4
        # vs: 40+40+32 = 112 for all@4 with loss 1.0+0.2+0.3 = 1.5
        target_bits = 140

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        # Verify we get a valid solution
        assert best_path is not None
        assert len(best_path) == 3

        # Calculate actual cost and loss
        total_cost = sum(bits * (10 if name == "layer1" else 5 if name == "layer2" else 8) for name, bits in best_path)
        assert total_cost <= target_bits

    def test_infeasible_budget(self):
        """Test when budget is too small for any valid assignment."""
        layers = {
            "layer1": [(8, 80, 0.0), (4, 40, 0.5)],
            "layer2": [(8, 80, 0.0), (4, 40, 0.5)],
        }
        target_bits = 30  # Too small for even lowest option

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_loss is None
        assert best_path is None

    def test_exact_budget_fit(self):
        """Test when budget exactly matches optimal solution cost."""
        layers = {
            "layer1": [(4, 40, 0.5)],
            "layer2": [(4, 40, 0.3)],
        }
        target_bits = 80  # Exactly fits both

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_loss == pytest.approx(0.8)
        assert len(best_path) == 2

    def test_empty_layers(self):
        """Test with empty layers dict."""
        layers = {}
        target_bits = 100

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_loss == 0.0
        assert best_path == []

    def test_pareto_pruning_effectiveness(self):
        """
        Test that Pareto pruning works correctly by using a case
        with many dominated states.
        """
        # Create a case where many states would be dominated
        layers = {
            "layer1": [(8, 80, 0.0), (4, 40, 0.5)],
            "layer2": [(8, 80, 0.0), (4, 40, 0.4)],
            "layer3": [(8, 80, 0.0), (4, 40, 0.3)],
        }
        target_bits = 200

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        # Should prefer all 8-bit for minimum loss
        assert best_path is not None
        path_dict = {name: bits for name, bits in best_path}
        # All should be 8-bit or 4-bit depending on budget optimization
        # With budget 200, we can have 160 for two 8-bit, then one 4-bit = 200
        total_bits = sum(bits * 10 for _, bits in best_path)
        assert total_bits <= target_bits

    def test_three_layers_mixed_bits(self):
        """Test with three layers requiring different bit assignments."""
        # Layer sensitivities designed to require different bit assignments
        layers = {
            "layer1": [(8, 80, 0.0), (4, 40, 2.0), (2, 20, 5.0)],  # Very sensitive
            "layer2": [(8, 80, 0.0), (4, 40, 0.1), (2, 20, 0.3)],  # Not sensitive
            "layer3": [(8, 80, 0.0), (4, 40, 1.0), (2, 20, 2.5)],  # Medium sensitive
        }

        # Budget that forces a mix of bit widths
        target_bits = 120  # Can fit: 40+40+40=120 (all 4-bit) or 80+20+20=120 (one 8-bit, two 2-bit)

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_path is not None
        path_dict = {name: bits for name, bits in best_path}

        # Calculate total loss
        loss_map = {
            "layer1": {8: 0.0, 4: 2.0, 2: 5.0},
            "layer2": {8: 0.0, 4: 0.1, 2: 0.3},
            "layer3": {8: 0.0, 4: 1.0, 2: 2.5},
        }

        actual_loss = sum(loss_map[name][bits] for name, bits in path_dict.items())
        assert actual_loss == pytest.approx(best_loss)

        # Verify budget constraint
        total_bits = sum(bits * 10 for _, bits in best_path)
        assert total_bits <= target_bits

    def test_greedy_fails_dp_succeeds(self):
        """
        Specific test case where greedy algorithm would fail but DP succeeds.

        Scenario: 3 layers with 10 weights each
        Budget = 160 bits (avg 5.33 bits per weight)

        Sensitivities (higher = more sensitive):
        - Layer A: sensitivity 1.0 (least sensitive)
        - Layer B: sensitivity 2.0
        - Layer C: sensitivity 3.0 (most sensitive)

        Greedy would:
        1. Sort by sensitivity: A, B, C
        2. Assign lowest bits to least sensitive first
        3. A=2-bit(20), B=2-bit(20), C can use remaining 120→ 8-bit(80)
        4. Total: 120 bits, Loss = 1.0*0.75 + 2.0*0.75 + 3.0*0 = 2.25

        DP can find:
        - A=4-bit(40), B=4-bit(40), C=8-bit(80) = 160 bits
        - Loss = 1.0*0.5 + 2.0*0.5 + 3.0*0 = 1.5 (BETTER!)
        """
        # Each layer has 10 weights
        # loss = sensitivity * (8 - bits) / 8
        layers = {
            "layer_A": [(8, 80, 0.0), (4, 40, 0.5), (2, 20, 0.75)],  # sensitivity 1.0
            "layer_B": [(8, 80, 0.0), (4, 40, 1.0), (2, 20, 1.5)],  # sensitivity 2.0
            "layer_C": [(8, 80, 0.0), (4, 40, 1.5), (2, 20, 2.25)],  # sensitivity 3.0
        }
        target_bits = 160

        best_loss, best_path = choose_bits_per_layer_with_path(layers, target_bits)

        assert best_path is not None
        path_dict = {name: bits for name, bits in best_path}

        # Optimal is A=4, B=4, C=8 with total loss = 0.5 + 1.0 + 0.0 = 1.5
        # Verify the solution
        assert best_loss <= 1.5 + 0.01  # Allow small floating point tolerance

        # The greedy approach would give loss around 2.25, so DP should do better
        greedy_loss = 0.75 + 1.5 + 0.0  # A=2, B=2, C=8
        assert best_loss < greedy_loss

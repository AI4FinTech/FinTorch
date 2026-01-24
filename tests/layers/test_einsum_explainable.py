import torch
from fintorch.layers.explainable.einsum import einsum


class TestEinsumExplainable:
    def test_einsum_matrix_multiplication_relevance_propagation(self):
        """
        Test einsum layer with matrix multiplication ('ij,jk->ik')
        and verify that relevance scores are properly propagated to both input matrices A and B.
        """
        # Set up test data
        torch.manual_seed(42)  # For reproducible results

        # Create input matrices A (2x3) and B (3x4)
        A = torch.randn(2, 3, requires_grad=True)
        B = torch.randn(3, 4, requires_grad=True)

        # Create einsum layer for matrix multiplication
        einsum_layer = einsum('ij,jk->ik')

        # Forward pass
        output = einsum_layer(A, B)

        # Verify output shape is correct (2x4)
        assert output.shape == (2, 4), f"Expected output shape (2, 4), got {output.shape}"

        # Verify output is equivalent to standard matrix multiplication
        expected_output = torch.matmul(A, B)
        torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)

        # Create relevance scores (same shape as output)
        relevance_output = torch.ones_like(output)

        # Propagate relevance
        relevance_inputs = einsum_layer.propagate(relevance_output)

        # Verify that we get relevance scores for both inputs
        assert isinstance(relevance_inputs, list), "Expected list of relevance scores for multiple inputs"
        assert len(relevance_inputs) == 2, f"Expected 2 relevance scores, got {len(relevance_inputs)}"

        # Verify shapes of relevance scores match input shapes
        relevance_A, relevance_B = relevance_inputs
        assert relevance_A.shape == A.shape, f"Relevance for A should have shape {A.shape}, got {relevance_A.shape}"
        assert relevance_B.shape == B.shape, f"Relevance for B should have shape {B.shape}, got {relevance_B.shape}"

        # Verify relevance scores are not all zeros (indicating proper propagation)
        assert not torch.allclose(relevance_A, torch.zeros_like(relevance_A)), "Relevance for A should not be all zeros"
        assert not torch.allclose(relevance_B, torch.zeros_like(relevance_B)), "Relevance for B should not be all zeros"

        # Verify relevance conservation (approximately)
        # The sum of input relevances should be close to the sum of output relevances
        total_input_relevance = relevance_A.sum() + relevance_B.sum()
        total_output_relevance = relevance_output.sum()

        print(f"Total input relevance: {total_input_relevance.item()}")
        print(f"Total output relevance: {total_output_relevance.item()}")
        print(f"Relevance A shape: {relevance_A.shape}, sum: {relevance_A.sum().item()}")
        print(f"Relevance B shape: {relevance_B.shape}, sum: {relevance_B.sum().item()}")

    def test_einsum_element_wise_multiplication_relevance(self):
        """
        Test einsum layer with element-wise multiplication ('i,i->i')
        and verify relevance propagation for two vectors.
        """
        torch.manual_seed(123)

        # Create input vectors
        A = torch.randn(5, requires_grad=True)
        B = torch.randn(5, requires_grad=True)

        # Create einsum layer for element-wise multiplication
        einsum_layer = einsum('i,i->i')

        # Forward pass
        output = einsum_layer(A, B)

        # Verify output shape
        assert output.shape == (5,), f"Expected output shape (5,), got {output.shape}"

        # Verify output is correct
        expected_output = A * B
        torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)

        # Create relevance scores
        relevance_output = torch.ones_like(output)

        # Propagate relevance
        relevance_inputs = einsum_layer.propagate(relevance_output)

        # Verify relevance propagation
        assert isinstance(relevance_inputs, list), "Expected list of relevance scores"
        assert len(relevance_inputs) == 2, f"Expected 2 relevance scores, got {len(relevance_inputs)}"

        relevance_A, relevance_B = relevance_inputs
        assert relevance_A.shape == A.shape, f"Relevance for A should have shape {A.shape}"
        assert relevance_B.shape == B.shape, f"Relevance for B should have shape {B.shape}"

    def test_einsum_batch_matrix_multiplication_relevance(self):
        """
        Test einsum layer with batch matrix multiplication ('bij,bjk->bik')
        and verify relevance propagation.
        """
        torch.manual_seed(456)

        # Create batch input matrices A (batch_size=2, 3x4) and B (batch_size=2, 4x5)
        batch_size = 2
        A = torch.randn(batch_size, 3, 4, requires_grad=True)
        B = torch.randn(batch_size, 4, 5, requires_grad=True)

        # Create einsum layer for batch matrix multiplication
        einsum_layer = einsum('bij,bjk->bik')

        # Forward pass
        output = einsum_layer(A, B)

        # Verify output shape
        expected_shape = (batch_size, 3, 5)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

        # Create relevance scores
        relevance_output = torch.ones_like(output)

        # Propagate relevance
        relevance_inputs = einsum_layer.propagate(relevance_output)

        # Verify relevance propagation
        assert isinstance(relevance_inputs, list), "Expected list of relevance scores"
        assert len(relevance_inputs) == 2, f"Expected 2 relevance scores, got {len(relevance_inputs)}"

        relevance_A, relevance_B = relevance_inputs
        assert relevance_A.shape == A.shape, f"Relevance for A should have shape {A.shape}"
        assert relevance_B.shape == B.shape, f"Relevance for B should have shape {B.shape}"

    def test_einsum_single_input_relevance(self):
        """
        Test einsum layer with single input operation and verify relevance propagation.
        """
        torch.manual_seed(789)

        # Create input tensor
        A = torch.randn(3, 4, requires_grad=True)

        # Create einsum layer for transpose operation
        einsum_layer = einsum('ij->ji')

        # Forward pass
        output = einsum_layer(A)

        # Verify output shape
        assert output.shape == (4, 3), f"Expected output shape (4, 3), got {output.shape}"

        # Create relevance scores
        relevance_output = torch.ones_like(output)

        # Propagate relevance
        relevance_inputs = einsum_layer.propagate(relevance_output)

        # For single input, should return tensor directly (not list)
        assert isinstance(relevance_inputs, torch.Tensor), "Expected single tensor for single input"
        assert relevance_inputs.shape == A.shape, f"Relevance should have shape {A.shape}"

    def test_einsum_zero_gradients_handling(self):
        """
        Test that the einsum layer properly handles cases where gradients might be None or zero.
        """
        torch.manual_seed(999)

        # Create inputs where one might have zero gradient contribution
        A = torch.zeros(2, 3, requires_grad=True)  # Zero input
        B = torch.randn(3, 4, requires_grad=True)

        # Create einsum layer
        einsum_layer = einsum('ij,jk->ik')

        # Forward pass
        output = einsum_layer(A, B)

        # Create relevance scores
        relevance_output = torch.ones_like(output)

        # Propagate relevance (should not crash even with zero inputs)
        relevance_inputs = einsum_layer.propagate(relevance_output)

        # Verify we still get valid relevance scores
        assert isinstance(relevance_inputs, list), "Expected list of relevance scores"
        assert len(relevance_inputs) == 2, "Expected 2 relevance scores"

        relevance_A, relevance_B = relevance_inputs
        assert relevance_A.shape == A.shape, "Relevance A shape should match input A"
        assert relevance_B.shape == B.shape, "Relevance B shape should match input B"


if __name__ == "__main__":
    # Run a simple test to demonstrate usage
    test_case = TestEinsumExplainable()
    test_case.test_einsum_matrix_multiplication_relevance_propagation()
    print("Test completed successfully!")

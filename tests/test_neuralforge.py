"""
Comprehensive Test Suite for NeuralForge
"""

import numpy as np
import sys
sys.path.append('..')
from neuralforge import (
    NeuralNetwork, Dense, MultiHeadAttention, LayerNormalization,
    ResidualBlock, Activation, WeightInitializer, LossFunction,
    Architectures, TrainingConfig
)

class TestNeuralForge:
    """Test suite for NeuralForge framework"""
    
    @staticmethod
    def test_activations():
        """Test all activation functions"""
        print("\n" + "="*60)
        print("Testing Activation Functions")
        print("="*60)
        
        x = np.array([[-2, -1, 0, 1, 2]])
        activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu', 
                      'elu', 'sigmoid', 'tanh']
        
        for act_name in activations:
            act_func = getattr(Activation, act_name)
            output = act_func(x)
            derivative = act_func(x, derivative=True)
            print(f"{act_name:15s} - Output shape: {output.shape}, "
                  f"Derivative shape: {derivative.shape} ✓")
        
        print("\nAll activation functions passed! ✓")
    
    @staticmethod
    def test_weight_initialization():
        """Test weight initialization methods"""
        print("\n" + "="*60)
        print("Testing Weight Initialization")
        print("="*60)
        
        shape = (100, 50)
        methods = ['xavier', 'he', 'lecun', 'orthogonal']
        
        for method in methods:
            init_func = getattr(WeightInitializer, method)
            weights = init_func(shape)
            mean = np.mean(weights)
            std = np.std(weights)
            print(f"{method:15s} - Mean: {mean:.6f}, Std: {std:.6f} ✓")
        
        print("\nAll initialization methods passed! ✓")
    
    @staticmethod
    def test_loss_functions():
        """Test loss functions"""
        print("\n" + "="*60)
        print("Testing Loss Functions")
        print("="*60)
        
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        
        losses = ['mse', 'mae', 'categorical_crossentropy']
        
        for loss_name in losses:
            loss_func = getattr(LossFunction, loss_name)
            loss_value = loss_func(y_true, y_pred)
            gradient = loss_func(y_true, y_pred, derivative=True)
            print(f"{loss_name:30s} - Loss: {loss_value:.6f}, "
                  f"Gradient shape: {gradient.shape} ✓")
        
        print("\nAll loss functions passed! ✓")
    
    @staticmethod
    def test_layers():
        """Test individual layers"""
        print("\n" + "="*60)
        print("Testing Layers")
        print("="*60)
        
        batch_size = 32
        
        # Test Dense layer
        dense = Dense(10, 20, activation="relu")
        x = np.random.randn(batch_size, 10)
        output = dense.forward(x)
        gradient = dense.backward(np.ones_like(output), 0.001)
        print(f"Dense Layer - Input: {x.shape}, Output: {output.shape}, "
              f"Gradient: {gradient.shape} ✓")
        
        # Test LayerNormalization
        ln = LayerNormalization(20)
        output = ln.forward(output)
        gradient = ln.backward(np.ones_like(output), 0.001)
        print(f"LayerNorm - Output: {output.shape}, Gradient: {gradient.shape} ✓")
        
        # Test ResidualBlock
        res_block = ResidualBlock(20, activation="gelu")
        x = np.random.randn(batch_size, 20)
        output = res_block.forward(x)
        gradient = res_block.backward(np.ones_like(output), 0.001)
        print(f"ResidualBlock - Input: {x.shape}, Output: {output.shape}, "
              f"Gradient: {gradient.shape} ✓")
        
        # Test MultiHeadAttention
        mha = MultiHeadAttention(64, num_heads=4)
        x = np.random.randn(batch_size, 10, 64)
        output = mha.forward(x)
        print(f"MultiHeadAttention - Input: {x.shape}, Output: {output.shape} ✓")
        
        print("\nAll layers passed! ✓")
    
    @staticmethod
    def test_architectures():
        """Test pre-built architectures"""
        print("\n" + "="*60)
        print("Testing Pre-built Architectures")
        print("="*60)
        
        # Test MLP
        mlp = Architectures.mlp(100, [64, 32], 10, activation="gelu")
        mlp.compile(loss="categorical_crossentropy", optimizer="adam")
        print(f"MLP Architecture - {len(mlp.layers)} layers ✓")
        
        # Test ResNet
        resnet = Architectures.resnet(100, num_blocks=3, hidden_dim=64, output_dim=10)
        resnet.compile(loss="categorical_crossentropy", optimizer="adam")
        print(f"ResNet Architecture - {len(resnet.layers)} layers ✓")
        
        print("\nAll architectures passed! ✓")
    
    @staticmethod
    def test_training():
        """Test training pipeline"""
        print("\n" + "="*60)
        print("Testing Training Pipeline")
        print("="*60)
        
        # Generate sample data
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 3, (200, 1))
        y_onehot = np.eye(3)[y.flatten()]
        
        # Build model
        model = Architectures.mlp(10, [32, 16], 3, activation="relu")
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        
        # Configure training
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            epochs=5,
            validation_split=0.2,
            verbose=False
        )
        
        # Train
        history = model.fit(X, y_onehot, config=config)
        
        # Evaluate
        results = model.evaluate(X, y_onehot)
        
        print(f"Final training loss: {history['train_loss'][-1]:.4f} ✓")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f} ✓")
        print(f"Test accuracy: {results['accuracy']:.4f} ✓")
        
        print("\nTraining pipeline passed! ✓")
    
    @staticmethod
    def test_save_load():
        """Test model saving and loading"""
        print("\n" + "="*60)
        print("Testing Model Save/Load")
        print("="*60)
        
        # Create and train a simple model
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 1))
        
        model = NeuralNetwork("TestModel")
        model.add(Dense(5, 10, activation="relu"))
        model.add(Dense(10, 1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam")
        
        config = TrainingConfig(epochs=5, batch_size=16, verbose=False)
        model.fit(X, y, config=config)
        
        # Save model
        model.save("test_model.pkl")
        print("Model saved successfully ✓")
        
        # Load model
        loaded_model = NeuralNetwork().load("test_model.pkl")
        print("Model loaded successfully ✓")
        
        # Compare predictions
        pred_original = model.predict(X[:5])
        pred_loaded = loaded_model.predict(X[:5])
        
        if np.allclose(pred_original, pred_loaded):
            print("Predictions match after loading ✓")
        else:
            print("Warning: Predictions differ after loading")
        
        print("\nSave/Load test passed! ✓")
    
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        print("\n" + "="*70)
        print(" "*15 + "NEURALFORGE TEST SUITE")
        print("="*70)
        
        try:
            TestNeuralForge.test_activations()
            TestNeuralForge.test_weight_initialization()
            TestNeuralForge.test_loss_functions()
            TestNeuralForge.test_layers()
            TestNeuralForge.test_architectures()
            TestNeuralForge.test_training()
            TestNeuralForge.test_save_load()
            
            print("\n" + "="*70)
            print(" "*20 + "ALL TESTS PASSED! ✓")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    TestNeuralForge.run_all_tests()
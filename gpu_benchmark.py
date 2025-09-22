#!/usr/bin/env python3
"""
Quick benchmark to compare CPU vs MPS (GPU) training speed on M2 Pro Mac.
"""

import torch
import torch.nn as nn
import time
from models.resnet import ResNet18

def benchmark_device(device_name, num_batches=10):
    """Benchmark training speed on specified device."""
    device = torch.device(device_name)
    print(f"\n🔥 Benchmarking {device_name.upper()} performance...")
    
    # Create model and move to device
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create dummy data (batch_size=32, same as your training)
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_target = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Warmup
    for _ in range(3):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    if device.type == 'mps':
        torch.mps.synchronize()
    
    start_time = time.time()
    
    for i in range(num_batches):
        # Forward pass
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{num_batches} completed")
    
    # Synchronize to ensure all operations are complete
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    time_per_batch = total_time / num_batches
    
    print(f"  ✅ Total time: {total_time:.2f}s")
    print(f"  ⚡ Time per batch: {time_per_batch:.3f}s")
    print(f"  🚀 Batches per second: {1/time_per_batch:.2f}")
    
    return time_per_batch

def main():
    print("🧪 ResNet-18 Training Speed Benchmark")
    print("=" * 50)
    
    # Check available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Benchmark CPU
    cpu_time = benchmark_device("cpu", num_batches=10)
    
    # Benchmark MPS if available
    if torch.backends.mps.is_available():
        mps_time = benchmark_device("mps", num_batches=10)
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        print(f"\n🎯 RESULTS:")
        print(f"  CPU time per batch: {cpu_time:.3f}s")
        print(f"  MPS time per batch: {mps_time:.3f}s")
        print(f"  🚀 MPS Speedup: {speedup:.2f}x faster!")
        
        # Estimate time savings for remaining epochs
        current_epoch = 13
        remaining_epochs = 50 - current_epoch
        batches_per_epoch = 1563  # Approximate for CIFAR-10 with batch_size=32
        
        remaining_batches = remaining_epochs * batches_per_epoch
        cpu_remaining_time = remaining_batches * cpu_time / 3600  # hours
        mps_remaining_time = remaining_batches * mps_time / 3600  # hours
        time_saved = cpu_remaining_time - mps_remaining_time
        
        print(f"\n⏰ TIME ESTIMATION for remaining {remaining_epochs} epochs:")
        print(f"  CPU: {cpu_remaining_time:.1f} hours")
        print(f"  MPS: {mps_remaining_time:.1f} hours")
        print(f"  💰 Time saved: {time_saved:.1f} hours ({time_saved*60:.0f} minutes)")
        
        if speedup > 2:
            print(f"\n✅ RECOMMENDATION: Switch to MPS! {speedup:.1f}x speedup is significant.")
        else:
            print(f"\n⚠️  RECOMMENDATION: Speedup is modest ({speedup:.1f}x). Consider if worth restarting.")
    else:
        print("\n❌ MPS not available on this system")

if __name__ == "__main__":
    main() 
import torch

def main():
    print("GPU Verification:")
    print(f"- PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"- CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        
        # Convert bytes to GB
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / (1024**3)
        print(f"- Memory: ~{memory_gb:.1f}GB")
    else:
        print("- GPU: None")
        print("- Memory: N/A")

if __name__ == "__main__":
    main()

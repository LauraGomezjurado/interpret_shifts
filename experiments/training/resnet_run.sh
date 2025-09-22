# When your ResNet is ready:
python visualize_saliency_maps_cifar100.py \
  --model_path resnet18_cifar10_best.pth \
  --model_type resnet \
  --method saliency \
  --num_samples 8 \
  --output_dir saliency_cifar100_resnet
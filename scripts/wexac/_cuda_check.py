import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device:", torch.cuda.get_device_name(i))
    props = torch.cuda.get_device_properties(i)
    print("VRAM (GB): %.1f" % (props.total_memory / 1e9))

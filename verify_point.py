import argparse
import gc
import torch

from verifier import ABCrown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    image = torch.load(args.image_path, map_location=device)
    image = image.to(device)

    verifier = ABCrown('cuda')
    result = verifier.verify(
        model,
        image,
        10,
        args.label,
        args.epsilon
    )

    print(result.status)

    # HARD CLEANUP
    del verifier, model, image, result
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
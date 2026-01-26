import torch
import torch.nn as nn
from typing import Literal
import torch.nn.functional as F
from abcrown.api import SolveResult
from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    VerificationSpec,
    input_vars,
    output_vars,
)

class ABCrown:
    def __init__(
        self,
        device:Literal['cpu', 'cuda', 'mps']='cuda',
        custom_config:dict|None=None
    ) -> None:
        """
        Initializes the verifier with the specified device and configuration.
        Args:
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to be used 
                for computation. Defaults to 'cuda'.
            custom_config (dict | None, optional): A dictionary containing custom 
                configuration options. If None, an empty dictionary is used. 
                Defaults to None.
        Returns:
            None
        """
        
        assert device in ['cpu', 'cuda', 'mps'], "device must be one of 'cpu', 'cuda', or 'mps'"
        assert custom_config is None or isinstance(custom_config, dict), "custom_config must be a dictionary or None"
        
        custom_config = custom_config if custom_config is not None else {}
        custom_config['general'] = {
            "device": device,
        }
        custom_config['attack'] = {
            "general_attack": False,
            "pgd_order": "after",
            "pgd_steps": 1,
        }
        self.config = ConfigBuilder().from_defaults()
        self.config = self.override_config(custom_config)
        
    def override_config(self, custom_config:dict) -> ConfigBuilder:
        """
        Overrides the current configuration with the provided custom configuration.
        Args:
            custom_config (dict): A dictionary containing the custom configuration 
                values to update the existing configuration.
        Returns:
            ConfigBuilder: The updated configuration builder instance.
        """
        
        assert isinstance(custom_config, dict), "custom_config must be a dictionary"
        
        return self.config.update(custom_config)
    
    def verify(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        n_classes: int,
        correct_class_index: int,
        eps: float,
    ) -> SolveResult:
        """
        Verifies the robustness of a given model for a specific input image under
        an epsilon perturbation constraint. The function checks whether the model
        consistently classifies the perturbed input as the correct class.
        Args:
            model (torch.nn.Module): The neural network model to verify.
            image (torch.Tensor): The input image tensor to verify against.
            n_classes (int): The total number of classes in the classification task.
            correct_class_index (int): The index of the correct class for the input image.
            eps (float): The maximum perturbation allowed for the input image.
        Returns:
            dict: A dictionary containing the result of the verification process with
            the following keys:
            - status: str, one of 'verified', 'unsafe-pgd', 'unsafe-bab', 
              'safe-incomplete' (no PGD needed), 'unknown'.
            - success: bool, True if the property is satisfied or a counterexample 
              is confirmed when unsafety is expected.
            - reference: dict, optional intermediate data (e.g., bounds, attack traces).
            - stats: dict, metadata such as elapsed time, PGD iterations, BaB splits.
        """
        
        assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        assert isinstance(image, torch.Tensor), "image must be a torch.Tensor"
        assert isinstance(n_classes, int) and n_classes > 0, "n_classes must be a positive integer"
        assert isinstance(correct_class_index, int) and 0 <= correct_class_index < n_classes, "correct_class_index must be an integer within the range of [0, n_classes)"
        assert isinstance(eps, float) and eps >= 0, "eps must be a non-negative float"
                
        x = input_vars(image.shape)
        y = output_vars(n_classes)
        input_constraint = (x >= image - eps) & (x <= image + eps)
        output_constraint = None
        for index in range(0, n_classes):
            if index != correct_class_index:
                if output_constraint is None:
                    output_constraint = y[correct_class_index] > y[index]
                else:
                    output_constraint = output_constraint & (y[correct_class_index] > y[index])
                
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
    
        solver = ABCrownSolver(spec, model, config=self.config) # type: ignore
        result = solver.solve()
        
        return result

class PGDVerifier():
    def __init__(self, device: Literal['cpu', 'mps', "cuda"] | str='cpu') -> None:
        self.device = device

    def verify(
        self,
        model:nn.Module,
        images:torch.Tensor,
        labels:torch.Tensor,
        epsilon:float=8/255,
        alpha:float=2/255,
        num_steps:int=10,
        clamp_min:float=0.0,
        clamp_max:float=1.0,
        random_start:bool=True,
    ):
        """
        Perform PGD attack (L-infinity).

        Args:
            model: torch.nn.Module
            images: input tensor of shape (N, C, H, W)
            labels: true labels
            epsilon: max perturbation
            alpha: step size
            num_steps: number of PGD iterations
            clamp_min, clamp_max: valid data range
            random_start: whether to start from a random point in the epsilon-ball

        Returns:
            Adversarial examples tensor
        """
        with torch.enable_grad():
            model.eval()

            # Clone inputs
            ori_images = images.detach()
            adv_images = ori_images.clone()

            if random_start:
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
                adv_images = torch.clamp(adv_images, clamp_min, clamp_max)

            for _ in range(num_steps):
                adv_images.requires_grad = True

                outputs = model(adv_images)
                successes = torch.argmax(outputs, dim=1) != labels
                if torch.all(successes):
                    break
                
                loss = F.cross_entropy(outputs, labels)

                grad = torch.autograd.grad(
                    loss, adv_images, retain_graph=False, create_graph=False
                )[0]

                # Gradient ascent step
                adv_images = adv_images + alpha * grad.sign()

                # Project back into epsilon-ball
                eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
                adv_images = ori_images + eta

                # Clamp to valid range
                adv_images = torch.clamp(adv_images, clamp_min, clamp_max).detach()

            return adv_images, successes

# NN-Project-DiffusionEditing
Use diffusion models to perform complex edits on images from user instructions

# requirements
This project will implement the InstructPix2Pix approach [@InstructPix2Pix: Learning to Follow Image Editing Instructions], which fine-tunes a text-to-image diffusion model for editing tasks. Students will generate a synthetic paired dataset of (input image, edit instruction, edited image) using existing image captioning and text-to-image models. Then they will train a conditional diffusion model that, given an input image and an instruction, directly outputs the edited image without per-instance fine-tuning. The model should learn to apply diverse transformations (e.g. “add fireworks in sky” or “make it snowy”) as demonstrated by Brooks et al..

* Evaluate the model on real user-provided instructions and images (beyond the synthetic training set) and integrate an evaluation metric for edit fidelity vs. prompt (similar to InstructPix2Pix’s zero-shot generalization test).


# Resources

### Repositorys

* [Pix2Pix Model](https://github.com/timothybrooks/instruct-pix2pix?tab=readme-ov-file)
* [Concadia Dataset](https://github.com/elisakreiss/concadia)

### Papers

* [DreamBooth](https://arxiv.org/abs/2208.12242)
* [S2Edit](https://arxiv.org/abs/2507.04584)
* [Diffusion Model](https://arxiv.org/abs/2209.00796)

### Models

* [Stable-diffusion-large](stabilityai/stable-diffusion-3.5-large)
* [Stable-diffusion-turbo]()

from nnsight import LanguageModel
import torch as t
from buffer import ActivationBuffer
from trainers import MatryoshkaBatchTopKSAE, MatryoshkaBatchTopKTrainer
from training import trainSAE
from utils import load_iterable_dataset

device = "cuda:0"
model_name = "EleutherAI/pythia-70m-deduped" # can be any Huggingface model

model = LanguageModel(
    model_name,
    device_map=device,
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

data = load_iterable_dataset("stas/openwebtext-10k", streaming=True)

buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=512,  # you can set this higher or lower depending on your available memory
    device=device,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": MatryoshkaBatchTopKTrainer,
    "dict_class": MatryoshkaBatchTopKSAE,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "steps": 1000,
    "k": 32,
    "layer": 1,
    "lm_name": model_name,
    "group_fractions": [0.1, 0.3, 0.6],
    "device": device,
    "warmup_steps": 0,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
    steps=1000,
    use_wandb=False,
    wandb_project="",
    save_steps=None,
    # save_dir="",
    log_steps=10,
    activations_split_by_head=False,
    transcoder=False,
    run_cfg={},
    normalize_activations=False,
    verbose=False,
    device=device,
    autocast_dtype=t.bfloat16,
)
import diffusers

# Dictionary mapping scheduler names to their configuration.
# This includes the class name of the scheduler and any keyword arguments.
scheduler_map = {
    "DPM++ 2M": {"class_name": "DPMSolverMultistepScheduler", "kwargs": {}},
    "DPM++ 2M Karras": {
        "class_name": "DPMSolverMultistepScheduler",
        "kwargs": {"use_karras_sigmas": "yes", "final_sigmas_type" : "sigma_min"},
    },
    # "DPM++ 2M SDE": {
    #     "class_name": "DPMSolverMultistepScheduler",
    #     "kwargs": {"algorithm_type": "sde-dpmsolver++"},
    # },
    # "DPM++ 2M SDE Karras": {
    #     "class_name": "DPMSolverMultistepScheduler",
    #     "kwargs": {"use_karras_sigmas": "yes", "algorithm_type": "sde-dpmsolver++"},
    # },
    "DPM++ SDE": {"class_name": "DPMSolverSinglestepScheduler", "kwargs": {}},
    "DPM++ SDE Karras": {
        "class_name": "DPMSolverSinglestepScheduler",
        "kwargs": {"use_karras_sigmas": "yes", "final_sigmas_type" : "sigma_min"},
    },
    "DPM2": {"class_name": "KDPM2DiscreteScheduler", "kwargs": {}},
    "DPM2 Karras": {
        "class_name": "KDPM2DiscreteScheduler",
        "kwargs": {"use_karras_sigmas": "yes"},
    },
    "DPM2 a": {"class_name": "KDPM2AncestralDiscreteScheduler", "kwargs": {}},
    "DPM2 a Karras": {
        "class_name": "KDPM2AncestralDiscreteScheduler",
        "kwargs": {"use_karras_sigmas": "yes"},
    },
    "Euler": {"class_name": "EulerDiscreteScheduler", "kwargs": {}},
    "Euler a": {"class_name": "EulerAncestralDiscreteScheduler", "kwargs": {}},
    "Heun": {"class_name": "HeunDiscreteScheduler", "kwargs": {}},
    "LMS": {"class_name": "LMSDiscreteScheduler", "kwargs": {}},
    "LMS Karras": {
        "class_name": "LMSDiscreteScheduler",
        "kwargs": {"use_karras_sigmas": "yes"},
    },
    "DEIS": {"class_name": "DEISMultistepScheduler", "kwargs": {}},
    "UniPC": {"class_name": "UniPCMultistepScheduler", "kwargs": {}},
    "DDIM": {"class_name": "DDIMScheduler", "kwargs": {}},
    "DDPM": {"class_name": "DDPMScheduler", "kwargs": {}},
    "EDM Euler": {"class_name": "EDMEulerScheduler", "kwargs": {}},
    "PNDM": {"class_name": "PNDMScheduler", "kwargs": {}},
    "LCM": {"class_name": "LCMScheduler", "kwargs": {
        'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': "scaled_linear", 'set_alpha_to_one': True, 'rescale_betas_zero_snr': False, 'thresholding': False
        }},
}

# 从 scheduler_map 获取调度器清单
schedulers = list(scheduler_map.keys()) # List of available scheduler names

def set_scheduler(pipe: diffusers.DiffusionPipeline, name: str):
    """
    Sets the scheduler for a diffusion pipeline based on the provided name. 
    If the name is "None", the pipeline's default scheduler is used.

    Args:
        pipe (diffusers.DiffusionPipeline): The diffusion pipeline to configure.
        name (str): The name of the scheduler to set.

    Raises:
        Exception: If the specified scheduler name is unknown.
    """
    print("---------------------debug ", name) # Debug print statement
    scheduler_cfg = scheduler_map.get(name) # Retrieve the scheduler configuration

    # Handle the case when no specific scheduler is requested (name is "None")
    if name == "None":
        if hasattr(pipe.scheduler, "scheduler_config"):
           default_class_name = pipe.scheduler.scheduler_config["_class_name"]
        else:
            default_class_name = pipe.scheduler.config["_class_name"]
        # same scheduler
        if default_class_name == type(pipe.scheduler).__name__:
            # If the pipeline already has the default scheduler, do nothing
            return 
        else:
            # Get the default scheduler class 
            scheduler_class = getattr(diffusers, default_class_name)
    # If a scheduler configuration is found
    elif scheduler_cfg is not None: 
        # Get the scheduler class from the diffusers library
        scheduler_class = getattr(diffusers, scheduler_cfg["class_name"])
    else:
        # If the scheduler name is not found, raise an exception
        raise Exception(f"unkown scheduler name \"{name}\"")
    print(f"load scheduler {name}")
    # Set the scheduler for the pipeline using its configuration and keyword arguments
    pipe.scheduler = scheduler_class.from_config(
        pipe.scheduler.config,  **scheduler_cfg["kwargs"]
    )


# while True:
#     print("Please select a scheduler:")

#     # 打印调度器清单
#     for i, scheduler_name in enumerate(schedulers):
#         print(
#             f"{i}: {scheduler_name}, class_name: {scheduler_map[scheduler_name]['class_name']}, kwargs: {scheduler_map[scheduler_name]['kwargs']}"
#         )

#     # 获取用户输入
#     choice = int(input("请输入你选择的调度器的序号："))

#     # 获取用户选择的调度器名称
#     selected_scheduler_name = schedulers[choice]

#     scheduler_cfg = scheduler_map[selected_scheduler_name]
#     scheduler_class = getattr(diffusers, scheduler_cfg["class_name"])
#     print(
#         f"Selected scheduler: {selected_scheduler_name}, class_name: {scheduler_cfg['class_name']}, kwargs: {scheduler_cfg['kwargs']}"
#     )
#     # pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
#     # pipe.scheduler.use_karras_sigmas = True if scheduler_cfg["kwargs"]["use_karras_sigmas"] == "yes" else False
#     # pipe.scheduler.algorithm_type = scheduler_cfg["kwargs"]["algorithm_type"]
#     pipe.scheduler = scheduler_class.from_pretrained(
#         model_id, subfolder="scheduler", cache_dir=cache_dir, **scheduler_cfg["kwargs"]
#     )

#     num_inference_steps = 30
#     guidance_scale = 7.5
#     generator = torch.Generator(device="xpu").manual_seed(1)

#     image = pipe(
#         prompt=prompt,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         generator=generator,
#     ).images[0]
#     image.show()
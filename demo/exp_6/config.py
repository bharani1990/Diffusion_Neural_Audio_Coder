class Config:
    def __init__(self, preset="balanced"):
        if preset == "quality":
            self.batch_size = 8
            self.max_epochs = 200
            self.lr = 5e-5
            self.patience = 25
            self.latent_dim = 24
            self.compression_weight = 0.4
            self.diffusion_weight = 0.6
        elif preset == "speed":
            self.batch_size = 32
            self.max_epochs = 50
            self.lr = 1e-4
            self.patience = 10
            self.latent_dim = 8
            self.compression_weight = 0.6
            self.diffusion_weight = 0.4
        elif preset == "memory":
            self.batch_size = 4
            self.max_epochs = 100
            self.lr = 1e-4
            self.patience = 15
            self.latent_dim = 12
            self.compression_weight = 0.5
            self.diffusion_weight = 0.5
            self.use_4bit = True
        else:
            self.batch_size = 16
            self.max_epochs = 50
            self.lr = 1e-4
            self.patience = 15
            self.latent_dim = 16
            self.compression_weight = 0.5
            self.diffusion_weight = 0.5
            self.use_4bit = True
        
        self.frames = 120
        self.timesteps = 1000
        self.num_workers = 4
        self.grad_accum = 4
        self.precision = "fp32"
        self.devices = 1

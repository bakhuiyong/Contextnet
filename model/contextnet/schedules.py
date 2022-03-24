
class transformer_learning_rate_scheduler:

    def __init__(self, optimizer, dim_model, warmup_steps, K):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.K = K

    def step(self):
        
        self.model_step += 1
        s = self.model_step + 1

        arg1 = s**-0.5 
        arg2 = s * (self.warmup_steps**-1.5) 
        self.optimizer.param_groups[0]['lr'] = self.K * self.dim_model**-0.5 * min(arg1, arg2)
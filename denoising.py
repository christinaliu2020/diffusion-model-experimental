import diffusion_model_current as dm

sampler = dm.Euler_Maruyama_sampler

sample_batch_size = 64

samples = sampler(dm.score_model,
                  dm.marginal_prob_std,
                  dm.solve_t_given_std,
                  dm.diffusion_coeff_fn,
                  sample_batch_size,
                  device='cuda',
                  eps_val= 0.05,
                  t_val = 0.08)
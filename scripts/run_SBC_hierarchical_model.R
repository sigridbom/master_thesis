# load packages
cat("Installing packages")
pacman::p_load(
  tidyverse,    # Core suite for data manipulation (dplyr) and plotting (ggplot2)
  here,         # For creating robust file paths relative to the project root (optional but good practice)
  cmdstanr,     # The R interface to Stan (via CmdStan)
  SBC,          # Simulation-Based Calibration Checks
  future,       # Parallel processing
  furrr,        # Parallel mapping
  pracma,       # to be used in the probit regression formula
  cli,
  progressr,     # to get progress bar in SBC
  remotes,
  msm # to install specific version of ggplot needed to make priorsense dens plot
)

# to get progress bar
options(progressr.enable = TRUE)
handlers(global = TRUE)
#handlers("txtprogressbar")
handlers("cli")
options(progressr.clear = FALSE)


devtools::install_github("hyunjimoon/SBC")

cat("Packages are installed \n")

# define relative paths
stan_model_dir <- here("stan")
stan_results_dir <- here("simmodels")

if (!dir.exists(stan_results_dir)) dir.create(stan_results_dir)
if (!dir.exists(stan_model_dir)) dir.create(stan_model_dir)

variable_name <- "hierarchical_hrd_model"
n_sims <- 1000   # number of SBC datasets/fits (matches SBC::generate_datasets(generator, 50) below)

# Parallel SBC across simulations (SBC auto-balances cores per fit)
# future::plan(multisession, workers = future::availableCores())
# options(SBC.min_chunk_size = 1)
# cat(sprintf("Parallel SBC: %d workers\n", future::nbrOfWorkers()))

n_cores_total <- parallel::detectCores()
n_workers     <- max(1, n_cores_total - 2)

future::plan(multisession, workers = n_workers)
options(SBC.min_chunk_size = 5)
cat(sprintf("Detected %d cores, running %d parallel SBC workers (parallel_chains = 1)\n",
            n_cores_total, n_workers))

# define functions
probit_regression <- function(x, alpha, beta, lambda) {
  lambda + (1 - 2 * lambda) * (0.5 + 0.5 * pracma::erf((x - alpha) / (beta * sqrt(2))))
}

inv_logit <- function(x) {
  y = 1 / (1 + exp(-x))
  return(y)
}

# --- GROUP SETTINGS ---
S <- 60     # number of participants per group dataset
N <- 40     # trials per participant

# ---------------------------------------------------------------------------
# Hierarchical data-generating process
#
# Draws the group-level hyperparameters (mu_*, sigma_*) from their priors,
# then draws each agent's individual parameters (alpha, b_log, lambda_logit)
# from those hyperparameters, then simulates trial-level HRD data for every
# agent. This replaces the flat (non-hierarchical) generate_group_dataset().
#
# NOTE: sigma_* hyperparameters are drawn as exponential(rate), matching the
# prior_sigma_*_rate naming used below in generate_sbc_dataset(). Adjust if
# your Stan model uses a different prior family for these.
# ---------------------------------------------------------------------------
simulate_hierarchical_hrd_data <- function(
    n_agent = 60,
    n_trials = 40,
    prior_mu_alpha_loc = -10,
    prior_mu_alpha_scale = 7,
    prior_sigma_alpha_rate = 0.1,
    prior_mu_b_log_loc = 2.5,
    prior_mu_b_log_scale = 0.4,
    prior_sigma_b_log_rate = 2.5,
    prior_mu_lambda_logit_loc = -3,
    prior_mu_lambda_logit_scale = 0.4,
    prior_sigma_lambda_logit_rate = 2.5) {

  # 1. Draw group-level hyperparameters from their priors
  mu_alpha    <- rnorm(1, prior_mu_alpha_loc, prior_mu_alpha_scale)
  sigma_alpha <- rexp(1, prior_sigma_alpha_rate)

  mu_b_log    <- rnorm(1, prior_mu_b_log_loc, prior_mu_b_log_scale)
  sigma_b_log <- rexp(1, prior_sigma_b_log_rate)

  mu_lambda_logit    <- rnorm(1, prior_mu_lambda_logit_loc, prior_mu_lambda_logit_scale)
  sigma_lambda_logit <- rexp(1, prior_sigma_lambda_logit_rate)

  # 2. Draw agent-level (subject-level) parameters from the hyperparameters
  alpha_agent        <- rnorm(n_agent, mu_alpha, sigma_alpha)
  b_log_agent         <- rnorm(n_agent, mu_b_log, sigma_b_log)
  beta_agent          <- exp(b_log_agent)
  lambda_logit_agent  <- rnorm(n_agent, mu_lambda_logit, sigma_lambda_logit)
  lambda_agent        <- 0.5 * inv_logit(lambda_logit_agent)

  # 3. Simulate trial-level data for every agent
  sim_data <- purrr::map_dfr(1:n_agent, function(s) {
    deltaBPM <- msm::rtnorm(n_trials, alpha_agent[s], 2 * beta_agent[s], lower = -50, upper = 50)
    
    p_choice <- probit_regression(
      x = deltaBPM,
      alpha = alpha_agent[s],
      beta = beta_agent[s],
      lambda = lambda_agent[s]
    )

    choice <- rbinom(n_trials, size = 1, prob = p_choice)

    tibble(
      n_agent  = s,
      n_trials = 1:n_trials,
      deltaBPM = deltaBPM,
      choice   = choice,
      alpha        = alpha_agent[s],
      b_log        = b_log_agent[s],
      beta         = beta_agent[s],
      lambda_logit = lambda_logit_agent[s],
      lambda       = lambda_agent[s],
      # Placeholder individual latent parameter kept for backward
      # compatibility with the theta_true extraction step below.
      # This model tracks 3 individual parameters (alpha/b_log/lambda_logit)
      # rather than a single theta -- adjust if you want per-parameter SBC.
      theta = alpha_agent[s],
      mu_alpha = mu_alpha,
      sigma_alpha = sigma_alpha,
      mu_b_log = mu_b_log,
      sigma_b_log = sigma_b_log,
      mu_lambda_logit = mu_lambda_logit,
      sigma_lambda_logit = sigma_lambda_logit
    )
  })

  sim_data
}

# ---------------------------------------------------------------------------
# Define the strict SBC generator wrapper
# ---------------------------------------------------------------------------
generate_sbc_dataset <- function(n_agent = 60, n_trials = 40) {

  # 1. Simulate the universe (draws hyperparams and agent params from priors)
  sim_data <- simulate_hierarchical_hrd_data(
    n_agent = n_agent,
    n_trials = n_trials
  )

  sim_data <- sim_data %>%
    rename(
      S = n_agent,
      N = n_trials
    )

  # 2. Extract the true generative hyperparameters
  mu_alpha_true <- sim_data$mu_alpha[1]
  sigma_alpha_true <- sim_data$sigma_alpha[1]
  mu_b_log_true <- sim_data$mu_b_log[1]
  sigma_b_log_true <- sim_data$sigma_b_log[1]
  mu_lambda_logit_true <- sim_data$mu_lambda_logit[1]
  sigma_lambda_logit_true <- sim_data$sigma_lambda_logit[1]
  #deltaBPM <- sim_data$deltaBPM

  # 3. Extract the true individual latent parameters (must preserve agent order)
  theta_true <- sim_data |>
    distinct(S, theta) |>
    arrange(S) |>
    pull(theta)

  # 4. Prepare the strictly typed list for CmdStanR
  stan_data <- list(
    N = nrow(sim_data),
    S = n_agent,
    subj = sim_data$S,
    dBPM = sim_data$deltaBPM,
    choice = sim_data$choice,
    prior_mu_alpha_loc            = -10,     # HRD subjects systematically underestimate -> negative mean
    prior_mu_alpha_scale          =  7,
    prior_sigma_alpha_rate        =  0.1,    # E[sigma_alpha] = 1/rate = 10 BPM
    prior_mu_b_log_loc            =  2.5,    # exp(2.5) ~ 12 BPM slope
    prior_mu_b_log_scale          =  0.4,
    prior_sigma_b_log_rate        =  2.5,    # E[sigma_b_log] = 1/2.5 = 0.4
    prior_mu_lambda_logit_loc     = -3,      # 0.5 * inv_logit(-3) ~ 0.024
    prior_mu_lambda_logit_scale   =  0.4,
    prior_sigma_lambda_logit_rate =  2.5,    # E[sigma_lambda_logit] = 0.4
    run_diagnostics = 0L # not needed now
  )

  # 5. Return the exact structure required by the SBC package
  list(
    variables = list(mu_alpha = mu_alpha_true,
                      sigma_alpha = sigma_alpha_true,
                      mu_b_log = mu_b_log_true,
                      sigma_b_log = sigma_b_log_true,
                      mu_lambda_logit = mu_lambda_logit_true,
                      sigma_lambda_logit = sigma_lambda_logit_true
    ),
    generated = stan_data
  )
}

# Define the model
cat("Load stan file\n")

file <- file.path(stan_model_dir, 'simple_model_hrd_hierarchical.stan')
model <- cmdstan_model(file)

# Define the backend engine
backend <- SBC::SBC_backend_cmdstan_sample(
  model,
  chains = 2,
  parallel_chains = 1,
  iter_warmup = 1000,
  iter_sampling =4000,
  refresh = 0,
  adapt_delta = 0.95,
  max_treedepth = 10
)

# Register the generator function
generator <- SBC::SBC_generator_function(generate_sbc_dataset, n_agent = S, n_trials = N)

# run SBC
cat("Run SBC\n")
t_start <- proc.time()
sbc_datasets_hier <- SBC::generate_datasets(generator, n_sims = n_sims)
sbc_results_hier <- SBC::compute_SBC(sbc_datasets_hier, backend, keep_fits = FALSE)
elapsed_sec <- (proc.time() - t_start)[["elapsed"]]
elapsed_min <- floor(elapsed_sec / 60)
elapsed_rem_sec <- round(elapsed_sec %% 60)

# build output path
timestamp_str <- format(Sys.time(), "%y%m%d%H%M")
sbc_filename <- sprintf("%s_%s_%d_sims_%dsubj_%dtrials_%dm_%ds.rds",
                        timestamp_str, variable_name, n_sims, S, N,
                        elapsed_min, elapsed_rem_sec)
sbc_path <- file.path(stan_results_dir, sbc_filename)

# save results
saveRDS(
  list(
    datasets = sbc_datasets_hier,
    results = sbc_results_hier,
    S = S,
    N = N
  ),
  sbc_path
)
cat(sprintf("Saved results to: %s\n", sbc_path))

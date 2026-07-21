# load packages
cat("Installing packages\n")
library(pacman)
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


# Only reinstall the dev SBC package if it's missing. Reinstalling from
# GitHub on every run silently costs several minutes (clone + recompile)
# before any simulation even starts. Set force_sbc_reinstall <- TRUE below
# if you specifically need to pull the latest commit.
force_sbc_reinstall <- FALSE
if (force_sbc_reinstall || !requireNamespace("SBC", quietly = TRUE)) {
  devtools::install_github("hyunjimoon/SBC")
}

cat("Packages are installed \n")

# define relative paths
stan_model_dir <- here("stan")
stan_results_dir <- here("simmodels")
batch_dir <- here("simmodels", "batches")   # per-batch checkpoint files live here

if (!dir.exists(stan_results_dir)) dir.create(stan_results_dir, recursive = TRUE)
if (!dir.exists(stan_model_dir)) dir.create(stan_model_dir, recursive = TRUE)
if (!dir.exists(batch_dir)) dir.create(batch_dir, recursive = TRUE)

variable_name <- "hierarchical_hrd_model"
prior_type <- "half_normal"

# ---------------------------------------------------------------------------
# BATCHING CONFIG
#
# Instead of running all simulations in one long call (which loses everything
# if the job dies partway through), we split the run into n_batches batches
# of batch_size simulations each. Every batch is fit and written to disk
# straight away. If the Slurm job gets killed/times out, just resubmit the
# same job: batches already saved to disk are detected and skipped, so you
# only pay for the work that hadn't finished yet.
#
# Set to production values (200 x 10 = 2000 sims) below. For a quick sanity
# test, temporarily lower these (e.g. batch_size <- 20, n_batches <- 1) and
# remember to change them back before the real run.
# ---------------------------------------------------------------------------
batch_size <- 100   # simulations per batch - needs to be at least 60 to work well with SBC.min_chunk_size = 5 (sims only get split into chunks of at least 5 sims, so to utilize all worker, you need a larger number of sims)
n_batches  <- 20    # number of batches -> n_batches * batch_size total sims
n_sims     <- batch_size * n_batches  # 2000 total simulations

# backend config
n_iter     = 4000
n_thin     = 5

# --- GROUP SETTINGS ---
S <- 60     # number of participants per group dataset
N <- 40     # trials per participant

# Stable identifier for this configuration, used to name batch checkpoint
# files. Deliberately NOT time-stamped, so that resubmitting the job after a
# crash matches the same batch filenames and can resume correctly. Includes
# n_batches (not just batch_size) so runs with a different total number of
# batches never collide on filenames, even if batch_size is the same.
run_id <- sprintf(
  "%s_%dsubj_%dtrials_%diter_%dthin_%s_%dsimsperbatch_%dbatches",
  variable_name, S, N, n_iter, n_thin, prior_type, batch_size, n_batches
)

# ---------------------------------------------------------------------------
# PARALLELIZATION SETUP
#
# When running under Slurm, use the number of cores actually allocated to
# this job (SLURM_CPUS_PER_TASK, set automatically from your #SBATCH -c
# value) rather than autodetecting the physical machine's core count. This
# way, the script always respects however many cores you asked Slurm for
# in the batch script - just change the #SBATCH -c line to scale up/down,
# no need to edit this script. Falls back to autodetection when run outside
# Slurm (e.g. testing interactively).
# ---------------------------------------------------------------------------
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = NA)
if (!is.na(slurm_cpus) && slurm_cpus != "") {
  n_cores_total <- as.integer(slurm_cpus)
  cat(sprintf("Running under Slurm: SLURM_CPUS_PER_TASK = %d\n", n_cores_total))
} else {
  n_cores_total <- parallelly::availableCores()
  cat(sprintf("Not running under Slurm: autodetected %d available cores\n", n_cores_total))
}

chains_per_fit   <- 2
cores_to_reserve <- 2
cores_available  <- max(1, n_cores_total - cores_to_reserve)
n_workers        <- max(1, floor(cores_available / chains_per_fit))

# Capping workers below what cores alone would allow. Testing at
# batch_size=200 showed that running the full 7 workers (14 cores) caused
# an OOM kill - each worker carries its own copy of loaded packages, the
# compiled Stan model, and batch results in memory, and that adds up fast
# at higher concurrency. Capping to 3 workers (6 cores) was confirmed stable
# in testing. Raise this only after increasing #SBATCH --mem accordingly
# and re-testing at the target batch_size.
n_workers <- min(n_workers, 12)

use_multicore <- parallelly::supportsMulticore()
strategy <- if (use_multicore) future::multicore else future::multisession
future::plan(strategy, workers = n_workers)

options(SBC.min_chunk_size = 5)
cat(sprintf(
  "Detected %d available cores, running %d parallel SBC workers x %d chains/fit via %s (%d cores used of %d available)\n",
  n_cores_total, n_workers, chains_per_fit, if (use_multicore) "multicore" else "multisession",
  n_workers * chains_per_fit, cores_available
))

# define functions
probit_regression <- function(x, alpha, beta, lambda) {
  lambda /2  + (1 - lambda) * (0.5 + 0.5 * pracma::erf((x - alpha) / (beta * sqrt(2))))
}

inv_logit <- function(x) {
  y = 1 / (1 + exp(-x))
  return(y)
}

# ---------------------------------------------------------------------------
# Hierarchical data-generating process
#
# Draws the group-level hyperparameters (mu_*, sigma_*) from their priors,
# then draws each agent's individual parameters (alpha, b_log, lambda_logit)
# from those hyperparameters, then simulates trial-level HRD data for every
# agent. This replaces the flat (non-hierarchical) generate_group_dataset().
#
# NOTE: sigma_alpha and sigma_b_log are now drawn as half-normal with both a
# location AND scale (prior_sigma_alpha_loc/scale, prior_sigma_b_log_loc/scale),
# rather than a plain half-normal centered at 0. sigma_lambda_logit remains
# exponential(rate). Adjust if your Stan model uses a different prior family.
# ---------------------------------------------------------------------------
simulate_hierarchical_hrd_data <- function(
    n_agent = 60,
    n_trials = 40,
    prior_mu_alpha_loc = -10,
    prior_mu_alpha_scale = 7,
    prior_sigma_alpha_loc = 10,
    prior_sigma_alpha_scale = 3,
    prior_mu_b_log_loc = 2.5,
    prior_mu_b_log_scale = 0.2,
    prior_sigma_b_log_loc = 0.4,
    prior_sigma_b_log_scale = 0.1,
    prior_mu_lambda_logit_loc = -3,
    prior_mu_lambda_logit_scale = 0.4,
    prior_sigma_lambda_logit_loc = 0.1,
    prior_sigma_lambda_logit_scale = 0.5) {

  # 1. Draw group-level hyperparameters from their priors
  mu_alpha    <- rnorm(1, prior_mu_alpha_loc, prior_mu_alpha_scale)
  sigma_alpha <- abs(rnorm(1, prior_sigma_alpha_loc, prior_sigma_alpha_scale))

  mu_b_log    <- rnorm(1, prior_mu_b_log_loc, prior_mu_b_log_scale)
  sigma_b_log <- abs(rnorm(1, prior_sigma_b_log_loc, prior_sigma_b_log_scale))

  mu_lambda_logit    <- rnorm(1, prior_mu_lambda_logit_loc, prior_mu_lambda_logit_scale)
  sigma_lambda_logit <- abs(rnorm(1, prior_sigma_lambda_logit_loc, prior_sigma_lambda_logit_scale))

  # 2. Draw agent-level (subject-level) parameters from the hyperparameters
  alpha_agent        <- rnorm(n_agent, mu_alpha, sigma_alpha)
  b_log_agent         <- rnorm(n_agent, mu_b_log, sigma_b_log)
  beta_agent          <- exp(b_log_agent)
  lambda_logit_agent  <- rnorm(n_agent, mu_lambda_logit, sigma_lambda_logit)
  lambda_agent        <- inv_logit(lambda_logit_agent)

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
    prior_sigma_alpha_loc         =  10,
    prior_sigma_alpha_scale       =  3,
    prior_mu_b_log_loc            =  2.5,    # exp(2.5) ~ 12 BPM slope
    prior_mu_b_log_scale          =  0.2,
    prior_sigma_b_log_loc         =  0.4,
    prior_sigma_b_log_scale       =  0.1,
    prior_mu_lambda_logit_loc     = -3,      # 0.5 * inv_logit(-3) ~ 0.024
    prior_mu_lambda_logit_scale   =  0.4,
    prior_sigma_lambda_logit_loc  = 0.1,
    prior_sigma_lambda_logit_scale =  0.5,    # E[sigma_lambda_logit] = 0.4
    run_diagnostics = 0L # not needed now
  )

  # 5. Return the exact structure required by the SBC package
  list(
    variables = list(mu_alpha = mu_alpha_true,
                      sigma_alpha = sigma_alpha_true,
                      mu_b_log = mu_b_log_true,
                      sigma_b_log = sigma_b_log_true,
                      mu_lambda_logit = mu_lambda_logit_true,
                      sigma_lambda_logit = sigma_lambda_logit_true#,
                     #alpha = sim_data$alpha # to get subject-level parameters,
                     # beta = sim_data$beta # osv 
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
  chains = chains_per_fit,
  parallel_chains = chains_per_fit,
  iter_warmup = 2000,
  iter_sampling = n_iter,
  refresh = 0,
  adapt_delta = 0.95,
  max_treedepth = 10
)

# Register the generator function
generator <- SBC::SBC_generator_function(generate_sbc_dataset, n_agent = S, n_trials = N)

# ---------------------------------------------------------------------------
# BATCHED, CHECKPOINTED SBC RUN
#
# Runs n_batches batches of batch_size simulations each. Each batch's
# datasets + results are saved to disk immediately after computing, so a
# crash/timeout at any point only loses the batch currently in progress -
# not the whole run. Batches whose checkpoint file already exists AND whose
# recorded metadata genuinely matches the current config are skipped, so
# resubmitting the same job after a failure resumes rather than starting
# over - without risking silently reusing a stale/mismatched file.
# ---------------------------------------------------------------------------
cat(sprintf("Run SBC: %d batches x %d sims = %d total sims with %d iterations each and %d thinning of chains\n", n_batches, batch_size, n_sims, n_iter, n_thin))

t_start_total <- proc.time()

for (batch_i in seq_len(n_batches)) {

  batch_file <- file.path(batch_dir, sprintf("%s_batch%03d.rds", run_id, batch_i))

  # A batch is only trusted as "already done" if the file exists AND its
  # recorded metadata genuinely matches the current run config (S, N,
  # batch_index, and the number of sims it actually contains). This guards
  # against stale/partial files silently being reused just because a
  # filename happened to match.
  skip_this_batch <- FALSE
  if (file.exists(batch_file)) {
    existing <- tryCatch(readRDS(batch_file), error = function(e) NULL)
    n_sims_in_file <- tryCatch(nrow(existing$datasets$variables), error = function(e) NA)

    if (is.null(existing)) {
      cat(sprintf("[batch %d/%d] existing file is unreadable/corrupt - will re-run: %s\n",
                   batch_i, n_batches, batch_file))
    } else if (!identical(existing$batch_index, batch_i) ||
               !identical(existing$S, S) ||
               !identical(existing$N, N) ||
               is.na(n_sims_in_file) || n_sims_in_file != batch_size) {
      cat(sprintf(
        "[batch %d/%d] existing file does not match current config (batch_index=%s, S=%s, N=%s, n_sims=%s) - will re-run: %s\n",
        batch_i, n_batches, existing$batch_index, existing$S, existing$N, n_sims_in_file, batch_file
      ))
    } else {
      cat(sprintf("[batch %d/%d] verified complete (%d sims), skipping: %s\n",
                   batch_i, n_batches, n_sims_in_file, batch_file))
      skip_this_batch <- TRUE
    }
  }

  if (skip_this_batch) next

  cat(sprintf("[batch %d/%d] starting %d sims...\n", batch_i, n_batches, batch_size))
  t_start_batch <- proc.time()

  # Use a distinct, reproducible seed per batch so re-running a skipped batch
  # (or rerunning the whole thing later) generates the same data again.
  set.seed(1000 + batch_i)

  batch_datasets <- SBC::generate_datasets(generator, n_sims = batch_size)
  batch_results  <- SBC::compute_SBC(batch_datasets, backend, keep_fits = FALSE, thin_ranks = n_thin)

  elapsed_batch_sec <- (proc.time() - t_start_batch)[["elapsed"]]
  cat(sprintf("[batch %d/%d] finished in %.1f min\n", batch_i, n_batches, elapsed_batch_sec / 60))

  # Save this batch immediately - this is the fail-safe step. Even if a
  # later batch (or the final combine step) fails, this batch's work is
  # already safely on disk.
  saveRDS(
    list(
      datasets = batch_datasets,
      results = batch_results,
      batch_index = batch_i,
      S = S,
      N = N
    ),
    batch_file
  )
  cat(sprintf("[batch %d/%d] saved to: %s\n", batch_i, n_batches, batch_file))
}

# Shut down parallel workers now that all batches are done, rather than
# leaving idle worker processes holding memory/cores.
future::plan(future::sequential)

# ---------------------------------------------------------------------------
# COMBINE ALL BATCHES
#
# Reload every batch checkpoint (whether computed just now or in an earlier,
# resumed run) and stitch the SBC_results objects together with
# SBC::bind_results(), which is designed specifically for combining SBC runs
# that were computed incrementally.
# ---------------------------------------------------------------------------
batch_files <- file.path(batch_dir, sprintf("%s_batch%03d.rds", run_id, seq_len(n_batches)))
missing_batches <- batch_files[!file.exists(batch_files)]

if (length(missing_batches) > 0) {
  stop(sprintf(
    "Cannot combine results: %d batch(es) are missing (did an earlier batch fail?). Missing: %s",
    length(missing_batches), paste(basename(missing_batches), collapse = ", ")
  ))
}

cat("All batches present. Combining results incrementally...\n")

# Combine batches one at a time instead of loading all n_batches files into
# memory simultaneously. Each batch is read, folded into a running combined
# result via bind_results(), then discarded before the next one is loaded -
# so peak memory only ever holds ~2 batches' worth of data (the running
# total + the one just read), not all of them at once. This was a real
# memory bottleneck at n_batches=10: loading every batch's full datasets +
# results into one big list at once could rival or exceed the memory used
# by the parallel sampling itself.
sbc_results_hier <- NULL
batch_data_summaries <- vector("list", length(batch_files))

for (i in seq_along(batch_files)) {
  batch_obj <- readRDS(batch_files[i])

  if (is.null(sbc_results_hier)) {
    sbc_results_hier <- batch_obj$results
  } else {
    sbc_results_hier <- SBC::bind_results(sbc_results_hier, batch_obj$results)
  }

  # We deliberately do NOT keep the full datasets (simulated data + fits)
  # for every batch in memory or in the final combined file - with 2000
  # sims across 10 batches this can be large, and it's already safely on
  # disk in each per-batch checkpoint file (batch_dir) if you ever need to
  # go back and inspect a specific batch's raw simulated data. We just keep
  # a lightweight summary here (batch index + n sims) for reference.
  batch_data_summaries[[i]] <- list(
    batch_index = batch_obj$batch_index,
    batch_file = batch_files[i],
    n_sims = tryCatch(nrow(batch_obj$datasets$variables), error = function(e) NA)
  )

  rm(batch_obj)
  gc(full = TRUE)

  cat(sprintf("[combine %d/%d] folded in, memory freed\n", i, length(batch_files)))
}

elapsed_sec_total <- (proc.time() - t_start_total)[["elapsed"]]
elapsed_min_total <- floor(elapsed_sec_total / 60)
elapsed_rem_sec_total <- round(elapsed_sec_total %% 60)

# build final combined output path
timestamp_str <- format(Sys.time(), "%y%m%d%H%M")
sbc_filename <- sprintf("%s_%s_%diter_%dthin_%dsims_%dsubj_%dtrials_%dm_%ds_%s_COMBINED.rds",
                        timestamp_str, variable_name, n_iter, n_thin, n_sims, S, N,
                        elapsed_min_total, elapsed_rem_sec_total, prior_type)
sbc_path <- file.path(stan_results_dir, sbc_filename)

# save combined results. Full per-sim datasets are intentionally not
# duplicated here - they remain available in each batch's own checkpoint
# file under batch_dir (see batch_data_summaries below for the exact paths),
# so nothing is lost, it's just not all loaded into RAM/this file at once.
saveRDS(
  list(
    results = sbc_results_hier,
    batch_data_summaries = batch_data_summaries,
    batch_dir = batch_dir,
    S = S,
    N = N,
    n_batches = n_batches,
    batch_size = batch_size,
    n_iter = n_iter,
    n_thin = n_thin
  ),
  sbc_path
)
cat(sprintf("Saved combined results (%d sims total) to: %s\n", n_sims, sbc_path))
cat(sprintf("(Full per-batch datasets remain available in: %s)\n", batch_dir))

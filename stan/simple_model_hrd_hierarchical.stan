// =============================================================================
//  Hierarchical psychometric model for the Heart Rate Discrimination (HRD) task
// =============================================================================
//
//  On each trial t, participant j sees a stimulus delta-BPM and decides whether
//  their own heart rate is FASTER (choice = 1) or SLOWER (choice = 0) than the
//  presented target. We model the choice with a probit psychometric function
//  plus a per-subject lapse rate:
//
//      theta_{tj} = lambda_j + (1 - 2*lambda_j) * Phi( (dBPM_t - alpha_j) / beta_j )
//      choice_{tj} ~ Bernoulli(theta_{tj})
//
//  Parameter interpretation:
//      alpha_j  - threshold (BPM):  PSE; the dBPM where the subject is 50/50.
//      beta_j   - slope (BPM):      internal noise / inverse sensitivity.
//      lambda_j - lapse rate:       prob. of attentional/key-press lapse,
//                                   constrained to [0, 0.5].
//
//  All subject-level parameters use a non-centered parameterization (NCP):
//      raw z_j  ~ Normal(0, 1)
//      theta_j  = mu + sigma * z_j   (then a link function for positivity/bounds)
//
//  Note: prior hyperparameters live in the `data` block. This is deliberate:
//      (1) the R simulator and Stan model share a single source of truth, so
//          you cannot accidentally invert R's rexp(rate = 1/x) against Stan's
//          exponential(rate = x) -- a class of bug that can pass casual review
//          but silently breaks SBC and prior–posterior plots.
//      (2) Phase 5 prior sensitivity sweeps no longer require recompilation.
//
// =============================================================================

data {
  // ----- experimental design -----
  int<lower=1> N;                           // total number of trials
  int<lower=1> S;                           // number of subjects
  vector[N] dBPM;                           // stimulus on each trial (BPM)
  array[N] int<lower=0, upper=1> choice;    // binary response on each trial
  array[N] int<lower=1, upper=S> subj;      // subject index for each trial

  // ----- prior hyperparameters (passed in from R) -----
  // Suggested defaults (based on the doc, but some changes)
  //  set these in your stan_data list, NOT here):
  //   prior_mu_alpha_loc            = -10     # HRD subjects systematically
  //   prior_mu_alpha_scale          =  7      #   underestimate -> negative mean
  //   prior_sigma_alpha_rate        =  0.1    # E[sigma_alpha] = 1/rate = 10 BPM
  //
  //   prior_mu_b_log_loc            =  2.5    # exp(2.5) ~ 12 BPM slope
  //   prior_mu_b_log_scale          =  0.4
  //   prior_sigma_b_log_rate        =  2.5    # E[sigma_b_log] = 1/2.5 = 0.4
  //
  //   prior_mu_lambda_logit_loc     = -3      # 0.5 * inv_logit(-3) ~ 0.024
  //   prior_mu_lambda_logit_scale   =  0.4
  //   prior_sigma_lambda_logit_rate =  2.5    # E[sigma_lambda_logit] = 0.4
  //
  // IMPORTANT: in R, exponential samples are rexp(n, rate = X) where rate=X
  // gives MEAN 1/X. In Stan, exponential(X) ALSO uses rate parameterization,
  // mean 1/X. Match them by passing the SAME rate value to both. The previous
  // version of this code had `rexp(1, rate = 1/10)` (mean 10) paired with
  // `exponential(10)` (mean 0.1) -- a 100x mismatch that silently propagated.

  real prior_mu_alpha_loc;
  real<lower=0> prior_mu_alpha_scale;
    real<lower=0> prior_sigma_alpha_loc;
  real<lower=0> prior_sigma_alpha_scale;

  real prior_mu_b_log_loc;
  real<lower=0> prior_mu_b_log_scale;
  real<lower=0> prior_sigma_b_log_loc;
  real<lower=0> prior_sigma_b_log_scale;

  real prior_mu_lambda_logit_loc;
  real<lower=0> prior_mu_lambda_logit_scale;
  real<lower=0> prior_sigma_lambda_logit_loc;
  real<lower=0> prior_sigma_lambda_logit_scale;
  
  int<lower=0, upper=1> run_diagnostics;  // 1 = full diagnostics; 0 = simulation based calibration only
}

parameters {
  // ----- population (group-level) parameters -----
  real mu_alpha;                  // population mean of threshold (BPM)
  real<lower=0> sigma_alpha;      // population SD of threshold (BPM)

  real mu_b_log;                  // population mean of LOG slope
  real<lower=0> sigma_b_log;      // population SD of log slope

  real mu_lambda_logit;           // population mean of LOGIT lapse
  real<lower=0> sigma_lambda_logit; // population SD of logit lapse

  // ----- subject-level RAW (unit-normal) parameters -----
  // These are the actual quantities the HMC sampler explores. The on-scale
  // subject parameters (alpha[s], beta[s], lambda[s]) are deterministic
  // functions of these raws and the population means/SDs, computed below.
  // The renaming `_z` (formerly `_subj`) makes the NCP intent explicit:
  // these are z-scores, not subject-level estimates on the natural scale.
  vector[S] alpha_z;
  vector[S] b_log_z;
  vector[S] lambda_logit_z;
}

transformed parameters {
  vector[S] alpha;            // per-subject threshold (BPM)
  vector[S] beta;             // per-subject slope     (BPM, strictly > 0)
  vector[S] lambda;           // per-subject lapse rate ([0, 0.5])

  // ----- Non-centered parameterization -----
  // Why NCP? With a centered parameterization
  //     alpha[s] ~ Normal(mu_alpha, sigma_alpha)
  // sampler geometry develops "Neal's funnel" as sigma_alpha -> 0: the joint
  // distribution becomes pinched and HMC can't traverse it without divergences.
  // NCP fixes this by sampling z ~ N(0,1) (a fixed, well-conditioned space)
  // and reconstructing alpha[s] = mu_alpha + sigma_alpha * z[s].
  //
  // Tradeoff: when data are very informative, NCP can suffer a "reverse funnel"
  // (the data force a tight posterior on alpha[s], which through the inverse
  // map forces a stretched posterior on z[s]). With ~40 trials per subject in
  // the HRD task, NCP is the right default. If you ever see divergences AFTER
  // increasing trial count substantially, run mcmc_pairs() on (mu, sigma, z)
  // and consider switching back to CP.
  for (s in 1:S) {
    alpha[s]  = mu_alpha + sigma_alpha * alpha_z[s];

    // Slope must be positive -> sample log slope, exponentiate. Equivalent to
    // a lognormal prior on beta[s] with location mu_b_log and scale sigma_b_log.
    beta[s]   = exp(mu_b_log + sigma_b_log * b_log_z[s]);

    // Lapse must be in [0, 0.5]. We use 0.5 * inv_logit(...) rather than
    // inv_logit(...) because chance performance in a 2AFC task is 0.5 -- a
    // pure lapse trial cannot produce systematic bias above chance. Cap is
    // generous; empirical lapse rates are typically <0.05.
    lambda[s] = inv_logit(mu_lambda_logit + sigma_lambda_logit * lambda_logit_z[s]);
  }
}

model {
  // ----- population priors -----
  // Using `target += ..._lpdf(...)` form rather than `~`. The two are
  // equivalent here, but `target +=` is the form you need if you ever want
  // to manually power-scale priors for sensitivity analysis.
  target += normal_lpdf(mu_alpha           | prior_mu_alpha_loc,
                                             prior_mu_alpha_scale);
  target += normal_lpdf(sigma_alpha        | prior_sigma_alpha_loc, 
                                             prior_sigma_alpha_scale);


  target += normal_lpdf(mu_b_log           | prior_mu_b_log_loc,
                                            prior_mu_b_log_scale);
  target += normal_lpdf(sigma_b_log        | prior_sigma_b_log_loc, 
                                             prior_sigma_b_log_scale);
  target += normal_lpdf(mu_lambda_logit    | prior_mu_lambda_logit_loc,
                                             prior_mu_lambda_logit_scale);
  target += normal_lpdf(sigma_lambda_logit | prior_sigma_lambda_logit_loc,
                                             prior_sigma_lambda_logit_scale);

  // ----- subject-level raw priors -----
  // The heart of the NCP: raw z-scores live in a unit-normal space that
  // doesn't depend on the population scale. Vectorized form is faster and
  // more idiomatic than looping. `std_normal()` is equivalent to
  // `normal(0,1)` but skips the location/scale arithmetic internally.
  target += std_normal_lpdf(alpha_z);
  target += std_normal_lpdf(b_log_z);
  target += std_normal_lpdf(lambda_logit_z);

  // ----- likelihood -----
  // Fully vectorized: we gather per-trial subject parameters via integer-array
  // indexing (`alpha[subj]` with `subj` an array[N] int) and compute theta in
  // a single elementwise expression. This avoids a per-trial scalar loop,
  // which Stan's autodiff would otherwise turn into N separate AD graph nodes
  // -- a major hidden cost at N >= 2400.
  //
  // `Phi_approx()` is a fast approximation to the standard normal CDF with
  // worst-case error ~1e-3, well below the ~1/sqrt(40) ~ 0.16 trial-level
  // binomial noise floor here. Replaces the older `0.5 + 0.5 * erf(z/sqrt(2))`
  // form, which is mathematically identical to Phi(z) but slower in Stan.
  //
  // We use bernoulli_lpmf (probability scale), NOT bernoulli_logit_lpmf,
  // because theta is a lapse–signal MIXTURE, not the inverse logit of a
  // linear predictor -- it has no closed-form logit. The lapse term keeps
  // theta strictly inside (lambda, 1 - lambda), so theta is never exactly
  // 0 or 1 and bernoulli_lpmf is numerically safe.
  {
    vector[N] a_n = alpha[subj];   // gather: per-trial threshold
    vector[N] b_n = beta[subj];    // gather: per-trial slope
    vector[N] l_n = lambda[subj];  // gather: per-trial lapse

    vector[N] theta = l_n /2 + (1 - l_n)
                            .* (0.5 + 0.5 * erf(dBPM - a_n) ./ (b_n * sqrt(2)));

    target += bernoulli_lpmf(choice | theta);
  }
}

generated quantities {
  // ===========================================================================
  // Everything in this block runs ONCE PER POSTERIOR DRAW. Use it for:
  //   - prior draws (independent of the posterior; used in prior–posterior
  //     overlay plots and for sanity-checking the prior the model "sees")
  //   - prior predictive simulations (Phase 2 of the validation battery)
  //   - posterior predictive simulations (Phase 3)
  //   - pointwise log-likelihood for PSIS-LOO and LOO-PIT (Phase 4 model comp.)
  // ===========================================================================

  // ----- (1) prior draws: independent samples from the prior -----
  // These DO NOT depend on the data and DO NOT change as the posterior
  // concentrates. Plotting them alongside the posterior shows information
  // gain. Drawing them in `generated quantities` (one per iteration) gives
  // you exactly the right number of samples to overlay.
  real mu_alpha_prior            = normal_rng(prior_mu_alpha_loc,
                                              prior_mu_alpha_scale);
  real sigma_alpha_prior  = abs(normal_rng(prior_sigma_alpha_loc, 
                                           prior_sigma_alpha_scale));
  
  real mu_b_log_prior            = normal_rng(prior_mu_b_log_loc,
                                              prior_mu_b_log_scale);
  real sigma_b_log_prior        = abs(normal_rng(prior_sigma_b_log_loc, 
                                                 prior_sigma_b_log_scale));
  real mu_lambda_logit_prior     = normal_rng(prior_mu_lambda_logit_loc,
                                              prior_mu_lambda_logit_scale);
  real sigma_lambda_logit_prior  = abs(normal_rng(prior_sigma_lambda_logit_loc,
                                                  prior_sigma_lambda_logit_scale));

  // ----- (2) subject-level prior draws -----
  // For each posterior draw, we generate a fresh "prior population" of S
  // subjects from the freshly drawn population priors above. This gives the
  // correct hierarchical prior predictive distribution.
  vector[S] alpha_prior;
  vector[S] beta_prior;
  vector[S] lambda_prior;

  for (s in 1:S) {
    real az = normal_rng(0, 1);
    real bz = normal_rng(0, 1);
    real lz = normal_rng(0, 1);

    alpha_prior[s]  = mu_alpha_prior + sigma_alpha_prior * az;
    beta_prior[s]   = exp(mu_b_log_prior + sigma_b_log_prior * bz);
    lambda_prior[s] = inv_logit(mu_lambda_logit_prior
                                      + sigma_lambda_logit_prior * lz);
  }

  // ----- (3) per-trial quantities -----
  vector[N] theta_prior_p;   // prior predictive choice probability per trial
  array[N] int choice_prior_pred; //
  int choice_prior_sum;
  vector[N] theta_p;         // posterior predictive choice probability per trial
  array[N] int choice_pred;
  int choice_sum;
  vector[N] log_lik;         // pointwise log-likelihood (for loo::loo)
  array[N] int choice_rep;   // posterior predictive REPLICATE choices
  int s;

  // Why include choice_rep? Phase 3 (posterior predictive checks) needs more
  // than just theta_p. Some misspecifications only show up in the SEQUENTIAL
  // structure of replicates -- e.g. autocorrelation, max run length, perfect-
  // streak counts. theta_p alone gives you marginal rates; choice_rep gives
  // you full draws you can compute any statistic on.

  real lprior_mu_alpha;
  real lprior_mu_b_log;
  real lprior_mu_lambda_logit;
  real lprior_sigma_alpha;
  real lprior_sigma_b_log;
  real lprior_sigma_lambda_logit;
  vector[N] theta;
  array[N] int y_rep;
  

  if (run_diagnostics) {
    for (n in 1:N) {
      s = subj[n];
      // prior predictive theta (using PRIOR-drawn subject params)
      theta_prior_p[n] = lambda_prior[s]
                         + (1 - 2 * lambda_prior[s])
                           * Phi_approx((dBPM[n] - alpha_prior[s]) / beta_prior[s]);
      choice_prior_pred[n] = bernoulli_rng(theta_prior_p[n]); 

      // posterior predictive theta (using POSTERIOR subject params)
      theta_p[n] = lambda[s] / 2
                   + (1 - lambda[s])
                     * (0.5+0.5*erf((dBPM[n] - alpha[s]) / (beta[s]*sqrt(2))));
      choice_pred[n] = bernoulli_rng(theta_p[n]);
  
      // pointwise log-likelihood. Required by loo::loo() for PSIS-LOO,
      // LOO-PIT, and loo_compare(). Cheap to compute, costs you nothing if
      // you never use it; expensive to add later (requires refit).
      
      theta[n] = lambda[s] / 2 + (1 - lambda[s]) * (0.5+0.5*erf((dBPM[n] - alpha[s]) / (beta[s]*sqrt(2))));
      y_rep[n] = bernoulli_rng(theta[n]);
      log_lik[n] = bernoulli_lpmf(choice[n] | theta_p[n]);
  
      // posterior predictive checks
      // bayesplot::ppc_*() functions consume this directly.
      choice_rep[n] = bernoulli_rng(theta_p[n]);
    }
    // summed variables
    choice_prior_sum = sum(choice_prior_pred); // added 29-06
    choice_sum = sum(choice_pred);

    lprior_mu_alpha = normal_lpdf(mu_alpha | prior_mu_alpha_loc, 
                                             prior_mu_alpha_scale);
    lprior_sigma_alpha = normal_lpdf(sigma_alpha | prior_sigma_alpha_loc, 
                                                   prior_sigma_alpha_scale);
    lprior_mu_b_log = normal_lpdf(mu_b_log | prior_mu_b_log_loc, 
                                             prior_mu_b_log_scale);
    lprior_sigma_b_log = normal_lpdf(sigma_b_log | prior_sigma_b_log_loc, 
                                                   prior_sigma_b_log_scale);
    lprior_mu_lambda_logit = normal_lpdf(mu_lambda_logit | prior_mu_lambda_logit_loc, 
                                                          prior_mu_lambda_logit_scale);
    lprior_sigma_lambda_logit = normal_lpdf(sigma_lambda_logit | prior_sigma_lambda_logit_loc,
                                                                 prior_sigma_lambda_logit_scale);
  }
}

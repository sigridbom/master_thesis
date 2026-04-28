//
// This Stan program defines a simple hierarchical model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;                        // n of trials
  int<lower=0> S;                        // number of participants
  vector[N] dBPM;                        // stimuli, delta beats pr minute
  array[N] int<lower=0, upper=1> choice; // decision on every trial
  array[N] int<lower=1, upper=S> subj;   // participant index for each trial 
  
}

// The parameters accepted by the model. Our model accepts alpha (logit), beta (log) and lambda (logit)
parameters {
  // population level paramters
  real mu_alpha;              // threshold mean 
  real <lower=0> sigma_alpha; // threshold SD
  
  real mu_b_log;              // slope mean
  real <lower=0>sigma_b_log;           // slope SD
  real mu_lambda_logit;          // lapse rate mean
  real <lower=0> sigma_lambda_logit; // lapse rate SD
  
  
  //subject level paramters
  vector[S] alpha_subj;
  vector[S] b_log_subj;
  vector[S] lambda_logit_subj;
}

// the transformations will bring the parameters back to the meaningful range
// because the ones in the parameter section need to be unbounded for better posterior
// exploration
transformed parameters{
  // subject-specific transformed parameters
    vector[S] alpha;
    vector[S] beta;
    vector[S] lambda;

// non-centeret parameterization?
    for (s in 1:S) {
      alpha[s] = mu_alpha + sigma_alpha * alpha_subj[s];
      beta[s]  = exp(mu_b_log + sigma_b_log * b_log_subj[s]);
      lambda[s] = 0.5 * inv_logit(mu_lambda_logit + sigma_lambda_logit * lambda_logit_subj[s]);
  }
}



// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  vector[N] theta; // stan doesn't allow constraints of parameters in model block (removed 0-1) - should this be in the data section?

  // population-level priors
  target += normal_lpdf(mu_alpha| -10, 7); 
  target += exponential_lpdf(sigma_alpha|10);
  
  target += normal_lpdf(mu_b_log | 2.5, .4); 
  target += exponential_lpdf(sigma_b_log | 0.4);
  
  target += normal_lpdf(mu_lambda_logit | -3, .4);
  target += exponential_lpdf(sigma_lambda_logit | 0.4);

  // subject-level priors - er det rigtigt???
  alpha_subj        ~ normal(0, 1);
  b_log_subj        ~ normal(0, 1);
  lambda_logit_subj ~ normal(0, 1);

  // likelihood
  for (n in 1:N) {
    int s = subj[n];
    real a = alpha[s];
    real b = beta[s];
    real l = lambda[s];

    theta[n] = l + (1 - 2*l) *
               (0.5 + 0.5 * erf((dBPM[n] - a)/(b*sqrt(2))));
  }

  target += bernoulli_lpmf(choice | theta);

  // likelihood
  // old theta = lambda + (1-2*lambda)*(0.5+0.5*erf((dBPM - alpha)/(beta*sqrt(2))));

}

generated quantities{
  // generate priors for visualization

  // --- POPULATION LEVEL PRIORS ---
  real mu_alpha_prior         = normal_rng(-10, 7);
  real sigma_alpha_prior      = exponential_rng(10);
  real mu_b_log_prior         = normal_rng(2.5, 0.4);
  real sigma_b_log_prior      = exponential_rng(0.4);
  real mu_lambda_logit_prior  = normal_rng(-3, 0.4);
  real sigma_lambda_logit_prior = exponential_rng(0.4);

  // --- SUBJECT LEVEL PRIORS ---
  // sample one "prior subject" from the prior predictive population
  vector[S] alpha_prior;
  vector[S] beta_prior;
  vector[S] lambda_prior;

  for (s in 1:S) {
    real alpha_subj_prior        = normal_rng(0, 1);
    real b_log_subj_prior        = normal_rng(0, 1);
    real lambda_logit_subj_prior = normal_rng(0, 1);

    alpha_prior[s]  = mu_alpha_prior + sigma_alpha_prior * alpha_subj_prior;
    beta_prior[s]   = exp(mu_b_log_prior + sigma_b_log_prior * b_log_subj_prior);
    lambda_prior[s] = 0.5 * inv_logit(mu_lambda_logit_prior + sigma_lambda_logit_prior * lambda_logit_subj_prior);
  }

  // --- PRIOR PREDICTIVE THETA ---
  vector[N] theta_prior_p;

  for (n in 1:N) {
    int s = subj[n];
    theta_prior_p[n] = lambda_prior[s] + (1 - 2 * lambda_prior[s]) *
                       (0.5 + 0.5 * erf((dBPM[n] - alpha_prior[s]) / (beta_prior[s] * sqrt(2))));
  }

  // --- POSTERIOR PREDICTIVE THETA ---
  vector[N] theta_p;

  for (n in 1:N) {
    int s = subj[n];
    theta_p[n] = lambda[s] + (1 - 2 * lambda[s]) *
                 (0.5 + 0.5 * erf((dBPM[n] - alpha[s]) / (beta[s] * sqrt(2))));
  }
}

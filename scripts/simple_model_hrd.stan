//
// This Stan program defines a simple model, with a
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
  int<lower=0> N; // n of trials
  vector[N] dBPM; // stimuli, delta beats pr minute
  //vector[N] choice;
  array[N] int<lower=0, upper=1> choice; // decision on every trial
  
  //priors ?

}

// The parameters accepted by the model. Our model accepts alpha (logit), beta (log) and lambda (log)
parameters {
  real alpha;// threshold  
  real b_log; // slope, we might add an upper boundary
  real lambda_logit;// lapse rate, we might add an upper boundary
}

// the transformations will bring the parameters back to the meaningful range
// because the ones in the parameter section need to be unbounded for better posterior
// exploration
transformed parameters{
  real<lower=0> beta = exp(b_log);
  real<lower=0> lambda = 0.5*inv_logit(lambda_logit);//exp(lambda_log);
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  vector[N] theta; // stan doesn't allow constraints of parameters in model block (removed 0-1) - should this be in the data section?

  // priors
  target += normal_lpdf(alpha| -10, 1.5); 
  target += normal_lpdf(b_log | 2.5, .4); 
  target += normal_lpdf(lambda_logit | -3, .4);
  
  // likelihood
  theta = lambda + (1-2*lambda)*(0.5+0.5*erf((dBPM - alpha)/(beta*sqrt(2))));
  
  target += bernoulli_lpmf(choice | theta); // log probability mass function - discrete scale due to discrete outcome

  //prob = lambda + (1-2*lambda)*inv_logit((sim_data$deltaBPM - alpha)/beta)

}

generated quantities{
  // generate priors for visualization
  
  real alpha_prior;
  real b_log_prior;
  real lambda_logit_prior;

  alpha_prior = normal_rng(-10, 1.5);
  b_log_prior       = normal_rng(2.5, 0.4);
  lambda_logit_prior  = normal_rng(-3, 0.4);
  
  // trying non-informed priors - did not improve things
  //alpha_prior = beta(1,1);
  //b_log_prior = beta(1,1);
  //lambda_log_prior = beta(1,1);
  
  real beta_prior = exp(b_log_prior);
  real lambda_prior = exp(lambda_logit_prior);
  
  vector<lower=0, upper=1>[N] theta_prior_p;
  
  for (n in 1:N) {
    theta_prior_p[n] = lambda_prior + (1 - 2 * lambda_prior) *
                 (0.5+0.5*erf((dBPM[n] - alpha_prior) / (beta_prior*sqrt(2))));
  }
  
  // generate model predictions given the data for posterior predictive checks
  vector<lower=0, upper=1>[N] theta_p;
  
  for (n in 1:N) {
    theta_p[n] = lambda + (1 - 2 * lambda) *
                 (0.5+0.5*erf((dBPM[n] - alpha) / (beta*sqrt(2))));
  }
}  
  // generate model predictions given only the priors for prior predictive checks
  // generate log_likehood estimates for model comparison

  //test
 // vector[N] theta;
//  array[N] int y_rep;
//  vector[N] log_lik;

  // recompute choice probabilities
//  theta = lambda + (1 - 2 * lambda) *
//          Phi((dBPM - alpha) / beta);

//  for (n in 1:N) {
//    y_rep[n] = bernoulli_rng(theta[n]);
//    log_lik[n] = bernoulli_lpmf(choice[n] | theta[n]);
 // }

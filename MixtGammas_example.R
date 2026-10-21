# -----------------------------------------------------------------------------
# R Code for "Comments on: CliPS: How to identify cluster distributions in 
# Bayesian mixture models" (Ausín, M. C., TEST)
# 
# This script demonstrates the advantage of Gaussian Mixture Copula Models (GMCM)
# over standard Euclidean k-means for clustering MCMC parameter draws with 
# bounded supports and non-linear posterior correlations.
# -----------------------------------------------------------------------------

rm(list = ls())

# Load required packages
library("GMCM")
library("stats")

set.seed(2222)

# -----------------------------------------------------------------------------
# 1. GENERATE SYNTHETIC GAMMA MIXTURE DATA
# -----------------------------------------------------------------------------
# We simulate data from a 2-component Gamma mixture model.
n <- 200
p_true <- 0.4 # True mixing probability

# True parameters (shape 'a', rate 'b')
# Note: These parameters are strictly positive (bounded in (0, Inf))
a_true <- c(2.0, 6.0)
b_true <- c(1.0, 1.5)

# Generate latent component labels
z_true <- rbinom(n, 1, p_true) + 1

# Generate the observed data 'x' conditional on the latent labels
x <- ifelse(z_true == 1, 
            rgamma(n, shape = a_true[1], rate = b_true[1]), 
            rgamma(n, shape = a_true[2], rate = b_true[2]))

# -----------------------------------------------------------------------------
# 2. METROPOLIS-WITHIN-GIBBS SAMPLER FOR GAMMA MIXTURE
# -----------------------------------------------------------------------------
K <- 2
n_iter <- 4000
burn_in <- 2000

# Storage for MCMC posterior draws
a_save <- matrix(NA, nrow = n_iter, ncol = K)
b_save <- matrix(NA, nrow = n_iter, ncol = K)

# Initialize MCMC chains
a_curr <- c(1.5, 5.0)
b_curr <- c(0.8, 2.0)
pi_curr <- c(0.5, 0.5)
z_curr <- sample(1:K, n, replace = TRUE)

# Proposal variance for the Adaptive Random Walk Metropolis-Hastings step
prop_sd <- 0.08

for (iter in 1:(burn_in + n_iter)) {
  
  # --- Step A: Update latent component allocations (z) ---
  prob_z <- matrix(0, nrow = n, ncol = K)
  for (k in 1:K) {
    prob_z[, k] <- pi_curr[k] * dgamma(x, shape = a_curr[k], rate = b_curr[k])
  }
  # Normalize probabilities
  prob_z <- prob_z / rowSums(prob_z)
  for (i in 1:n) {
    z_curr[i] <- sample(1:K, size = 1, prob = prob_z[i, ])
  }
  
  # --- Step B: Update mixture weights (pi) via Dirichlet posterior ---
  nk <- tabulate(z_curr, nbins = K)
  dir_draw <- rgamma(K, shape = 1 + nk, rate = 1)
  pi_curr <- dir_draw / sum(dir_draw)
  
  # --- Step C: Update Gamma Parameters (a, b) via RW-Metropolis ---
  for (k in 1:K) {
    xk <- x[z_curr == k]
    nk_k <- length(xk)
    
    if (nk_k > 2) {
      # Propose in log-scale to strictly respect the (0, infinity) bounds
      log_a_prop <- rnorm(1, mean = log(a_curr[k]), sd = prop_sd)
      log_b_prop <- rnorm(1, mean = log(b_curr[k]), sd = prop_sd)
      
      a_prop <- exp(log_a_prop)
      b_prop <- exp(log_b_prop)
      
      # Log posterior likelihood comparison (assuming Uninformative Exponential Priors)
      curr_ll <- sum(dgamma(xk, shape = a_curr[k], rate = b_curr[k], log = TRUE)) - a_curr[k] - b_curr[k]
      prop_ll <- sum(dgamma(xk, shape = a_prop, rate = b_prop, log = TRUE)) - a_prop - b_prop
      
      # MH acceptance ratio (Jacobian adjustments for log-transformation cancel out here)
      log_acc <- prop_ll - curr_ll + (log_a_prop + log_b_prop) - (log(a_curr[k]) + log(b_curr[k]))
      
      # Accept/Reject
      if (log(runif(1)) < log_acc) {
        a_curr[k] <- a_prop
        b_curr[k] <- b_prop
      }
    }
  }
  
  # Save draws after burn-in phase
  if (iter > burn_in) {
    idx <- iter - burn_in
    a_save[idx, ] <- a_curr
    b_save[idx, ] <- b_curr
  }
}

# -----------------------------------------------------------------------------
# 3. CLUSTERING IN PARAMETER SPACE (Step 3 of CliPS)
# -----------------------------------------------------------------------------

# Pool all MCMC draws across iterations and components
a_vec <- as.vector(a_save)
b_vec <- as.vector(b_save)
draws_raw <- cbind(a_vec, b_vec)

# --- A. Standard Euclidean k-means on Raw Bounded Parameters ---
# This method struggles with the skewed geometries and bounded supports
km_res <- kmeans(draws_raw, centers = K, nstart = 20)
kmeans_labels <- km_res$cluster

# Plot initial k-means allocation
plot(a_vec, b_vec, col = kmeans_labels, main = "Posterior samples (k-means)", 
     xlab = expression(alpha), ylab = expression(beta), pch = 20, cex = 0.5)


# --- B. Proposed GMCM Copula Parameter Clustering ---
# Step B1: Empirical Probability Integral Transform (PIT)
# Maps draws to the unit square [0,1]^2 based strictly on ranks
u <- rank(a_vec) / (length(a_vec) + 1)
v <- rank(b_vec) / (length(b_vec) + 1)
ranked_data <- cbind(u, v)

# Step B2: Initialize GMCM parameters mapping pseudo-data to Gaussian space
z_data <- qnorm(ranked_data)
init_pie <- as.numeric(table(kmeans_labels) / length(kmeans_labels))
names(init_pie) <- paste0("pie", 1:K)

init_mu <- lapply(1:K, function(k) colMeans(z_data[kmeans_labels == k, , drop = FALSE]))
names(init_mu) <- paste0("comp", 1:K)

init_sigma <- lapply(1:K, function(k) cov(z_data[kmeans_labels == k, , drop = FALSE]))
names(init_sigma) <- paste0("comp", 1:K)

# Remove dimension names to comply with GMCM package strict formatting
init_sigma <- lapply(init_sigma, function(mat) { dimnames(mat) <- NULL; mat })

# Construct the custom 'theta' class object
init_theta <- list(m = K, d = 2, pie = init_pie, mu = init_mu, sigma = init_sigma)
class(init_theta) <- "theta"

# Step B3: Fit the Gaussian Mixture Copula Model via Pseudo-EM Algorithm
mle_theta <- fit.full.GMCM(u = ranked_data, theta = init_theta, method = "PEM", max.ite = 1000, eps = 1e-4)

# Extract GMCM cluster allocations
gmcm_labels <- apply(get.prob(ranked_data, theta = mle_theta), 1, which.max)

# -----------------------------------------------------------------------------
# 4. NON-PERMUTATION RATE EVALUATION
# -----------------------------------------------------------------------------

# Function to compute the non-permutation rate (v)
# v is the proportion of MCMC iterations that do not form a valid label permutation
compute_v <- function(labels, M, K_hat) {
  alloc_mat <- matrix(labels, nrow = M, ncol = K_hat, byrow = FALSE)
  valid <- apply(alloc_mat, 1, function(row) length(unique(row)) == K_hat)
  return(1 - mean(valid))
}

v_kmeans <- compute_v(kmeans_labels, n_iter, K)
v_gmcm   <- compute_v(gmcm_labels, n_iter, K)

cat("\n--- Non-Permutation Rate (v) for Gamma Mixture ---\n")
cat("Standard Euclidean k-means v :", round(v_kmeans, 4), "\n")
cat("Copula-Based GMCM v         :", round(v_gmcm, 4), "\n")

# -----------------------------------------------------------------------------
# 5. VISUALIZATION OF RESULTS
# -----------------------------------------------------------------------------

# Plot 1: Raw posterior samples colored by GMCM allocation
plot(a_vec, b_vec, col = gmcm_labels, main = "Posterior samples", 
     xlab = expression(alpha), ylab = expression(beta), pch = 20, cex = 0.5)

# Plot 2: Empirical PIT (Pseudo-copula data)
plot(ranked_data, col = gmcm_labels, main = "GMCM pseudo-copula", 
     xlab = expression(u[1]), ylab = expression(u[2]), pch = 20, cex = 0.5)

# Retrieve final parameters to map back to latent Gaussian space
pie <- as.numeric(mle_theta$pie)
pie <- pie / sum(pie)
mu <- do.call(rbind, mle_theta$mu)
sigma <- mle_theta$sigma

# Extract marginal standard deviations
sd_mat <- do.call(rbind, lapply(sigma, function(S) {
  sqrt(diag(S))
}))

# Define Marginal CDF of the Gaussian mixture
F_mix <- function(x, j) {
  sum(pie * pnorm(x, mean = mu[, j], sd = sd_mat[, j]))
}

# Define Inverse marginal CDF via numerical root-finding
F_mix_inv <- function(p, j) {
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lo <- min(mu[, j] - 10 * sd_mat[, j])
  hi <- max(mu[, j] + 10 * sd_mat[, j])
  
  # Ensure the root is bracketed
  while (F_mix(lo, j) > p) lo <- lo - (hi - lo)
  while (F_mix(hi, j) < p) hi <- hi + (hi - lo)
  
  uniroot(
    function(x) F_mix(x, j) - p,
    interval = c(lo, hi),
    tol = 1e-10
  )$root
}

# Transform the uniform pseudo-data into the unbounded latent Gaussian space
uv <- cbind(u, v)
z <- cbind(
  z1 = vapply(uv[, 1], F_mix_inv, numeric(1), j = 1),
  z2 = vapply(uv[, 2], F_mix_inv, numeric(1), j = 2)
)

# Plot 3: Latent Gaussian mixture representation
plot(z, col = gmcm_labels, main = "Latent Gaussian mixture", 
     xlab = expression(z[1]), ylab = expression(z[2]), pch = 20, cex = 0.5)


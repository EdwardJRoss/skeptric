# Exact calculation
ci_median_extreme <- function(num_samples, nth_largest=0, nth_smallest=0) {
    stopifnot(nth_largest + nth_smallest < num_samples)
    # Number of ways all but the nth largest values could be bigger than median
    freq_larger <- sum(choose(num_samples, seq(0, nth_largest)))
    # Number of ways all but the nth smallest values could be bigger than median
    freq_smaller <- sum(choose(num_samples, seq(0, nth_smallest)))
    1 - (freq_larger + freq_smaller) / 2^num_samples
}

# Test calculation with random normal sample
sample_median_extreme <- function(sample_size, nth, num_sample=10000) {
    distribution <- rnorm
    distribution_median <- 0

    samples <- distribution(sample_size * num_sample)
    dim(samples) <- c(sample_size, num_sample)

    sorted_samples <- apply(samples, 2, sort)
    mean((sorted_samples[1+nth,] <= distribution_median) &
         (sorted_samples[sample_size-nth,] >= distribution_median))
}


# Calculate the number of samples at nth largest to reach at least ci
median_extreme_samples_at_ci <- function(nth, ci=0.9) {
    sample_size <- 2*nth + 1
    while(ci_median_extreme(sample_size, nth, nth) < ci) {
        sample_size <- sample_size + 1
    }
    sample_size
}

# Generate a table of confidences
ci_median_extreme_table <- function(n_max=10, ci=0.9) {
    nth_largest <- seq(0, n_max)
    sample_size <- vapply(nth_largest,
                          function(n) median_extreme_samples_at_ci(n, ci),
                          double(1))
    confidence_interval <- mapply(ci_median_extreme, sample_size, nth_largest, nth_largest)

    data.frame(nth_largest, sample_size, confidence_interval)
}

# Power depends a lot on the distribution

ci_median_extreme_table(10, 0.9)

ci_median_extreme_table(10, 0.75)

ci_median_extreme_table(10, 0.95)

ci_median_extreme_table(10, 0.99)

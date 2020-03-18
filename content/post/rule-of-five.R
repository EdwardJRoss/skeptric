# Exact calculation
ci_quantile_extreme <- function(num_samples, nth_smallest=1, nth_largest=1, quantile=0.5) {
    stopifnot(nth_largest + nth_smallest < num_samples)
    stopifnot((0 < quantile) && (quantile < 1))
    # Number of ways all but the nth largest values could be bigger than quantile
    freq_larger <- sum(choose(num_samples, seq(0, nth_largest - 1)))
    # Number of ways all but the nth smallest values could be bigger than quantile
    freq_smaller <- sum(choose(num_samples, seq(0, nth_smallest - 1)))
    1 - (freq_larger * quantile ^ num_samples + freq_smaller * (1 - quantile) ^num_samples)
}

# Test calculation with random sample
sample_quantile_extreme <- function(sample_size, nth_smallest, nth_largest, num_sample=10000, quantile=0.5) {
    distribution <- runif
    distribution_median <- quantile

    samples <- distribution(sample_size * num_sample)
    dim(samples) <- c(sample_size, num_sample)

    sorted_samples <- apply(samples, 2, sort)
    mean((sorted_samples[nth_smallest,] <= distribution_median) &
         (sorted_samples[sample_size+1-nth_largest,] >= distribution_median))
}


# Calculate the number of samples at nth largest to reach at least ci
median_extreme_samples_at_ci <- function(nth, ci=0.9) {
    sample_size <- 2*nth + 1
    while(ci_quantile_extreme(sample_size, nth, nth) < ci) {
        sample_size <- sample_size + 1
    }
    sample_size
}

# Generate a table of confidences
ci_median_extreme_table <- function(n_max=10, ci=0.9) {
    nth_largest <- seq(1, n_max)
    sample_size <- vapply(nth_largest,
                          function(n) median_extreme_samples_at_ci(n, ci),
                          double(1))
    confidence_interval <- mapply(ci_quantile_extreme, sample_size, nth_largest, nth_largest)

    data.frame(nth_largest, sample_size, confidence_interval, ci=ci)
}

# Power depends a lot on the distribution

df <- rbind(
ci_median_extreme_table(20, 0.75),
ci_median_extreme_table(20, 0.9),
ci_median_extreme_table(20, 0.95),
ci_median_extreme_table(20, 0.99)
)

df$confidence_interval = paste0(as.character(df$ci * 100), '%')

library(ggplot2)

ggplot(df, mapping=aes(nth_largest, sample_size, color=confidence_interval, group=confidence_interval)) +
    geom_line() +
    scale_x_continuous(minor_breaks=seq(1, 20)) +
    scale_y_continuous(minor_breaks=seq(0, 100, by=5)) +
    labs(x="nth largest/smallest items", y="Sample Size", color="Confidence Interval",
         title="Median confidence interval between nth extreme items")

ggsave('rule-of-n.png')

# Estimate first quartile?
# For max power do we try to max n_largest + n_smallest?
sample_quantile_extreme(20, 1, 11, 10000, 0.25)

sample_quantile_extreme(20, 2, 0, 10000, 0.25)

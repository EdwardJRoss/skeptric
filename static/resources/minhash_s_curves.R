library(dplyr)
library(tibble)
library(ggformula)
library(forcats)

theme_set(theme_light())

prob <- function(b, r, s) 1 - (1 - s^r) ^b
dprob <- function(b, r, s) r*b * (1 - s^r)^(b-1) * s^(r-1)



# Where the probability increases fastest
prob_threshold <- function(b, r) ((1 - 1/r)) / (b - 1/r)^(1/r)

# Generate all bands/rows that exactly divide 2^7
# You can obviously use other combinations like 5 bands of 25 rows
n <- 7
r <- 2 ^ seq(1, n-1)
b <- 2^n / r

# Our similarities vector
s <- seq(0, 1, by=0.01)

# Put it all into a dataframe for each combination of b with s
df <- tibble(r=rep(r, each=length(s)),
             b=rep(b, each=length(s)),
             s=rep(s, n-1)) %>%
  mutate(p = prob(b, r, s),
         threshold = prob_threshold(b, r),
         rate = dprob(b, r, threshold),
         threshold_value = prob(b, r, threshold),
         label=factor(paste0(b, ' bands of ', r, ' rows') %>% fct_inorder()))


df %>%
  filter(r==16) %>%
  gf_line(p ~ s) +
  theme_light() +
  labs(title='Probability a random pair will be emitted from LSH with 8 bands of 16 rows',
       x='Jaccard Similarity', y = 'Probability')



df %>% 
  gf_line(p ~ s | label) %>%
  labs(title='Probability a random pair will be emitted from LSH',
       x='Jaccard Similarity', y = 'Probability')



################################################################################
## Analysis
################################################################################

# Maximum slope a.k.a "threshold"
max_slope_point <- prob_threshold(b, r)

# As r increases the "threshold" moves quickly to the right
max_slope_point

# (1/b)^(1/r) is a pretty good approximation, especially as r becomes larger
max_slope_point / (1/b)^(1/r)



# For large b and small r the maximum is ~ 1-exp(-1) ~ 0.63
# Here it's between ~0.4 and ~0.75

prob(b, r, max_slope_point)

# The derivative is around  exp(-1) * r / (max_slope_point)

dprob(b, r, max_slope_point)

# It's a reasonable approximation for middling r and b
dprob(b, r, max_slope_point) / (exp(-1) * r / max_slope_point)

# Polynomial expansion around max slope point is
# Prob(s+x) = Prob(s) + x dProb(s) + O(x^3)
#           ~ 1 + exp(-1) (r * x / s - 1)
# So s/r is an approximate width for the bands at max slope point
# This works pretty poorly for small b, because approximation (1 + 1/b)^(1/b) breaks down

width <- max_slope_point / dprob(b, r, max_slope_point)

max_slope_point + width

prob(b, r, min(1, max_slope_point + width))

prob(b, r, max(0, max_slope_point - width))

##
# Looking at fixed threshold point

b <- c(4, 16, 256)
r <- c(4, 8, 16)

find_prob <- function(b, r, p) {
  uniroot(function(x) prob(b, r, x) - p, c(0,1))$root[[1]]
}

# Roughly halves with each step
c(find_prob(4, 4, 0.99) - 1/sqrt(2), 1 /sqrt(2) - find_prob(4, 4, 0.01))
c(find_prob(16, 8, 0.99) - 1/sqrt(2), 1 /sqrt(2) - find_prob(16, 8, 0.01))
c(find_prob(256, 16, 0.99) - 1/sqrt(2), 1 /sqrt(2) - find_prob(256, 16, 0.01))
c(find_prob(65536, 32, 0.99) - 1/sqrt(2), 1 /sqrt(2) - find_prob(65536, 32, 0.01))

n <- length(b) + 1

(1/b)^(1/r)


df <- tibble(r=rep(r, each=length(s)),
             b=rep(b, each=length(s)),
             s=rep(s, n-1)) %>%
  mutate(p = 1 - (1-s^r)^b,
         threshold = ((1 - 1/r) / (b - 1/r))^(1/r),
         rate = r * b * (1 - threshold^r)^(b-1) * threshold^(r-1),
         threshold_value = 1 - (1 - threshold^r)^b,
         label=factor(paste0(b, ' bands of ', r, ' rows') %>% fct_inorder()))


df %>% 
  gf_line(p ~ s, col=~label) %>%
  gf_point(threshold_value ~ threshold) %>%
  gf_abline(slope=~rate, intercept =~ threshold_value - rate * threshold, linetype="dashed", col=~label) +
  labs(title='Probability a random pair will be emitted from LSH',
       x='Jaccard Similarity', y = 'Probability')


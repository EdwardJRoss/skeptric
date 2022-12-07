factorial(4)

choose(10, 5) / 2^10

p <- function(N, m, p=0.5) {
  choose(N, m) * p^m * (1-p)^(N-m)
}


library(ggformula)

m <- seq(1, 100)
n <- seq(1, 400)
t <- c(0.2, 0.5, 0.8)

df <- tibble(
  N=c(rep(100,100), rep(400, 400)),
  m=c(m, n),
  p=0.5
) %>%
  mutate(prob=choose(N, m) * (p^m) * (1-p)^(N-m),
         frac=m/N)

df %>%
  mutate(N=as.factor(N)) %>%
  gf_point(prob ~ frac, col=~N) +
  labs(y='Probability', x='Measured Proportion of Successes',
       title='As size quadruples the spread halves',
       col='Sample Size') 

x <- seq(0, 1, by=0.01)
a <- c(4, 31)
b <- c(11, 101)

df <- tibble(
  p = rep(x, 2),
  a = rep(a, each=length(x)),
  b = rep(b, each=length(x)),
) %>%
  mutate(prob=dbeta(p, a, b),
         label = paste0(a, ';', b))


df %>% gf_line(prob~p, col=~label)


x <- seq(0, 1, by=0.01)
a <- c(0.1, 1, 2.5, 10)
b <- c(0.1, 1, 2.5, 10)

df <- expand.grid(x=x, a=a, b=b) %>%
  mutate(p = dbeta(x, a, b),
         label = paste0("a = ", a, ", b = ", b)) %>%
  as_tibble() 

df %>% 
  gf_line(p ~ x) + 
  facet_grid(b ~ a, labeller=label_both) +
  scale_x_continuous(name ="p",
                     limits=c(0, 1),
                     breaks=seq(0, 1, by=0.5)) +
  scale_y_continuous(name="Probability density",
                     limits=c(0, 5),
                     breaks = c(0, 5)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(title="Beta distribution")


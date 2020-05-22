library(rwalkr)
library(ggplot2)
library(dplyr)
library(ggformula)
library(lubridate)

library(sugrrants)

start_date <- as.Date("2020-03-01")
end_date <- as.Date("2020-04-16")


sensors <- rwalkr::pull_sensor()

ped_walk <- melb_walk(start_date, end_date)

ped_walk %>% group_by(Sensor) %>% summarise(Count = sum(Count)) %>% arrange(desc(Count))

 ped_walk <- melb_walk_fast(year=c('2019', '2020'), sensor='Flinders St-Elizabeth St (East)')


ped_walk %>% 
  filter(Date_Time >= '2020-03-01') %>%
  gf_line(Count ~ Date_Time,  col=~year(Date_Time))

ped_walk %>%
  filter(Date_Time >= '2020-03-01') %>%
  gf_point(Count ~ Date_Time) +
  geom_hline(yintercept=861, col='blue', size=2) +
  labs(title='A constant model')
  


ped_walk %>%
  filter(Date >= as.Date('2020-02-01')) %>%
  ggplot(aes(x = Time, y = Count)) +
  geom_line() +
  facet_calendar(~ Date) + # a variable contains dates
  theme_bw() +
  theme(legend.position = "bottom")

seq(1, 100)

nrow(ped_walk)

ped_walk %>% summarise(n())

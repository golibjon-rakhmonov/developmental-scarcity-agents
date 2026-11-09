# Calibration of version 0

These runs were made before the first commit of this repository, with earlier states of the code,
on seeds 900 and above. They are added here after the fact, so that the path to the version 0
settings can be checked. The code that produced them was not kept; the settings of each run are in
its `.json` file. Logs that survived are `cal*.log`.

| Run | Seeds | What changed | Wellbeing, last 10% of childhood | Collapses per 1,000 steps |
|---|---|---|---|---|
| cal1 | 901 | first world: 15x15 map, 12 food patches, 3M childhood | 0.75-0.77 | about 7 |
| cal2 | 902-903 | 11x11 map, 8 patches; learning rate 3e-4 and 1e-3 | 0.83-0.85 | 2.5-3.3 |
| cal3 | 904 | 8M childhood steps, learning rate 1e-3 | 0.84-0.88 | 1.3-2.0 |
| cal4 | 905 | short tries of single changes (only the log survived) | - | - |
| cal5 | 906 | gamma as a setting (0.995), 4M childhood | 0.83-0.88 | 1.8-2.5 |
| cal6 | 907 | critic check (only the log survived) | - | - |
| cal7 | 908 | rewards scaled by (1 - gamma), learning rate 3e-4, 2M childhood | 0.965 | 0.08 |
| e2e | 950-952 | the whole pipeline on short lives (200k childhood) | test of the scripts | - |
| smoke | 900, 999 | first smoke tests | - | - |

What these runs already showed about the hypothesis: in cal3 the rich-childhood agent carried more
food than the scarce one in the frozen check, which is why the pilot hypothesis was two-sided. In
e2e (after the reward fix, but with a 200k childhood) the rich agents spent less daytime on cold
ground than the scarce ones, while collapses and food carried did not differ.

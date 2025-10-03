Graphics cards (GPUs) are powerful tools in classical simulations of amplifiers, couplers, etc.

GPUs allow parallel sweeps in parameter space -> speedups up to 2 orders of magnitude

Not just convenient, but a matter of necessity for certain useful computations, like noise studies that require many identical simulation shots

This Github repo has tools that can translate familiar equations of motion into GPU-readable code, along with useful analysis code.


### Todos/buglist

- incorrectly declaring variables should be caught, current error is confusing ("not supported in C")
- ~~arbitrary number of sweeps/automatic reshaping~~
- ~~decimation~~
- incorrect order of incidices on multi-parameter sweeps


## Near-term list
- save to plottr file for artibrary slicing with GUI
- add statistics for sweeps, like number of derivatives calculated
- add constant pulse and chirped pulse
- add multiple pulses for one input

## Wishlist
- generate cumulant EoMs
- allow sweep of initial conditions

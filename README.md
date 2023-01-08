A Julia implementation of all plots from [Reinforcement Learning: An Introduction 2nd Edition](http://incompleteideas.net/book/the-book-2nd.html). Based on code from Shangtong Zhang's [original](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) python3 implementation.
# How to run
Start Julia, activate the project, `cd` into the right chapter and run the `.jl` file corresponding to your problem.

```sh-session
$ julia
$ julia> ]
$ (@v1.8) pkg> activate . # press backspace after this to return to Julia REPL
$ julia> cd("chapter06")
$ julia> include("random_walk.jl")
```
module nn

"""
Very basic perceptron neural network 
    --> Supports arbitrary layered network customization
    --> Supports per layer activation function customization
    --> MSE cost function only

TODO : 
    --> Extend to Convolutional layer support
    --> Arbitrary loss function
    --> Use AutoDiff
"""

export sigmoid, tanh, sin, relu, Network, train!, forward!, backprop!, cost, init

# ACTIVATION FUNCTIONS (with derivatives)

function sigmoid(x, derivative::Bool)
    if derivative
        return x .* (1 .- x)
    end
    1 ./ (1 .+ exp.(-x))
end

function tanh(x, derivative::Bool)
    if derivative
        return 1.0 .- x .^ 2
    end
    @. Base.tanh(x)
end

function identity(x, derivative::Bool)
    if derivative
        return ones(size(x))
    end
    x
end

function sin(x, derivative::Bool)
    if derivative
        return @. cos(asin(x))
    end
    @. Base.sin(x)
end

function relu(x, derivative::Bool)
    if derivative
        return 0.0 .+ 1.0 .* (x .>= 0.0)
    end
    x .* (x .>= 0.0) .+ 0.0
end

# NETWORK STRUCTURE

mutable struct Network
    a::Array{Any,1}
    W::Array{Any,1}
    b::Array{Any,1}
    act::Array{Function,1}
end


# INITIALIZATION : random weights and null biases / sigmoid as default activation functions

function init(sizes::Vector{Int}, activations=[])

    W = []
    b = []
    act = activations
    if length(act) != length(sizes)
        act = [sigmoid for i in 1:length(sizes)]
    end
    for i in 2:length(sizes)
        push!(W, randn((sizes[i-1], sizes[i])))
        push!(b, zeros((1, sizes[i])))
    end
    Network([], W, b, act)
end



# PUSH FORWARD

function forward!(nn::Network, X)
    nn.a = [X']
    for i in 1:length(nn.W)
        push!(nn.a, nn.act[i](nn.a[i] * nn.W[i] .+ nn.b[i], false))
    end
    nn.a[end]
end

# BACK-PROPAGATION ALGORITHM

function backprop!(nn, X, y, rate)
    result = forward!(nn, X)
    error = y' - result
    delta = error .* nn.act[end](result, true)

    nn.W[end] += rate * transpose(nn.a[end-1]) * delta
    nn.b[end] .+= rate * mean(delta)

    depth = length(nn.W)

    for i in depth-1:-1:1

        error = delta * transpose(nn.W[i+1])
        delta = error .* nn.act[i](nn.a[i+1], true)

        nn.W[i] += rate * transpose(nn.a[i]) * delta
        nn.b[i] .+= rate * mean(delta)
    end
end

mean(x) = sum(x) / length(x)

# MSE COST FUNCTION
cost(nn, X, y) = 0.5 * mean((forward!(nn, X) .- y) .^ 2)

# TRAINING ALGORITHM
function train!(nn, batch_X, batch_y, n_epoch, rate)

    ec = []
    c = []
    for i in 1:n_epoch
        ec = []
        for (X, y) in zip(batch_X, batch_y)
            backprop!(nn, X, y, rate)
            push!(ec, cost(nn, X, y))
        end
        push!(c, ec[end])

        println("Epoch $(i) : error : $(mean(c))")
    end
    return c
end

end
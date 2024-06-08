include("nn.jl")
using .nn
using Plots

plot()

begin # AND gate
    network = init([2, 1], [nn.sigmoid, relu])

    batch_X = [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ]

    batch_y = []
    for x in batch_X
        push!(batch_y, [x[1] & x[2]])
    end

    err_hist = train!(network, batch_X, batch_y, 1000, 0.8)


    println(repeat("-", 30))
    for x in batch_X
        result = forward!(network, x)
        println("$(x[1]) & $(x[2]) = $(x[1] & x[2]) / ($(round(result[1], digits=5))) \t| accuracy : $(round(100*(1.0 - abs(result[1]-(x[1] & x[2]))), digits=3))%")
    end

    plot!(err_hist, label="AND")
end

begin # XOR gate
    network = init([2, 1], [nn.sigmoid, relu])

    batch_X = [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ]

    batch_y = []
    for x in batch_X
        push!(batch_y, [x[1]^x[2]])
    end

    err_hist = train!(network, batch_X, batch_y, 1000, 0.8)


    println(repeat("-", 30))
    for x in batch_X
        result = forward!(network, x)
        println("$(x[1]) ^ $(x[2]) = $(x[1] ^ x[2]) / ($(round(result[1], digits=5))) \t| accuracy : $(round(100*(1.0 - abs(result[1]-(x[1] ^ x[2]))), digits=3))%")
    end

    plot!(err_hist, label="XOR")
end


begin # OR gate
    network = init([2, 1], [nn.sigmoid, relu])

    batch_X = [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ]

    batch_y = []
    for x in batch_X
        push!(batch_y, [x[1] | x[2]])
    end

    err_hist = train!(network, batch_X, batch_y, 1000, 0.8)


    println(repeat("-", 30))
    for x in batch_X
        result = forward!(network, x)
        println("$(x[1]) | $(x[2]) = $(x[1] | x[2]) / ($(round(result[1], digits=5))) \t| accuracy : $(round(100*(1.0 - abs(result[1]-(x[1] | x[2]))), digits=3))%")
    end

    plot!(err_hist, label="OR")
end

plot!(yscale=:log10)